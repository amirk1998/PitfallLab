// ============================================================================
// SECTION 1: SCENARIO
// ============================================================================
// Fast CSV Reader (standard library only).
//
// Read a very large CSV file into memory as fast as possible, without any
// third-party module and without converting the file to another format.
//
// Context
// -------
// A service (or data-prep tool) must load a multi-hundred-MB / multi-GB CSV
// and then use the values in ordinary Go code. Load time dominates the job,
// so it is the number to minimise. The file may come from an untrusted source.
//
// Requirements
// ------------
//   - Input : path to a UTF-8 CSV file with a header row (RFC 4180 style:
//     fields may be quoted; quoted fields may contain commas, doubled quotes
//     and even newlines). CRLF or LF line endings, optional UTF-8 BOM.
//   - Output: a column-oriented Table: []int64 for int columns, []float64 for
//     float columns, []string for text columns. Only requested columns are
//     converted.
//   - Errors: missing file / unknown column / bad number / ragged row /
//     unterminated quote / oversized record -> wrapped sentinel errors that
//     work with errors.Is, never a silent wrong answer, never a panic.
//   - Empty float cells become NaN; empty int cells are an error.
//
// Assumptions
// -----------
//   - Every record has the same number of fields as the header.
//   - The input follows RFC 4180: a '"' inside an UNQUOTED field is rejected
//     (this is also what makes quote counting a sound record delimiter).
//   - The result fits in RAM (for larger data, stream chunks instead).
//   - Benchmark schema: id:int, amount:float, category:str (the file also has
//     user_id, ts and note columns that are never converted).
//   - Two synthetic profiles are used:
//     clean : no quote characters anywhere.
//     messy : ~2% of rows have a quoted "note" holding commas, doubled quotes
//     or embedded newlines.
//
// Goal
// ----
// One linear pass, O(n) time, O(n) compact space, near-zero allocation per
// row, and optional multi-core scaling with deterministic output.
//
// Run:            go run ./001_fast_csv_reader [max_rows]     (from the go/ dir)
// Verify races:   go run -race ./001_fast_csv_reader 100000
// ============================================================================
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Shared types (used by all three approaches)
// ---------------------------------------------------------------------------

// Kind is the target type of a requested column.
type Kind int

// Supported column kinds.
const (
	KindInt Kind = iota
	KindFloat
	KindString
)

func (k Kind) String() string {
	switch k {
	case KindInt:
		return "int"
	case KindFloat:
		return "float"
	case KindString:
		return "string"
	}
	return "invalid"
}

// ColumnSpec names a CSV column and the type it must be converted to.
type ColumnSpec struct {
	Name string
	Kind Kind
}

// Schema is an ordered list of requested columns.
type Schema []ColumnSpec

// Column holds the values of one column; only the slice matching the kind of
// its ColumnSpec is used.
type Column struct {
	Ints   []int64
	Floats []float64
	Strs   []string
}

// Table is a column store: Cols[i] holds the data for Schema[i].
type Table struct {
	Schema Schema
	Cols   []Column
}

// Sentinel errors; callers test them with errors.Is.
var (
	ErrUnknownColumn     = errors.New("unknown column")
	ErrDuplicateColumn   = errors.New("duplicate column in schema")
	ErrEmptyInput        = errors.New("empty input: no header row")
	ErrFieldCount        = errors.New("wrong number of fields")
	ErrBadValue          = errors.New("invalid value")
	ErrUnterminatedQuote = errors.New("unterminated quoted field")
	ErrBareQuote         = errors.New("bare quote in unquoted field or text after closing quote")
	ErrRecordTooLarge    = errors.New("record exceeds MaxRecordBytes")
	ErrInvalidOptions    = errors.New("invalid options")
)

// benchSchema is the schema used by the benchmark for all approaches.
var benchSchema = Schema{
	{Name: "id", Kind: KindInt},
	{Name: "amount", Kind: KindFloat},
	{Name: "category", Kind: KindString},
}

var csvHeader = []string{"id", "user_id", "amount", "category", "ts", "note"}

// columnIndexes maps every schema column to its position in the header.
func columnIndexes(header []string, schema Schema) ([]int, error) {
	idx := make([]int, len(schema))
	for i, s := range schema {
		j := slices.Index(header, s.Name)
		if j < 0 {
			return nil, fmt.Errorf("%w: %q (file has %q)", ErrUnknownColumn, s.Name, header)
		}
		idx[i] = j
	}
	return idx, nil
}

// appendCell converts one string cell with the strconv package (used by the
// common approach and by the reference; the optimal one has its own parsers).
func appendCell(c *Column, s ColumnSpec, cell string) error {
	switch s.Kind {
	case KindInt:
		v, err := strconv.ParseInt(cell, 10, 64)
		if err != nil {
			return fmt.Errorf("column %q: %w: %q", s.Name, ErrBadValue, cell)
		}
		c.Ints = append(c.Ints, v)
	case KindFloat:
		if strings.TrimSpace(cell) == "" {
			c.Floats = append(c.Floats, math.NaN())
			return nil
		}
		v, err := strconv.ParseFloat(cell, 64)
		if err != nil {
			return fmt.Errorf("column %q: %w: %q", s.Name, ErrBadValue, cell)
		}
		c.Floats = append(c.Floats, v)
	default:
		c.Strs = append(c.Strs, cell)
	}
	return nil
}

// ============================================================================
// SECTION 2: APPROACH 1 - THE WRONG WAY
// ============================================================================
// What the author was thinking:
//
//	"A CSV is just text. Read the whole file, split on newlines, split every
//	 line on commas, and convert. strconv errors are noise, skip them."
//
// Why it is wrong:
//  1. CORRECTNESS: quoted fields may contain commas, quotes and newlines.
//     Splitting on "," and "\n" cuts records in the wrong places. The shredded
//     rows have too few fields, so row[j] PANICS (index out of range). Had it
//     not panicked, the next flaw would have hidden the damage:
//  2. IGNORED ERRORS: `v, _ := strconv.ParseInt(...)` turns every unparsable
//     cell into 0 without a word. Wrong numbers flow into the program silently.
//     A CRLF file leaves a trailing '\r' on the last column, which triggers the
//     same silent zero.
//  3. MEMORY: os.ReadFile holds the file once; string(data) copies it a second
//     time; strings.Split builds a slice of every line; strings.Split per line
//     allocates a []string for every row. Garbage is produced at the rate of
//     the input size, several times over.
//  4. SAFETY: no size limit at all. A hostile multi-GB upload is read straight
//     into RAM (denial of service), and an unknown column name panics instead
//     of returning an error. No BOM handling either.
//
// Failure mode: on clean data it is merely slow and allocation heavy; on data
// with quoted fields it panics or returns wrong numbers.
// Complexity: time O(n) with a large constant, space about 2x file + rows.
// ============================================================================

// ReadWrong is the mistake people actually make. Do not use it.
func ReadWrong(path string, schema Schema) (Table, error) {
	data, err := os.ReadFile(path) // whole file in memory, no size limit
	if err != nil {
		return Table{}, err
	}
	lines := strings.Split(string(data), "\n") // copy #2 plus a slice of lines
	header := strings.Split(lines[0], ",")
	idx := make([]int, len(schema))
	for i, s := range schema {
		idx[i] = slices.Index(header, s.Name) // -1 when missing => panic below
	}
	cols := make([]Column, len(schema)) // no preallocation: repeated regrowth
	for _, line := range lines[1:] {
		if line == "" {
			continue
		}
		row := strings.Split(line, ",") // one []string allocation per row
		for i, s := range schema {
			cell := row[idx[i]] // panics on shredded rows
			switch s.Kind {
			case KindInt:
				v, _ := strconv.ParseInt(cell, 10, 64) // error ignored: silent 0
				cols[i].Ints = append(cols[i].Ints, v)
			case KindFloat:
				v, _ := strconv.ParseFloat(cell, 64) // error ignored: silent 0
				cols[i].Floats = append(cols[i].Floats, v)
			default:
				cols[i].Strs = append(cols[i].Strs, cell)
			}
		}
	}
	return Table{Schema: schema, Cols: cols}, nil
}

// ============================================================================
// SECTION 3: APPROACH 2 - THE COMMON WAY
// ============================================================================
// Why most developers write this:
//
//	encoding/csv is the idiomatic answer and Reader.ReadAll() gives a tidy
//	[][]string. It is correct: quotes, embedded newlines and CRLF are handled,
//	ragged rows are rejected, errors are returned.
//
// What it leaves on the table:
//   - ReadAll materialises EVERY column of EVERY row as strings first (about
//     one string plus one []string allocation per record), then a second
//     pass converts the few columns we need. Peak memory is a multiple of the
//     file size; the GC has a large live heap to scan while we work.
//   - The general-purpose tokenizer copies each record into its own buffer and
//     checks quoting rules for every byte, even for the majority of lines that
//     contain no quote at all.
//   - Numbers are parsed from freshly allocated strings.
//   - No BOM handling (a BOM ends up inside the first header name), no limit
//     on record size, single core only.
//
// Complexity: time O(n * c) with a moderate constant, space O(n * c) strings.
// ============================================================================

// ReadCommon is the idiomatic encoding/csv + ReadAll solution.
func ReadCommon(path string, schema Schema) (Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll() // all columns, all rows, as strings
	if err != nil {
		return Table{}, err
	}
	if len(records) == 0 {
		return Table{}, ErrEmptyInput
	}
	idx, err := columnIndexes(records[0], schema)
	if err != nil {
		return Table{}, err
	}
	out := Table{Schema: schema, Cols: make([]Column, len(schema))}
	for _, rec := range records[1:] {
		for i, s := range schema {
			if err := appendCell(&out.Cols[i], s, rec[idx[i]]); err != nil {
				return Table{}, err
			}
		}
	}
	return out, nil
}

// ============================================================================
// SECTION 4: APPROACH 3 - THE OPTIMAL, PRODUCTION-GRADE WAY
// ============================================================================
// Key insight:
//
//	A newline is a record boundary exactly when it is NOT inside a quoted
//	field, and in RFC 4180 that is decidable by parity: the number of '"'
//	bytes before it is even (a quote either opens/closes a field or is doubled,
//	so pairs cancel). Parity is computed by bytes.Count, which is SIMD-fast.
//	That single fact lets us
//	  (a) cut the input into big chunks that end on record boundaries,
//	  (b) parse every chunk independently, hence in parallel,
//	  (c) use a very cheap path for the many lines that contain no quote.
//
// Design:
//  1. Chunker  - reads ~1 MiB blocks with io.ReadFull, cuts each block at the
//     last newline outside quotes (safeCut) and carries the tail into the
//     next block. Chunk buffers are recycled through a sync.Pool, so the
//     steady-state allocation of the reader is ~zero. A newline byte can
//     never be part of a multi-byte UTF-8 sequence, so cutting is safe.
//  2. Per record, two paths:
//     FAST (no '"' in the line): one bytes.Count(",") validates the field
//     count, then bytes.IndexByte hops from comma to comma and only the
//     REQUESTED fields are converted, straight from []byte. No []string, no
//     per-row allocation. The scan stops after the last wanted column.
//     SAFE (line has quotes): the record is extended over following lines
//     until its quote count is even, then a small RFC 4180 state machine
//     splits it. Escaped quotes allocate; everything else is a sub-slice.
//  3. Parsers - hand-rolled int parser for <= 18 digits (no overflow check
//     needed, no allocation); float via strconv.ParseFloat(string(b)), which
//     does not allocate for short inputs because the string never escapes.
//     Text columns are interned in a bounded map (InternLimit): a repeated
//     category costs one map lookup instead of one allocation, and a hostile
//     file with millions of distinct values cannot grow the map without bound.
//  4. Storage  - every chunk is parsed into its own typed slices, sized from
//     the newline count of the chunk (an upper bound on rows), so appends
//     never regrow; the parts are joined once at the end into exactly-sized
//     columns. That avoids the ~4x garbage that repeated append doubling of
//     one giant slice would create.
//  5. Concurrency - with Workers > 1 a producer goroutine feeds chunks to N
//     worker goroutines; each parses into its own partial columns; the caller
//     collects partials strictly in chunk order, so the output (and the first
//     reported error) is deterministic and identical to the sequential run.
//     A token channel caps in-flight chunks at 2*Workers: memory stays bounded
//     even when one chunk is slow. Every goroutine stops on ctx cancellation
//     or on the first error, and ReadOptimal waits for all of them before
//     returning: no leaks, and the file is closed after the last reader.
//     Workers == 1 runs in the calling goroutine with zero goroutine overhead.
//  6. Hardening for untrusted input - MaxRecordBytes bounds the buffer when a
//     quote is never closed (otherwise one stray '"' would buffer the whole
//     file); errors quote cell text with %q so control characters cannot
//     forge log lines; strict RFC 4180 rejects ambiguous quoting instead of
//     guessing; nothing panics on malformed data.
//
// Complexity: time O(n) total (each byte is scanned a small constant number of
// times), space O(result) + O(Workers * ChunkBytes) temporary.
//
// Trade-offs / when NOT to use this:
//   - The whole selected data set is held in RAM. If it does not fit, feed the
//     chunks to a streaming consumer instead of appending them.
//   - Strict RFC 4180 only: no lazy quotes, no custom delimiters (both are
//     small extensions, but every option weakens the parity argument).
//   - Number parsing is strict (no surrounding spaces, no thousands
//     separators, no locale). Clean the data upstream or extend parseInt.
//   - If the same data is read many times, convert once to a binary format
//     (or keep a gob/mmap cache); no CSV parser beats not parsing CSV.
//   - I/O bound cases (cold cache on a slow disk or network mount) will not
//     speed up with more workers.
// ============================================================================

// Options tunes ReadOptimal. Zero values pick production defaults.
type Options struct {
	ChunkBytes     int // bytes read per block (default 1 MiB, min 4 KiB)
	Workers        int // parsing goroutines; 1 = sequential (default min(GOMAXPROCS, 8))
	MaxRecordBytes int // cap for one (possibly multi-line) record (default 16 MiB)
	InternLimit    int // max distinct strings interned per text column and worker (default 4096)
}

func (o Options) normalize() (Options, error) {
	if o.ChunkBytes == 0 {
		o.ChunkBytes = 1 << 20
	}
	if o.Workers == 0 {
		o.Workers = min(runtime.GOMAXPROCS(0), 8)
	}
	if o.MaxRecordBytes == 0 {
		o.MaxRecordBytes = 16 << 20
	}
	if o.InternLimit == 0 {
		o.InternLimit = 4096
	}
	switch {
	case o.ChunkBytes < 4096:
		return o, fmt.Errorf("%w: ChunkBytes must be >= 4096", ErrInvalidOptions)
	case o.Workers < 1:
		return o, fmt.Errorf("%w: Workers must be >= 1", ErrInvalidOptions)
	case o.MaxRecordBytes < 1:
		return o, fmt.Errorf("%w: MaxRecordBytes must be positive", ErrInvalidOptions)
	case o.InternLimit < 0:
		return o, fmt.Errorf("%w: InternLimit must be >= 0", ErrInvalidOptions)
	}
	return o, nil
}

var (
	bom       = []byte{0xEF, 0xBB, 0xBF}
	quoteByte = []byte{'"'}
	commaByte = []byte{','}
)

// --- chunker ----------------------------------------------------------------

// chunk is a block of complete records.
type chunk struct {
	data   []byte
	buf    *[]byte // pool handle to return after use (nil if not pooled)
	offset int64   // file offset of data[0], for error messages
	seq    int
}

type chunker struct {
	r      io.Reader
	size   int
	maxRec int
	tail   []byte // unfinished record carried to the next block
	offset int64
	eof    bool
	first  bool
	pool   sync.Pool
}

func newChunker(r io.Reader, o Options) *chunker {
	return &chunker{r: r, size: o.ChunkBytes, maxRec: o.MaxRecordBytes, first: true}
}

func (c *chunker) getBuf(need int) *[]byte {
	if v := c.pool.Get(); v != nil {
		if b := v.(*[]byte); cap(*b) >= need {
			*b = (*b)[:need]
			return b
		}
	}
	b := make([]byte, need)
	return &b
}

func (c *chunker) release(ch chunk) {
	if ch.buf != nil {
		c.pool.Put(ch.buf)
	}
}

// safeCut returns the index just after the last newline that lies outside a
// quoted field, or 0 when there is none. data must start on a record boundary
// (even quote parity). O(len(data)): each step counts a disjoint segment.
func safeCut(data []byte) int {
	cut := bytes.LastIndexByte(data, '\n')
	if cut < 0 {
		return 0
	}
	quotes := bytes.Count(data[:cut], quoteByte)
	for cut >= 0 && quotes%2 == 1 { // odd parity: this newline is inside quotes
		prev := bytes.LastIndexByte(data[:cut], '\n')
		quotes -= bytes.Count(data[prev+1:cut], quoteByte)
		cut = prev
	}
	if cut < 0 {
		return 0
	}
	return cut + 1
}

// next returns the next block of complete records, or io.EOF.
func (c *chunker) next() (chunk, error) {
	for {
		if c.eof && len(c.tail) == 0 {
			return chunk{}, io.EOF
		}
		bp := c.getBuf(len(c.tail) + c.size)
		buf := *bp
		copy(buf, c.tail)
		n := 0
		if !c.eof {
			var err error
			n, err = io.ReadFull(c.r, buf[len(c.tail):])
			if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				c.release(chunk{buf: bp})
				return chunk{}, fmt.Errorf("read: %w", err)
			}
			c.eof = err != nil // a short read means end of file
		}
		data := buf[:len(c.tail)+n]
		c.tail = c.tail[:0]
		if c.first {
			c.first = false
			if bytes.HasPrefix(data, bom) {
				data = data[len(bom):]
				c.offset += int64(len(bom))
			}
		}
		if c.eof { // last block: everything left is complete (or malformed)
			if len(data) == 0 {
				c.release(chunk{buf: bp})
				return chunk{}, io.EOF
			}
			ch := chunk{data: data, buf: bp, offset: c.offset}
			c.offset += int64(len(data))
			return ch, nil
		}
		cut := safeCut(data)
		if cut == 0 { // a single record spans the whole block: keep growing
			if len(data) > c.maxRec {
				c.release(chunk{buf: bp})
				return chunk{}, fmt.Errorf("byte offset %d: %w (%d bytes)", c.offset, ErrRecordTooLarge, len(data))
			}
			c.tail = append(c.tail, data...)
			c.release(chunk{buf: bp})
			continue
		}
		c.tail = append(c.tail, data[cut:]...)
		ch := chunk{data: data[:cut], buf: bp, offset: c.offset}
		c.offset += int64(cut)
		return ch, nil
	}
}

// --- record splitting -------------------------------------------------------

func lineEnd(data []byte, pos int) int {
	if i := bytes.IndexByte(data[pos:], '\n'); i >= 0 {
		return pos + i + 1
	}
	return len(data)
}

func trimEOL(b []byte) []byte {
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	if n := len(b); n > 0 && b[n-1] == '\r' {
		b = b[:n-1]
	}
	return b
}

// nextRecord returns the record starting at pos without its line ending, the
// position after it, and whether it contains a quote. A record with an odd
// quote count continues over the following lines.
func nextRecord(data []byte, pos int) (rec []byte, next int, quoted bool, err error) {
	end := lineEnd(data, pos)
	if bytes.IndexByte(data[pos:end], '"') >= 0 {
		quoted = true
		q := bytes.Count(data[pos:end], quoteByte)
		for q%2 == 1 {
			if end >= len(data) {
				return nil, end, true, ErrUnterminatedQuote
			}
			nl := lineEnd(data, end)
			q += bytes.Count(data[end:nl], quoteByte)
			end = nl
		}
	}
	return trimEOL(data[pos:end]), end, quoted, nil
}

// splitQuoted splits one RFC 4180 record into fields, appending to out. Fields
// are sub-slices of rec except for fields containing escaped quotes.
func splitQuoted(rec []byte, out [][]byte) ([][]byte, error) {
	n := len(rec)
	i := 0
	for {
		if i < n && rec[i] == '"' { // quoted field
			i++
			start := i
			var unesc []byte
			for {
				j := bytes.IndexByte(rec[i:], '"')
				if j < 0 {
					return out, ErrUnterminatedQuote
				}
				i += j
				if i+1 < n && rec[i+1] == '"' { // doubled quote => literal quote
					unesc = append(unesc, rec[start:i+1]...)
					i += 2
					start = i
					continue
				}
				if unesc == nil {
					out = append(out, rec[start:i])
				} else {
					out = append(out, append(unesc, rec[start:i]...))
				}
				i++ // closing quote
				break
			}
			if i == n {
				return out, nil
			}
			if rec[i] != ',' {
				return out, ErrBareQuote
			}
			i++
			if i == n { // trailing comma => empty last field
				return append(out, nil), nil
			}
			continue
		}
		j := bytes.IndexByte(rec[i:], ',') // unquoted field
		end := n
		if j >= 0 {
			end = i + j
		}
		if bytes.IndexByte(rec[i:end], '"') >= 0 {
			return out, ErrBareQuote
		}
		out = append(out, rec[i:end])
		if j < 0 {
			return out, nil
		}
		i = end + 1
		if i == n {
			return append(out, nil), nil
		}
	}
}

// --- conversion -------------------------------------------------------------

func parseInt(b []byte) (int64, bool) {
	if len(b) == 0 {
		return 0, false
	}
	if len(b) <= 18 { // at most 18 digits always fit in int64: no overflow checks
		i := 0
		neg := false
		switch b[0] {
		case '-':
			neg, i = true, 1
		case '+':
			i = 1
		}
		if i == len(b) {
			return 0, false
		}
		var n int64
		for ; i < len(b); i++ {
			d := b[i] - '0'
			if d > 9 {
				return 0, false
			}
			n = n*10 + int64(d)
		}
		if neg {
			n = -n
		}
		return n, true
	}
	v, err := strconv.ParseInt(string(b), 10, 64) // long numbers: let strconv check range
	return v, err == nil
}

func parseFloat(b []byte) (float64, bool) {
	if len(b) == 0 {
		return math.NaN(), true
	}
	v, err := strconv.ParseFloat(string(b), 64) // short strings do not escape: no alloc
	if err != nil {
		if len(bytes.TrimSpace(b)) == 0 {
			return math.NaN(), true
		}
		return 0, false
	}
	return v, true
}

// --- chunk parser -----------------------------------------------------------

type wantedCol struct {
	fileIdx int // position in the CSV record
	dst     int // position in Schema / Table.Cols
	kind    Kind
	name    string
}

type plan struct {
	ncols  int         // fields per record in the file
	nout   int         // columns in the output Table
	wanted []wantedCol // sorted by fileIdx, unique
}

func buildPlan(header []string, schema Schema) (*plan, error) {
	if len(schema) == 0 {
		return nil, fmt.Errorf("%w: schema must name at least one column", ErrInvalidOptions)
	}
	seen := make(map[string]bool, len(schema))
	p := &plan{ncols: len(header), nout: len(schema)}
	for i, s := range schema {
		if seen[s.Name] {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateColumn, s.Name)
		}
		seen[s.Name] = true
		if s.Kind < KindInt || s.Kind > KindString {
			return nil, fmt.Errorf("%w: column %q has invalid kind", ErrInvalidOptions, s.Name)
		}
		j := slices.Index(header, s.Name)
		if j < 0 {
			return nil, fmt.Errorf("%w: %q (file has %q)", ErrUnknownColumn, s.Name, header)
		}
		p.wanted = append(p.wanted, wantedCol{fileIdx: j, dst: i, kind: s.Kind, name: s.Name})
	}
	slices.SortFunc(p.wanted, func(a, b wantedCol) int { return a.fileIdx - b.fileIdx })
	return p, nil
}

// parser owns per-goroutine scratch state; it is never shared between goroutines.
type parser struct {
	plan     *plan
	interned []map[string]string // per wanted column (nil unless text)
	limit    int
	fields   [][]byte
}

func newParser(pl *plan, internLimit int) *parser {
	p := &parser{plan: pl, limit: internLimit, interned: make([]map[string]string, len(pl.wanted))}
	for i, w := range pl.wanted {
		if w.kind == KindString {
			p.interned[i] = make(map[string]string)
		}
	}
	return p
}

func (p *parser) intern(wi int, b []byte) string {
	m := p.interned[wi]
	if s, ok := m[string(b)]; ok { // lookup with string(b) does not allocate
		return s
	}
	s := string(b)
	if len(m) < p.limit {
		m[s] = s
	}
	return s
}

func valueErr(w *wantedCol, b []byte) error {
	if len(b) > 32 {
		b = b[:32]
	}
	return fmt.Errorf("column %q: %w: %q", w.name, ErrBadValue, b) // %q neutralises control characters
}

func (p *parser) store(wi int, b []byte, dst []Column) error {
	w := &p.plan.wanted[wi]
	c := &dst[w.dst]
	switch w.kind {
	case KindInt:
		v, ok := parseInt(b)
		if !ok {
			return valueErr(w, b)
		}
		c.Ints = append(c.Ints, v)
	case KindFloat:
		v, ok := parseFloat(b)
		if !ok {
			return valueErr(w, b)
		}
		c.Floats = append(c.Floats, v)
	default:
		c.Strs = append(c.Strs, p.intern(wi, b))
	}
	return nil
}

// fastRecord handles a record without any quote character.
func (p *parser) fastRecord(rec []byte, dst []Column) error {
	if got := bytes.Count(rec, commaByte) + 1; got != p.plan.ncols {
		return fmt.Errorf("%w: got %d, want %d", ErrFieldCount, got, p.plan.ncols)
	}
	start, col := 0, 0
	for wi := range p.plan.wanted {
		w := &p.plan.wanted[wi]
		for ; col < w.fileIdx; col++ { // hop over unwanted fields
			start += bytes.IndexByte(rec[start:], ',') + 1
		}
		end := len(rec)
		if i := bytes.IndexByte(rec[start:], ','); i >= 0 {
			end = start + i
		}
		if err := p.store(wi, rec[start:end], dst); err != nil {
			return err
		}
		start, col = end+1, col+1 // start may exceed len(rec) only after the last field
	}
	return nil
}

// quotedRecord handles a record that contains quotes.
func (p *parser) quotedRecord(rec []byte, dst []Column) error {
	fields, err := splitQuoted(rec, p.fields[:0])
	p.fields = fields[:0]
	if err != nil {
		return err
	}
	if len(fields) != p.plan.ncols {
		return fmt.Errorf("%w: got %d, want %d", ErrFieldCount, len(fields), p.plan.ncols)
	}
	for wi := range p.plan.wanted {
		if err := p.store(wi, fields[p.plan.wanted[wi].fileIdx], dst); err != nil {
			return err
		}
	}
	return nil
}

// parseChunk appends every record of data (complete records only) to dst.
func (p *parser) parseChunk(data []byte, base int64, dst []Column) error {
	upper := bytes.Count(data, []byte{'\n'}) + 1 // upper bound on the row count
	for _, w := range p.plan.wanted {
		c := &dst[w.dst]
		switch w.kind {
		case KindInt:
			c.Ints = slices.Grow(c.Ints, upper)
		case KindFloat:
			c.Floats = slices.Grow(c.Floats, upper)
		default:
			c.Strs = slices.Grow(c.Strs, upper)
		}
	}
	for pos := 0; pos < len(data); {
		start := pos
		rec, next, quoted, err := nextRecord(data, pos)
		pos = next
		if err == nil && len(rec) > 0 { // blank lines are skipped
			if quoted {
				err = p.quotedRecord(rec, dst)
			} else {
				err = p.fastRecord(rec, dst)
			}
		}
		if err != nil {
			return fmt.Errorf("byte offset %d: %w", base+int64(start), err)
		}
	}
	return nil
}

// --- public entry point -----------------------------------------------------

// concat joins the per-chunk parts (in chunk order) into exactly-sized columns.
// One allocation per column: no append regrowth garbage, no over-allocation.
func concat(parts [][]Column, dst []Column) {
	for i := range dst {
		var ni, nf, ns int
		for _, part := range parts {
			ni += len(part[i].Ints)
			nf += len(part[i].Floats)
			ns += len(part[i].Strs)
		}
		if ni > 0 {
			dst[i].Ints = make([]int64, 0, ni)
		}
		if nf > 0 {
			dst[i].Floats = make([]float64, 0, nf)
		}
		if ns > 0 {
			dst[i].Strs = make([]string, 0, ns)
		}
		for _, part := range parts {
			dst[i].Ints = append(dst[i].Ints, part[i].Ints...)
			dst[i].Floats = append(dst[i].Floats, part[i].Floats...)
			dst[i].Strs = append(dst[i].Strs, part[i].Strs...)
		}
	}
}

// ReadOptimal loads the columns named in schema from the CSV file at path.
//
// It honours ctx cancellation, never panics on malformed input, and returns
// the first error in file order. Time O(n); space O(result) + O(Workers*ChunkBytes).
func ReadOptimal(ctx context.Context, path string, schema Schema, opts Options) (Table, error) {
	opts, err := opts.normalize()
	if err != nil {
		return Table{}, err
	}
	if err := ctx.Err(); err != nil {
		return Table{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer f.Close()

	ck := newChunker(f, opts)
	first, err := ck.next()
	if errors.Is(err, io.EOF) {
		return Table{}, ErrEmptyInput
	}
	if err != nil {
		return Table{}, err
	}
	hdrRec, hdrEnd, _, err := nextRecord(first.data, 0)
	if err != nil {
		return Table{}, fmt.Errorf("header: %w", err)
	}
	hdrFields, err := splitQuoted(hdrRec, nil)
	if err != nil || len(hdrRec) == 0 {
		if err == nil {
			err = ErrEmptyInput
		}
		return Table{}, fmt.Errorf("header: %w", err)
	}
	header := make([]string, len(hdrFields))
	for i, h := range hdrFields {
		header[i] = string(h)
	}
	pl, err := buildPlan(header, schema)
	if err != nil {
		return Table{}, err
	}
	first.data, first.offset = first.data[hdrEnd:], first.offset+int64(hdrEnd)

	var parts [][]Column
	if opts.Workers == 1 {
		parts, err = readSequential(ctx, ck, first, pl, opts)
	} else {
		parts, err = readParallel(ctx, ck, first, pl, opts)
	}
	if err != nil {
		return Table{}, err
	}
	out := Table{Schema: slices.Clone(schema), Cols: make([]Column, len(schema))}
	concat(parts, out.Cols)
	return out, nil
}

func readSequential(ctx context.Context, ck *chunker, first chunk, pl *plan, o Options) ([][]Column, error) {
	p := newParser(pl, o.InternLimit)
	var parts [][]Column
	for ch, err := first, error(nil); ; ch, err = ck.next() {
		if errors.Is(err, io.EOF) {
			return parts, nil
		}
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		part := make([]Column, pl.nout)
		perr := p.parseChunk(ch.data, ch.offset, part)
		ck.release(ch)
		if perr != nil {
			return nil, perr
		}
		parts = append(parts, part)
	}
}

type result struct {
	seq  int
	part []Column
	err  error
}

func readParallel(parent context.Context, ck *chunker, first chunk, pl *plan, o Options) ([][]Column, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	inflight := 2 * o.Workers
	tokens := make(chan struct{}, inflight) // bounds chunks alive at once
	jobs := make(chan chunk)
	results := make(chan result, inflight+1)
	var wg sync.WaitGroup

	// Producer: the only goroutine that touches the chunker and the file.
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(jobs)
		seq := 0
		send := func(ch chunk) bool {
			select {
			case tokens <- struct{}{}:
			case <-ctx.Done():
				return false
			}
			ch.seq = seq
			seq++
			select {
			case jobs <- ch:
				return true
			case <-ctx.Done():
				return false
			}
		}
		if !send(first) {
			return
		}
		for {
			ch, err := ck.next()
			if errors.Is(err, io.EOF) {
				return
			}
			if err != nil { // reported in file order, after all earlier chunks
				select {
				case results <- result{seq: seq, err: err}:
				case <-ctx.Done():
				}
				return
			}
			if !send(ch) {
				return
			}
		}
	}()

	// Workers: each owns its parser (no shared mutable state, no locks).
	for w := 0; w < o.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p := newParser(pl, o.InternLimit)
			for ch := range jobs {
				part := make([]Column, pl.nout)
				err := p.parseChunk(ch.data, ch.offset, part)
				ck.release(ch)
				select {
				case results <- result{seq: ch.seq, part: part, err: err}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	// Consumer: merge strictly in chunk order => deterministic output and errors.
	var firstErr error
	var parts [][]Column
	pending := make(map[int]result)
	next := 0
loop:
	for res := range results {
		pending[res.seq] = res
		for {
			r, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			next++
			if r.err != nil {
				firstErr = r.err
				break loop
			}
			parts = append(parts, r.part)
			<-tokens
		}
	}
	cancel()
	wg.Wait() // every goroutine has stopped; the caller may now close the file
	if firstErr == nil {
		firstErr = parent.Err()
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return parts, nil
}

// ============================================================================
// SECTION 5: COMPARISON HARNESS
// ============================================================================
// Plain time.Now() harness (a testing.B benchmark would need a _test.go file).
// Correctness first against a trusted encoding/csv row-by-row reference, then
// timings (best of N), allocation volume (TotalAlloc delta) and the live heap
// held by the result after a GC. Numbers are only ever printed when measured.
// ============================================================================

type reader func(path string, schema Schema) (Table, error)

type approach struct {
	name string
	fn   reader
}

const (
	timeBudget    = 20 * time.Second // skip an approach at larger sizes once it exceeds this
	mib           = 1 << 20
	defaultMaxRow = 500_000
)

var sink Table // defeats dead-code elimination

// safeCall converts a panic into an error so the harness can report it.
func safeCall(fn reader, path string, schema Schema) (t Table, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	return fn(path, schema)
}

func optimalReader(o Options) reader {
	return func(path string, schema Schema) (Table, error) {
		return ReadOptimal(context.Background(), path, schema, o)
	}
}

// ReadReference is the trusted, slow, obviously-correct implementation.
func ReadReference(path string, schema Schema) (Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return Table{}, err
	}
	idx, err := columnIndexes(header, schema)
	if err != nil {
		return Table{}, err
	}
	out := Table{Schema: schema, Cols: make([]Column, len(schema))}
	for {
		rec, err := r.Read()
		if errors.Is(err, io.EOF) {
			return out, nil
		}
		if err != nil {
			return Table{}, err
		}
		for i, s := range schema {
			if err := appendCell(&out.Cols[i], s, rec[idx[i]]); err != nil {
				return Table{}, err
			}
		}
	}
}

func floatsEqual(a, b []float64) bool {
	return slices.EqualFunc(a, b, func(x, y float64) bool {
		return x == y || (math.IsNaN(x) && math.IsNaN(y))
	})
}

func sameTable(a, b Table) bool {
	if len(a.Cols) != len(b.Cols) {
		return false
	}
	for i := range a.Cols {
		if !slices.Equal(a.Cols[i].Ints, b.Cols[i].Ints) ||
			!floatsEqual(a.Cols[i].Floats, b.Cols[i].Floats) ||
			!slices.Equal(a.Cols[i].Strs, b.Cols[i].Strs) {
			return false
		}
	}
	return true
}

// writeCSV builds a deterministic synthetic file. messy adds tricky quoted notes.
func writeCSV(path string, n int, messy bool) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	w := csv.NewWriter(bw)
	rng := rand.New(rand.NewSource(7))
	cats := []string{"books", "games", "music", "tools", "food", "travel", "health", "garden"}
	tricky := []string{"hello, world", `say "hi"`, "line1\nline2", "a,b,\"c\"\nd"}
	_ = w.Write(csvHeader)
	rec := make([]string, 6)
	for i := 0; i < n; i++ {
		note := "ok"
		if messy && rng.Float64() < 0.02 {
			note = tricky[rng.Intn(len(tricky))]
		}
		rec[0] = strconv.Itoa(i)
		rec[1] = strconv.Itoa(rng.Intn(1_000_000))
		rec[2] = strconv.FormatFloat(rng.Float64()*1000, 'f', 2, 64)
		rec[3] = cats[rng.Intn(len(cats))]
		rec[4] = fmt.Sprintf("2025-01-%02dT%02d:00:00", 1+rng.Intn(28), rng.Intn(24))
		rec[5] = note
		if err := w.Write(rec); err != nil {
			f.Close()
			return err
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func verdict(err error, ok bool) string {
	switch {
	case err != nil:
		msg := err.Error()
		if len(msg) > 60 {
			msg = msg[:60] + "..."
		}
		return "FAIL (" + msg + ")"
	case !ok:
		return "FAIL (wrong values)"
	}
	return "PASS"
}

func checkCorrectness(tmp string, approaches []approach) bool {
	allOK := true
	note := func(ok bool) {
		if !ok {
			allOK = false
		}
	}
	fmt.Println("Correctness check (5,000 rows, compared with an encoding/csv reference):")
	for _, profile := range []string{"clean", "messy"} {
		path := filepath.Join(tmp, "check_"+profile+".csv")
		if err := writeCSV(path, 5000, profile == "messy"); err != nil {
			fmt.Println("  cannot create test file:", err)
			return false
		}
		ref, err := ReadReference(path, benchSchema)
		if err != nil {
			fmt.Println("  reference failed:", err)
			return false
		}
		for _, a := range approaches {
			got, err := safeCall(a.fn, path, benchSchema)
			v := verdict(err, err == nil && sameTable(got, ref))
			// Approach 1 is EXPECTED to fail on messy data: that is the lesson.
			note(v == "PASS" || a.name == "1. wrong")
			fmt.Printf("  %-6s %-16s %s\n", profile, a.name, v)
		}
	}

	// Chunk-boundary and concurrency stress on the messy file.
	msPath := filepath.Join(tmp, "check_messy.csv")
	ref, _ := ReadReference(msPath, benchSchema)
	stressOK := true
	for _, chunkSize := range []int{4096, 5000, 8191, 1 << 16} {
		for _, workers := range []int{1, 4} {
			got, err := ReadOptimal(context.Background(), msPath, benchSchema,
				Options{ChunkBytes: chunkSize, Workers: workers})
			if err != nil || !sameTable(got, ref) {
				stressOK = false
			}
		}
	}
	note(stressOK)
	fmt.Printf("  stress 3. optimal   chunk sizes x {1,4} workers on messy file: %s\n", verdict(nil, stressOK))

	// Edge cases: BOM, CRLF, quoted CRLF, empty float, no final newline.
	edge := filepath.Join(tmp, "edge.csv")
	body := "\xef\xbb\xbfid,amount,category\r\n1,2.5,\"a,b\"\r\n2,,\"x\r\ny\"\r\n3,4.0,z"
	_ = os.WriteFile(edge, []byte(body), 0o600)
	sch := Schema{{"id", KindInt}, {"amount", KindFloat}, {"category", KindString}}
	got, err := ReadOptimal(context.Background(), edge, sch, Options{ChunkBytes: 4096})
	edgeOK := err == nil && slices.Equal(got.Cols[0].Ints, []int64{1, 2, 3}) &&
		slices.Equal(got.Cols[2].Strs, []string{"a,b", "x\r\ny", "z"}) &&
		got.Cols[1].Floats[0] == 2.5 && math.IsNaN(got.Cols[1].Floats[1]) && got.Cols[1].Floats[2] == 4.0
	note(edgeOK)
	fmt.Printf("  edge   3. optimal   BOM, CRLF, quoted CRLF, empty float, no final newline: %s\n", verdict(err, edgeOK))

	// Error handling: every failure is a typed error, never a panic.
	type errCase struct {
		label string
		want  error
		run   func() error
	}
	writeTmp := func(name, content string) string {
		p := filepath.Join(tmp, name)
		_ = os.WriteFile(p, []byte(content), 0o600)
		return p
	}
	big := "id\n1,\"" + strings.Repeat("x", 60_000) // never-closed quote
	cases := []errCase{
		{"unknown column", ErrUnknownColumn, func() error {
			_, e := ReadOptimal(context.Background(), edge, Schema{{"nope", KindInt}}, Options{})
			return e
		}},
		{"bad int value", ErrBadValue, func() error {
			_, e := ReadOptimal(context.Background(), edge, Schema{{"category", KindInt}}, Options{})
			return e
		}},
		{"missing file", os.ErrNotExist, func() error {
			_, e := ReadOptimal(context.Background(), filepath.Join(tmp, "absent.csv"), sch, Options{})
			return e
		}},
		{"ragged row", ErrFieldCount, func() error {
			_, e := ReadOptimal(context.Background(), writeTmp("ragged.csv", "a,b\n1,2\n3\n"), Schema{{"a", KindInt}}, Options{})
			return e
		}},
		{"unterminated quote", ErrUnterminatedQuote, func() error {
			_, e := ReadOptimal(context.Background(), writeTmp("unterm.csv", "a,b\n1,\"oops\n"), Schema{{"a", KindInt}}, Options{})
			return e
		}},
		{"bare quote", ErrBareQuote, func() error {
			_, e := ReadOptimal(context.Background(), writeTmp("bare.csv", "a,b\n1,x\"y\"\n"), Schema{{"a", KindInt}}, Options{})
			return e
		}},
		{"oversized record", ErrRecordTooLarge, func() error {
			_, e := ReadOptimal(context.Background(), writeTmp("big.csv", big),
				Schema{{"id", KindInt}}, Options{ChunkBytes: 4096, MaxRecordBytes: 10_000})
			return e
		}},
		{"cancelled context", context.Canceled, func() error {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, e := ReadOptimal(ctx, edge, sch, Options{})
			return e
		}},
		{"first error wins (parallel)", ErrBadValue, func() error {
			// Two bad cells far apart: the EARLIER one must be reported, every time.
			var sb strings.Builder
			sb.WriteString("a,b\n")
			for i := 0; i < 9000; i++ {
				switch i {
				case 3000:
					sb.WriteString("x,2\n")
				case 6000:
					sb.WriteString("y,2\n")
				default:
					sb.WriteString("1,2\n")
				}
			}
			p := writeTmp("twobad.csv", sb.String())
			for i := 0; i < 20; i++ {
				_, e := ReadOptimal(context.Background(), p, Schema{{"a", KindInt}}, Options{ChunkBytes: 4096, Workers: 4})
				if e == nil || !strings.Contains(e.Error(), `"x"`) {
					return fmt.Errorf("run %d reported %v", i, e)
				}
			}
			return ErrBadValue
		}},
		{"invalid options", ErrInvalidOptions, func() error {
			_, e := ReadOptimal(context.Background(), edge, sch, Options{ChunkBytes: 100})
			return e
		}},
	}
	for _, c := range cases {
		e := c.run()
		ok := errors.Is(e, c.want)
		note(ok)
		fmt.Printf("  error  3. optimal   %-20s %s\n", c.label, map[bool]string{true: "PASS (typed error)", false: fmt.Sprintf("FAIL (%v)", e)}[ok])
	}
	fmt.Println()
	return allOK
}

type measurement struct {
	best      time.Duration
	allocMiB  float64
	liveMiB   float64
	err       error
	completed bool
}

func measure(fn reader, path string, repeats int) measurement {
	m := measurement{best: time.Duration(math.MaxInt64)}
	for i := 0; i < repeats; i++ {
		start := time.Now()
		t, err := safeCall(fn, path, benchSchema)
		el := time.Since(start)
		if err != nil {
			m.err = err
			return m
		}
		sink = t
		m.best = min(m.best, el)
	}
	// Allocation volume and retained heap, measured separately from timing.
	sink = Table{} // drop the result kept from the timing runs
	runtime.GC()
	var before, after, withRes, withoutRes runtime.MemStats
	runtime.ReadMemStats(&before)
	t, err := safeCall(fn, path, benchSchema)
	runtime.ReadMemStats(&after)
	if err != nil {
		m.err = err
		return m
	}
	sink = t
	runtime.GC()
	runtime.ReadMemStats(&withRes)
	sink = Table{}
	runtime.GC()
	runtime.ReadMemStats(&withoutRes)
	m.allocMiB = float64(after.TotalAlloc-before.TotalAlloc) / mib
	m.liveMiB = float64(int64(withRes.HeapAlloc)-int64(withoutRes.HeapAlloc)) / mib // heap held by the result alone
	m.completed = true
	return m
}

func runBenchmark(tmp string, maxRows int, approaches []approach) {
	var sizes []int
	for _, s := range []int{10_000, 100_000, 500_000, 1_000_000} {
		if s <= maxRows {
			sizes = append(sizes, s)
		}
	}
	if len(sizes) == 0 {
		sizes = []int{maxRows}
	}
	head := fmt.Sprintf("%-6s | %9s | %5s | %-16s | %9s | %9s | %8s | %9s",
		"profile", "rows", "MB", "approach", "time (s)", "alloc MiB", "live MiB", "vs common")
	fmt.Println(head)
	fmt.Println(strings.Repeat("-", len(head)))
	for _, profile := range []string{"clean", "messy"} {
		skipped := map[string]bool{}
		for _, n := range sizes {
			path := filepath.Join(tmp, fmt.Sprintf("bench_%s_%d.csv", profile, n))
			if err := writeCSV(path, n, profile == "messy"); err != nil {
				fmt.Println("cannot create benchmark file:", err)
				return
			}
			st, _ := os.Stat(path)
			mb := float64(st.Size()) / 1e6
			ms := map[string]measurement{}
			for _, a := range approaches {
				if skipped[a.name] {
					continue
				}
				repeats := 3
				if n > 100_000 {
					repeats = 1
				}
				m := measure(a.fn, path, repeats)
				ms[a.name] = m
				if m.err == nil && m.best > timeBudget {
					skipped[a.name] = true
				}
			}
			base, hasBase := ms["2. common"]
			for _, a := range approaches {
				m, ran := ms[a.name]
				switch {
				case !ran:
					fmt.Printf("%-6s | %9d | %5.0f | %-16s | skipped (too slow)\n", profile, n, mb, a.name)
				case m.err != nil:
					msg := m.err.Error()
					if len(msg) > 44 {
						msg = msg[:44] + "..."
					}
					fmt.Printf("%-6s | %9d | %5.0f | %-16s | invalid: %s\n", profile, n, mb, a.name, msg)
				default:
					ratio := "n/a"
					if hasBase && base.err == nil {
						ratio = fmt.Sprintf("%.2fx", base.best.Seconds()/m.best.Seconds())
					}
					fmt.Printf("%-6s | %9d | %5.0f | %-16s | %9.3f | %9.1f | %8.1f | %9s\n",
						profile, n, mb, a.name, m.best.Seconds(), m.allocMiB, m.liveMiB, ratio)
				}
			}
			fmt.Println(strings.Repeat("-", len(head)))
			os.Remove(path)
		}
	}
}

func printSummary(workers int) {
	fmt.Printf(`
Complexity summary
  wrong   : O(n), ~2x file + a []string per row, panics or silent zeros on quoted data
  common  : O(n*c), every column of every row becomes a string first, correct
  optimal : O(n), quote-parity chunks, allocation-free fast path, typed columns

Verdict
  The win comes from three places: (1) never materialise data you did not ask
  for, (2) decide record boundaries by quote parity so lines without quotes
  take an allocation-free path, (3) chunks are independent, so they can be
  parsed by several goroutines and merged in order with identical output.
  "3. optimal" above is the single-goroutine path; "3b" (shown only when more
  than one CPU is available) uses the parallel pipeline. This run had
  GOMAXPROCS=%d. Timings come from this machine; ratios and scaling matter.
`, workers)
}

// ============================================================================
// SECTION 6: ENTRY POINT
// ============================================================================

func main() {
	maxRows := defaultMaxRow
	if len(os.Args) > 1 {
		n, err := strconv.Atoi(os.Args[1])
		if err != nil || n < 1 {
			fmt.Fprintln(os.Stderr, "usage: fast_csv_reader [max_rows]")
			os.Exit(2)
		}
		maxRows = n
	}
	tmp, err := os.MkdirTemp("", "pitfalllab-csv-")
	if err != nil {
		fmt.Fprintln(os.Stderr, "cannot create temp dir:", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmp)

	approaches := []approach{
		{"1. wrong", ReadWrong},
		{"2. common", ReadCommon},
		{"3. optimal", optimalReader(Options{Workers: 1})},
	}
	if procs := runtime.GOMAXPROCS(0); procs > 1 {
		approaches = append(approaches, approach{
			fmt.Sprintf("3b. optimal x%d", min(procs, 8)),
			optimalReader(Options{Workers: min(procs, 8)}),
		})
	}
	ok := checkCorrectness(tmp, approaches)
	runBenchmark(tmp, maxRows, approaches)
	printSummary(runtime.GOMAXPROCS(0))
	if !ok {
		fmt.Fprintln(os.Stderr, "\nUNEXPECTED correctness failure in a non-wrong approach")
		os.Exit(1)
	}
}
