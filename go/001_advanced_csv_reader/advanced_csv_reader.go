// ============================================================================
// SECTION 1: SCENARIO
// ============================================================================
// Advanced CSV Reader (standard library only).
//
// Read a very large CSV file into memory as fast as possible, without any
// third-party module and without converting the file to another format.
//
// Architecture (System Design):
//   - Chunking: Splits file based on RFC 4180 quote parity (SIMD-friendly safeCut).
//   - Scatter-Gather: Distributes chunks to N Lock-free worker goroutines.
//   - Zero-Allocation Hot Path: Uses recycled buffers (unescBuf) for both
//     clean and quoted fields to completely eliminate heap allocations per row.
//   - Deterministic Merge: Collects results strictly in chunk order.
//
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

type Kind int

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

type ColumnSpec struct {
	Name string
	Kind Kind
}

type Schema []ColumnSpec

type Column struct {
	Ints   []int64
	Floats []float64
	Strs   []string
}

type Table struct {
	Schema Schema
	Cols   []Column
}

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

var benchSchema = Schema{
	{Name: "id", Kind: KindInt},
	{Name: "amount", Kind: KindFloat},
	{Name: "category", Kind: KindString},
}

var csvHeader = []string{"id", "user_id", "amount", "category", "ts", "note"}

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

func ReadWrong(path string, schema Schema) (Table, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Table{}, err
	}
	lines := strings.Split(string(data), "\n")
	header := strings.Split(lines[0], ",")
	idx := make([]int, len(schema))
	for i, s := range schema {
		idx[i] = slices.Index(header, s.Name)
	}
	cols := make([]Column, len(schema))
	for _, line := range lines[1:] {
		if line == "" {
			continue
		}
		row := strings.Split(line, ",")
		for i, s := range schema {
			cell := row[idx[i]]
			switch s.Kind {
			case KindInt:
				v, _ := strconv.ParseInt(cell, 10, 64)
				cols[i].Ints = append(cols[i].Ints, v)
			case KindFloat:
				v, _ := strconv.ParseFloat(cell, 64)
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

func ReadCommon(path string, schema Schema) (Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll()
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

type Options struct {
	ChunkBytes     int
	Workers        int
	MaxRecordBytes int
	InternLimit    int
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

type chunk struct {
	data   []byte
	buf    *[]byte
	offset int64
	seq    int
}

type chunker struct {
	r      io.Reader
	size   int
	maxRec int
	tail   []byte
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

func safeCut(data []byte) int {
	cut := bytes.LastIndexByte(data, '\n')
	if cut < 0 {
		return 0
	}
	quotes := bytes.Count(data[:cut], quoteByte)
	for cut >= 0 && quotes%2 == 1 {
		prev := bytes.LastIndexByte(data[:cut], '\n')
		quotes -= bytes.Count(data[prev+1:cut], quoteByte)
		cut = prev
	}
	if cut < 0 {
		return 0
	}
	return cut + 1
}

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
			c.eof = err != nil
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
		if c.eof {
			if len(data) == 0 {
				c.release(chunk{buf: bp})
				return chunk{}, io.EOF
			}
			ch := chunk{data: data, buf: bp, offset: c.offset}
			c.offset += int64(len(data))
			return ch, nil
		}
		cut := safeCut(data)
		if cut == 0 {
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

// splitQuoted is retained solely for safely extracting the initial header row
func splitQuoted(rec []byte, out [][]byte) ([][]byte, error) {
	n := len(rec)
	i := 0
	for {
		if i < n && rec[i] == '"' {
			i++
			start := i
			var unesc []byte
			for {
				j := bytes.IndexByte(rec[i:], '"')
				if j < 0 {
					return out, ErrUnterminatedQuote
				}
				i += j
				if i+1 < n && rec[i+1] == '"' {
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
				i++
				break
			}
			if i == n {
				return out, nil
			}
			if rec[i] != ',' {
				return out, ErrBareQuote
			}
			i++
			if i == n {
				return append(out, nil), nil
			}
			continue
		}
		j := bytes.IndexByte(rec[i:], ',')
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
	if len(b) <= 18 {
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

		_ = b[len(b)-1] // BCE (Bounds Check Elimination) optimization

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
	v, err := strconv.ParseInt(string(b), 10, 64)
	return v, err == nil
}

func parseFloat(b []byte) (float64, bool) {
	if len(b) == 0 {
		return math.NaN(), true
	}
	v, err := strconv.ParseFloat(string(b), 64)
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
	fileIdx int
	dst     int
	kind    Kind
	name    string
}

type plan struct {
	ncols  int
	nout   int
	wanted []wantedCol
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

type parser struct {
	plan     *plan
	interned []map[string]string
	limit    int
	fields   [][]byte
	unescBuf []byte // zero-allocation buffer reused for resolving escaped quotes
}

func newParser(pl *plan, internLimit int) *parser {
	p := &parser{
		plan:     pl,
		limit:    internLimit,
		interned: make([]map[string]string, len(pl.wanted)),
		unescBuf: make([]byte, 0, 1024),
	}
	for i, w := range pl.wanted {
		if w.kind == KindString {
			p.interned[i] = make(map[string]string)
		}
	}
	return p
}

func (p *parser) intern(wi int, b []byte) string {
	m := p.interned[wi]
	if s, ok := m[string(b)]; ok {
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
	return fmt.Errorf("column %q: %w: %q", w.name, ErrBadValue, b)
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

func (p *parser) fastRecord(rec []byte, dst []Column) error {
	if got := bytes.Count(rec, commaByte) + 1; got != p.plan.ncols {
		return fmt.Errorf("%w: got %d, want %d", ErrFieldCount, got, p.plan.ncols)
	}
	start, col := 0, 0
	for wi := range p.plan.wanted {
		w := &p.plan.wanted[wi]
		for ; col < w.fileIdx; col++ {
			start += bytes.IndexByte(rec[start:], ',') + 1
		}
		end := len(rec)
		if i := bytes.IndexByte(rec[start:], ','); i >= 0 {
			end = start + i
		}
		if err := p.store(wi, rec[start:end], dst); err != nil {
			return err
		}
		start, col = end+1, col+1
	}
	return nil
}

// quotedRecord seamlessly handles records containing quotes strictly avoiding all allocations
// by effectively multiplexing p.unescBuf and slicing directly on internal state.
func (p *parser) quotedRecord(rec []byte, dst []Column) error {
	p.fields = p.fields[:0]
	p.unescBuf = p.unescBuf[:0]

	n := len(rec)
	i := 0
	for {
		if i < n && rec[i] == '"' {
			i++
			start := i
			unescStart := len(p.unescBuf)
			for {
				j := bytes.IndexByte(rec[i:], '"')
				if j < 0 {
					return ErrUnterminatedQuote
				}
				i += j
				if i+1 < n && rec[i+1] == '"' {
					p.unescBuf = append(p.unescBuf, rec[start:i+1]...)
					i += 2
					start = i
					continue
				}

				if unescStart == len(p.unescBuf) {
					p.fields = append(p.fields, rec[start:i])
				} else {
					p.unescBuf = append(p.unescBuf, rec[start:i]...)
					p.fields = append(p.fields, p.unescBuf[unescStart:])
				}
				i++
				break
			}
			if i == n {
				break
			}
			if rec[i] != ',' {
				return ErrBareQuote
			}
			i++
			if i == n {
				p.fields = append(p.fields, nil)
				break
			}
			continue
		}

		j := bytes.IndexByte(rec[i:], ',')
		end := n
		if j >= 0 {
			end = i + j
		}
		if bytes.IndexByte(rec[i:end], '"') >= 0 {
			return ErrBareQuote
		}
		p.fields = append(p.fields, rec[i:end])
		if j < 0 {
			break
		}
		i = end + 1
		if i == n {
			p.fields = append(p.fields, nil)
			break
		}
	}

	if len(p.fields) != p.plan.ncols {
		return fmt.Errorf("%w: got %d, want %d", ErrFieldCount, len(p.fields), p.plan.ncols)
	}
	for wi := range p.plan.wanted {
		if err := p.store(wi, p.fields[p.plan.wanted[wi].fileIdx], dst); err != nil {
			return err
		}
	}
	return nil
}

func (p *parser) parseChunk(data []byte, base int64, dst []Column) error {
	upper := bytes.Count(data, []byte{'\n'}) + 1
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
		if err == nil && len(rec) > 0 {
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
	tokens := make(chan struct{}, inflight)
	jobs := make(chan chunk)
	results := make(chan result, inflight+1)
	var wg sync.WaitGroup

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
			if err != nil {
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
	wg.Wait()
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

type reader func(path string, schema Schema) (Table, error)

type approach struct {
	name string
	fn   reader
}

const (
	timeBudget    = 20 * time.Second
	mib           = 1 << 20
	defaultMaxRow = 500_000
)

var sink Table

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
			note(v == "PASS" || a.name == "1. wrong")
			fmt.Printf("  %-6s %-16s %s\n", profile, a.name, v)
		}
	}

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
	big := "id\n1,\"" + strings.Repeat("x", 60_000)
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
	sink = Table{}
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
	m.liveMiB = float64(int64(withRes.HeapAlloc)-int64(withoutRes.HeapAlloc)) / mib
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
