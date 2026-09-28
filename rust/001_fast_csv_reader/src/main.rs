// =============================================================================
// triad: reading a huge CSV file as fast as possible (Rust, std only)
//
// Build (optimizations are mandatory, debug timings are meaningless):
//     rustc --edition 2021 -C opt-level=3 csv_fast_read.rs -o csv_fast_read
//     ./csv_fast_read            # default: up to 2,000,000 rows
//     ./csv_fast_read 500000     # custom maximum row count
// or, inside a cargo project:  cargo run --release -- 500000
// =============================================================================

// =============================================================================
// SECTION 1: SCENARIO
// =============================================================================
//
// CONTEXT
//   A service must load a large CSV file (10^6 .. 10^8 rows, hundreds of MiB to
//   several GiB) into memory so that later code can run aggregations on it.
//   No DataFrame library (no Pandas/Polars equivalents, no `csv` crate), and
//   the file must NOT be converted to Parquet or any other format first. The
//   only goal of this module is: file on disk -> typed in-memory table, in the
//   least possible wall-clock time, with correct handling of real-world CSV.
//
// INPUT FORMAT (RFC 4180 subset, what real exports look like)
//   header:   id,user_id,note,amount,category      (validated, exact match)
//   id        unsigned decimal, u64
//   user_id   unsigned decimal, u32
//   note      free text; may be quoted and then may contain commas, escaped
//             quotes ("") and even NEWLINES. Not needed by the caller.
//   amount    decimal with exactly two fraction digits ("19.99") or none ("20")
//   category  short text with few distinct values (dictionary-encoded)
//   Line ends: LF or CRLF. Optional UTF-8 BOM. Last line may lack a newline.
//
// OUTPUT
//   `Table`: a COLUMNAR struct (one Vec per column). Amounts are exact integer
//   cents (i64), categories are u16 codes into a dictionary. The `note` column
//   is validated structurally but not materialized (projection: you only pay
//   for the columns you use).
//
// CONSTRAINTS AND ASSUMPTIONS
//   * std only, safe Rust (the only `unsafe` is the tracking allocator used by
//     the benchmark to measure peak memory).
//   * The file fits in RAM (see the trade-off notes of approach 3 for the
//     streaming variant when it does not).
//   * Malformed input must produce an error with a byte offset, never silently
//     wrong data. Results must be deterministic and independent of thread count.
//   * Benchmarks read from the OS page cache (warm file). On a cold cache the
//     disk is the bottleneck and every approach converges to disk speed; the
//     memory and CPU savings of approach 3 still apply.
//
// WHAT "GOOD" MEANS
//   Time O(n) with a tiny constant (several times faster than a line-by-line
//   reader on the same core, and scaling with the number of cores), ZERO heap
//   allocations per row, memory close to file size + 22 bytes per row, and
//   identical output to a trusted reference.
//
// ---- shared data model used by all three approaches -------------------------

use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

/// The exact header every input file must start with.
pub const HEADER: [&str; 5] = ["id", "user_id", "note", "amount", "category"];
const NFIELDS: usize = HEADER.len();

#[derive(Debug)]
pub enum CsvError {
    Io(io::Error),
    BadHeader,
    /// `byte_offset` is the start of the offending record in the file.
    Malformed {
        byte_offset: usize,
        reason: &'static str,
    },
    InvalidUtf8,
    TooManyCategories,
    OutOfMemory,
    Internal(&'static str),
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsvError::Io(e) => write!(f, "I/O error: {e}"),
            CsvError::BadHeader => write!(f, "missing or unexpected header"),
            CsvError::Malformed {
                byte_offset,
                reason,
            } => {
                write!(f, "malformed record at byte {byte_offset}: {reason}")
            }
            CsvError::InvalidUtf8 => write!(f, "category is not valid UTF-8"),
            CsvError::TooManyCategories => write!(f, "more than 65536 distinct categories"),
            CsvError::OutOfMemory => write!(f, "cannot allocate memory for the file"),
            CsvError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl Error for CsvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CsvError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CsvError {
    fn from(e: io::Error) -> Self {
        CsvError::Io(e)
    }
}

/// Columnar result. Dictionary order is "first appearance in the file", which
/// makes the whole struct comparable with `==` across approaches.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub id: Vec<u64>,
    pub user_id: Vec<u32>,
    pub amount_cents: Vec<i64>,
    pub category: Vec<u16>,
    pub category_dict: Vec<String>,
}

impl Table {
    pub fn len(&self) -> usize {
        self.id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }
    pub fn category_name(&self, row: usize) -> &str {
        &self.category_dict[self.category[row] as usize]
    }
    pub fn total_cents(&self) -> i64 {
        self.amount_cents.iter().sum()
    }
    /// Example consumer: revenue per category, largest first (ties by name).
    pub fn total_by_category(&self) -> Vec<(&str, i64)> {
        let mut sums = vec![0i64; self.category_dict.len()];
        for (&c, &a) in self.category.iter().zip(&self.amount_cents) {
            sums[c as usize] += a;
        }
        let mut v: Vec<(&str, i64)> = self
            .category_dict
            .iter()
            .map(String::as_str)
            .zip(sums)
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

/// Strict decimal parser: `D+` or `D+.DD` -> cents. Exact integer arithmetic,
/// overflow-checked, no floats (so no 19.99 * 100 = 1998.9999999999998 bugs).
fn parse_cents(s: &[u8]) -> Option<i64> {
    let (int, frac) = match s.iter().position(|&b| b == b'.') {
        Some(p) => (&s[..p], Some(&s[p + 1..])),
        None => (s, None),
    };
    if int.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for &b in int {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as i64)?;
    }
    let cents = match frac {
        None => 0,
        Some(f) if f.len() == 2 && f.iter().all(u8::is_ascii_digit) => {
            ((f[0] - b'0') * 10 + (f[1] - b'0')) as i64
        }
        _ => return None,
    };
    v.checked_mul(100)?.checked_add(cents)
}

/// Builds a `Table` row by row (used by approaches 1, 2 and the reference).
#[derive(Default)]
struct TableBuilder {
    table: Table,
    index: HashMap<String, u16>,
}

impl TableBuilder {
    fn push(&mut self, id: u64, user_id: u32, cents: i64, category: &str) -> Result<(), CsvError> {
        let code = match self.index.get(category) {
            Some(&c) => c,
            None => {
                let c = u16::try_from(self.table.category_dict.len())
                    .map_err(|_| CsvError::TooManyCategories)?;
                self.index.insert(category.to_owned(), c);
                self.table.category_dict.push(category.to_owned());
                c
            }
        };
        self.table.id.push(id);
        self.table.user_id.push(user_id);
        self.table.amount_cents.push(cents);
        self.table.category.push(code);
        Ok(())
    }
    fn finish(self) -> Table {
        self.table
    }
}

// =============================================================================
// SECTION 2: APPROACH 1 - WRONG
// =============================================================================
//
// WHAT THE AUTHOR WAS THINKING
//   "A CSV is just lines of comma separated values. Read it all, split on
//   newlines, split on commas, parse what I need. Done in five lines."
//
// WHY IT IS WRONG (each one is a real, common production bug)
//   1. Naive `split(',')` and `lines()` are not CSV parsing. A quoted field
//      like "Smith, John" shifts every later column; a quoted field with a
//      newline is torn into two bogus rows. With `note` in the middle of the
//      schema, `amount` and `category` are read from the wrong columns.
//   2. `unwrap_or(0)` turns every parse failure into a plausible-looking zero.
//      The program never fails, it just produces wrong numbers: the worst kind
//      of bug because nobody notices until a report is wrong.
//   3. `f64 * 100.0` then `as i64` TRUNCATES: 19.99 * 100.0 = 1998.9999999999998
//      becomes 1998 cents. Money must be parsed as integers, or rounded.
//   4. Memory blow-up: the whole file as a `String`, PLUS a `Vec<Vec<String>>`
//      with ~6 heap allocations per row (each String header alone is 24 bytes,
//      plus allocator overhead), all alive at the same time. Typically 5-10x
//      the file size.
//   5. `read_to_string` rejects the entire file if a single byte is not UTF-8.
//
// FAILURE MODE:  silent data corruption + memory explosion.
// COMPLEXITY:    O(n) time, but ~6n allocations; O(6n * 40 B + file) memory.

pub fn read_csv_wrong(path: &Path) -> Result<Table, CsvError> {
    let content = fs::read_to_string(path)?; // whole file, plus a UTF-8 validation pass
    let rows: Vec<Vec<String>> = content
        .lines()
        .skip(1) // "skip the header" (never checked)
        .map(|line| line.split(',').map(|f| f.to_string()).collect())
        .collect();

    let mut b = TableBuilder::default();
    for r in &rows {
        let get = |i: usize| r.get(i).map(String::as_str).unwrap_or("");
        let id = get(0).parse::<u64>().unwrap_or(0);
        let user = get(1).parse::<u32>().unwrap_or(0);
        let cents = (get(3).parse::<f64>().unwrap_or(0.0) * 100.0) as i64; // truncates!
        b.push(id, user, cents, get(4))?;
    }
    Ok(b.finish())
}

// =============================================================================
// SECTION 3: APPROACH 2 - COMMON
// =============================================================================
//
// WHAT MOST DEVELOPERS WRITE
//   `BufReader` + `read_line`, a small quote-aware splitter, `str::parse` for
//   the numbers, and a `Result` for bad input. This is exactly what you would
//   write without a CSV crate, and it is CORRECT: quoted commas, escaped
//   quotes and embedded newlines work (a record is extended with more physical
//   lines while the number of quote characters is odd).
//
// WHY PEOPLE CHOOSE IT
//   Streaming (memory does not grow with the file), simple, readable, easy to
//   debug, and fast enough for files up to a few hundred MB.
//
// WHERE IT STARTS TO HURT
//   * ~6 heap allocations per row: one `Vec<String>` plus one `String` per
//     field, all thrown away immediately. Allocator traffic dominates the time.
//   * Every byte is decoded as UTF-8 `char`s, including the `note` column the
//     caller never uses.
//   * Numbers via `f64`: `"1e3"`, `"NaN"` or `"1.234"` are accepted as amounts
//     (looser than the contract), and money goes through floating point.
//   * Default `BufReader` buffer is only 8 KiB (many small `read` syscalls).
//   * One core only.
//
// WHAT IT LEAVES ON THE TABLE
//   Borrowing instead of allocating, byte parsing instead of `char` decoding,
//   column projection and parallelism: an order of magnitude in total.
//
// COMPLEXITY: O(n) time, ~6n allocations, O(row + output) memory.

fn split_fields(rec: &str) -> Result<Vec<String>, &'static str> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut chars = rec.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_q) {
            ('"', true) => {
                if chars.peek() == Some(&'"') {
                    cur.push('"'); // escaped quote
                    chars.next();
                } else {
                    in_q = false;
                }
            }
            ('"', false) => in_q = true,
            (',', false) => out.push(std::mem::take(&mut cur)),
            (ch, _) => cur.push(ch),
        }
    }
    if in_q {
        return Err("unterminated quoted field");
    }
    out.push(cur);
    Ok(out)
}

pub fn read_csv_common(path: &Path) -> Result<Table, CsvError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut offset = 0usize;

    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Err(CsvError::BadHeader);
    }
    offset += n;
    let header = split_fields(
        line.trim_start_matches('\u{feff}')
            .trim_end_matches(&['\r', '\n'][..]),
    )
    .map_err(|_| CsvError::BadHeader)?;
    if !header.iter().map(String::as_str).eq(HEADER) {
        return Err(CsvError::BadHeader);
    }

    let mut b = TableBuilder::default();
    loop {
        line.clear();
        let start = offset;
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        offset += n;
        let bad = |reason: &'static str| CsvError::Malformed {
            byte_offset: start,
            reason,
        };

        // A quoted field may span physical lines: keep reading while quotes are unbalanced.
        while line.bytes().filter(|&c| c == b'"').count() % 2 == 1 {
            let m = reader.read_line(&mut line)?;
            if m == 0 {
                return Err(bad("unterminated quoted field"));
            }
            offset += m;
        }

        let rec = line.trim_end_matches(&['\r', '\n'][..]);
        let f = split_fields(rec).map_err(bad)?;
        if f.len() != NFIELDS {
            return Err(bad("wrong number of fields"));
        }
        let id = f[0].parse::<u64>().map_err(|_| bad("invalid id"))?;
        let user = f[1].parse::<u32>().map_err(|_| bad("invalid user_id"))?;
        let amount = f[3].parse::<f64>().map_err(|_| bad("invalid amount"))?;
        if f[4].is_empty() {
            return Err(bad("empty category"));
        }
        b.push(id, user, (amount * 100.0).round() as i64, &f[4])?;
    }
    Ok(b.finish())
}

// =============================================================================
// SECTION 4: APPROACH 3 - OPTIMAL
// =============================================================================
//
// KEY INSIGHT
//   Fast CSV reading is three separate problems, and each has a standard answer:
//
//   (1) I/O: read the file with ONE `read` into ONE pre-sized `Vec<u8>`
//       (exact capacity from metadata, no regrowth, no per-line syscalls, no
//       UTF-8 validation of the whole file). `try_reserve_exact` turns an
//       impossible allocation into an error instead of an abort.
//
//   (2) Parsing: work on BYTES and BORROW. A record is scanned once with a
//       256-entry lookup table that classifies each byte (ordinary / quote /
//       comma / newline), so the hot loop is one load and one predictable
//       branch per byte. Fields are (start, end) ranges into the buffer: ZERO
//       heap allocations per row. Numbers are parsed straight from bytes with
//       checked integer arithmetic. The unused `note` column is never decoded.
//       Categories are dictionary-encoded to u16 (a row costs 22 bytes total).
//
//   (3) Parallelism WITHOUT breaking quoting. Splitting a CSV at arbitrary
//       '\n' bytes is wrong, because a newline may sit inside a quoted field.
//       The trick: a quote toggles "inside quotes" state, and an escaped quote
//       ("") toggles twice, so the state at any byte offset is simply the
//       PARITY of the number of '"' bytes before it. Therefore:
//         phase A (parallel):   count quotes/newlines in each raw chunk
//         phase B (sequential): prefix-XOR the parities -> the exact quote
//                               state at every chunk start; scan a few bytes
//                               forward to the next TRUE record boundary
//         phase C (parallel):   parse each record-aligned chunk independently
//         merge:                concatenate columns, remap local dictionaries
//                               in chunk order (= first-appearance order)
//       The result is bit-identical for any thread count, including 1.
//
// DESIGN CHOICES
//   * Columnar output with integer cents: deterministic parallel sums, cache
//     friendly scans later, no float rounding surprises.
//   * FNV-1a hasher for the tiny per-chunk category map (keys are trusted
//     local data, so HashDoS resistance of SipHash buys nothing here).
//   * The file buffer is dropped BEFORE the merge, so peak memory is
//     ~ file + columns, not file + columns + merged copy.
//   * `thread::scope` (no 'static bounds, no Arc), workers own their output,
//     so it is data-race free by construction. Spawn failures are errors.
//   * Errors carry a byte offset; the earliest error in the file wins, so the
//     error is deterministic too.
//
// COMPLEXITY
//   Time:  O(n) total work, ~O(n / cores) wall clock, plus O(rows) merge memcpy.
//   Space: O(file size) for the buffer + O(rows) for the columns.
//
// TRADE-OFFS AND WHEN NOT TO USE THIS
//   * The whole file is held in memory. If the file can exceed RAM, keep the
//     same `scan_record`/`parse_chunk` core but read in 8-64 MiB blocks and
//     carry the incomplete tail record to the next block (bounded memory,
//     single reader thread feeding a worker pool).
//   * Without `mmap` (needs libc/unsafe or a crate) the read is one memcpy from
//     the page cache; parallel `pread` per chunk is the next optimization.
//   * Strictness: a stray quote inside an UNquoted projected field is an error,
//     amounts must be `D+` or `D+.DD`, and blank lines are errors. If you must
//     accept sloppier files, relax `field()` and `parse_cents()` on purpose.
//   * For tiny files (< 1 MiB) threads only add overhead: `Options` keeps them
//     single-threaded automatically.

/// Tunables. `min_chunk_bytes` also makes the boundary logic testable on small inputs.
#[derive(Clone, Debug)]
pub struct Options {
    pub threads: usize,
    pub min_chunk_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            threads: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            min_chunk_bytes: 1 << 20,
        }
    }
}

// Byte classes for the scanner lookup table.
const Q: u8 = 1;
const COMMA: u8 = 2;
const NL: u8 = 3;

static CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b'"' as usize] = Q;
    t[b',' as usize] = COMMA;
    t[b'\n' as usize] = NL;
    t
};

type Fields = [(usize, usize); NFIELDS];

/// Scans ONE record starting at `start` (which must be a record boundary).
/// Fills `out` with the byte range of every raw field (quotes still included)
/// and returns the offset just past the record terminator.
/// Invariant: `in_q` toggles on every '"' byte, so `""` inside a quoted field
/// toggles twice and leaves the state unchanged, exactly what the parity trick needs.
#[inline]
fn scan_record(buf: &[u8], start: usize, out: &mut Fields) -> Result<usize, &'static str> {
    let mut nf = 0usize;
    let mut field_start = start;
    let mut in_q = false;
    for (k, &b) in buf[start..].iter().enumerate() {
        let class = CLASS[b as usize];
        if class == 0 {
            continue; // the overwhelmingly common case
        }
        let i = start + k;
        if class == Q {
            in_q = !in_q;
            continue;
        }
        if in_q {
            continue; // comma or newline inside quotes is data
        }
        let mut end = i;
        if class == NL && end > field_start && buf[end - 1] == b'\r' {
            end -= 1; // CRLF
        }
        if nf == NFIELDS {
            return Err("too many fields");
        }
        out[nf] = (field_start, end);
        nf += 1;
        field_start = i + 1;
        if class == NL {
            return if nf == NFIELDS {
                Ok(i + 1)
            } else {
                Err("too few fields")
            };
        }
    }
    // End of input without a terminating newline.
    if in_q {
        return Err("unterminated quoted field");
    }
    if nf == NFIELDS {
        return Err("too many fields");
    }
    out[nf] = (field_start, buf.len());
    nf += 1;
    if nf == NFIELDS {
        Ok(buf.len())
    } else {
        Err("too few fields")
    }
}

/// Returns the logical content of a raw field. Borrowed in the fast path;
/// allocates only for a quoted field that contains escaped quotes.
#[inline]
fn field(chunk: &[u8], (s, e): (usize, usize)) -> Result<Cow<'_, [u8]>, &'static str> {
    let raw = &chunk[s..e];
    if raw.first() != Some(&b'"') {
        return if raw.contains(&b'"') {
            Err("unexpected quote in unquoted field")
        } else {
            Ok(Cow::Borrowed(raw))
        };
    }
    if raw.len() < 2 || raw[raw.len() - 1] != b'"' {
        return Err("malformed quoted field");
    }
    let inner = &raw[1..raw.len() - 1];
    if !inner.contains(&b'"') {
        return Ok(Cow::Borrowed(inner));
    }
    let mut out = Vec::with_capacity(inner.len());
    let mut k = 0;
    while k < inner.len() {
        let b = inner[k];
        if b == b'"' {
            if inner.get(k + 1) == Some(&b'"') {
                out.push(b'"');
                k += 2;
                continue;
            }
            return Err("stray quote inside quoted field");
        }
        out.push(b);
        k += 1;
    }
    Ok(Cow::Owned(out))
}

fn parse_u64(s: &[u8]) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    let mut v = 0u64;
    for &b in s {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as u64)?;
    }
    Some(v)
}

/// FNV-1a: tiny, fast for short keys. Not DoS resistant (fine for trusted files).
struct Fnv(u64);
impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}
impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}
type FnvMap<K, V> = HashMap<K, V, BuildHasherDefault<Fnv>>;

/// Output of one worker: columns plus a chunk-local category dictionary.
#[derive(Default)]
struct ChunkOut {
    id: Vec<u64>,
    user_id: Vec<u32>,
    cents: Vec<i64>,
    code: Vec<u16>,
    dict: Vec<Vec<u8>>,
    index: FnvMap<Vec<u8>, u16>,
}

/// Parses a record-aligned chunk. `base` is the chunk's offset in the file
/// (for error positions), `row_hint` a capacity hint (newline count).
fn parse_chunk(chunk: &[u8], base: usize, row_hint: usize) -> Result<ChunkOut, CsvError> {
    // A record has at least 5 bytes (4 commas + newline), which bounds any hint.
    let cap = row_hint.min(chunk.len() / 5 + 1);
    let mut out = ChunkOut {
        id: Vec::with_capacity(cap),
        user_id: Vec::with_capacity(cap),
        cents: Vec::with_capacity(cap),
        code: Vec::with_capacity(cap),
        ..ChunkOut::default()
    };
    let mut fields: Fields = [(0, 0); NFIELDS];
    let mut pos = 0usize;
    while pos < chunk.len() {
        let at = base + pos;
        let bad = move |reason: &'static str| CsvError::Malformed {
            byte_offset: at,
            reason,
        };

        let end = scan_record(chunk, pos, &mut fields).map_err(bad)?;

        let id =
            parse_u64(&field(chunk, fields[0]).map_err(bad)?).ok_or_else(|| bad("invalid id"))?;
        let user = parse_u64(&field(chunk, fields[1]).map_err(bad)?)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| bad("invalid user_id"))?;
        // fields[2] (note) is intentionally never decoded.
        let cents = parse_cents(&field(chunk, fields[3]).map_err(bad)?)
            .ok_or_else(|| bad("invalid amount"))?;
        let cat = field(chunk, fields[4]).map_err(bad)?;
        if cat.is_empty() {
            return Err(bad("empty category"));
        }
        let code = if let Some(&c) = out.index.get(&*cat) {
            c
        } else {
            let c = u16::try_from(out.dict.len()).map_err(|_| CsvError::TooManyCategories)?;
            let owned = cat.into_owned(); // only on the first sighting of a category
            out.index.insert(owned.clone(), c);
            out.dict.push(owned);
            c
        };

        out.id.push(id);
        out.user_id.push(user);
        out.cents.push(cents);
        out.code.push(code);
        pos = end;
    }
    Ok(out)
}

/// Runs `f(0..n)` on `n` scoped threads and returns the results in index order.
fn run_parallel<T, F>(n: usize, f: F) -> Result<Vec<T>, CsvError>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    thread::scope(|s| {
        let f = &f;
        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            handles.push(
                thread::Builder::new()
                    .name(format!("csv-{i}"))
                    .spawn_scoped(s, move || f(i))?,
            );
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| CsvError::Internal("worker thread panicked"))
            })
            .collect()
    })
}

fn read_all(path: &Path) -> Result<Vec<u8>, CsvError> {
    let mut f = File::open(path)?;
    let len = usize::try_from(f.metadata()?.len())
        .map_err(|_| CsvError::Internal("file larger than address space"))?;
    let mut buf = Vec::new();
    // +1: lets `read_to_end` see EOF without a probing realloc.
    buf.try_reserve_exact(len.saturating_add(1))
        .map_err(|_| CsvError::OutOfMemory)?;
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// First true record boundary at or after `from`, given the quote state at `from`.
fn next_boundary(body: &[u8], from: usize, mut in_q: bool) -> usize {
    for (k, &b) in body[from..].iter().enumerate() {
        match b {
            b'"' => in_q = !in_q,
            b'\n' if !in_q => return from + k + 1,
            _ => {}
        }
    }
    body.len()
}

/// Auto-vectorizable counting pass: (number of '"', number of '\n').
fn count_quotes_newlines(chunk: &[u8]) -> (usize, usize) {
    let (mut q, mut nl) = (0usize, 0usize);
    for &b in chunk {
        q += (b == b'"') as usize;
        nl += (b == b'\n') as usize;
    }
    (q, nl)
}

fn parse_parallel(data: &[u8], opts: &Options) -> Result<Vec<ChunkOut>, CsvError> {
    // Optional UTF-8 BOM, then the header record.
    let off = if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    };
    if data.len() == off {
        return Err(CsvError::BadHeader);
    }
    let mut hf: Fields = [(0, 0); NFIELDS];
    let body_off = scan_record(data, off, &mut hf).map_err(|_| CsvError::BadHeader)?;
    for (k, name) in HEADER.iter().enumerate() {
        if &data[hf[k].0..hf[k].1] != name.as_bytes() {
            return Err(CsvError::BadHeader);
        }
    }
    let body = &data[body_off..];

    let n = (body.len() / opts.min_chunk_bytes.max(1))
        .min(opts.threads.max(1))
        .max(1);
    if n == 1 {
        let hint = count_quotes_newlines(body).1;
        return Ok(vec![parse_chunk(body, body_off, hint + 16)?]);
    }

    // Phase A (parallel): quote and newline counts of the raw, unaligned chunks.
    let raw: Vec<usize> = (0..=n)
        .map(|i| ((body.len() as u64 * i as u64) / n as u64) as usize)
        .collect();
    let counts = run_parallel(n, |i| count_quotes_newlines(&body[raw[i]..raw[i + 1]]))?;

    // Phase B (sequential, microseconds): exact quote state at each raw start,
    // then advance to the next true record boundary.
    let mut starts = vec![0usize; n + 1];
    starts[n] = body.len();
    let mut odd = false; // parity of '"' bytes in body[..raw[i]]
    for i in 0..n {
        if i > 0 {
            starts[i] = next_boundary(body, raw[i], odd);
        }
        odd ^= counts[i].0 % 2 == 1;
    }
    // starts[] is non-decreasing: the first boundary >= a larger point is never smaller.

    // Phase C (parallel): every chunk begins and ends on a record boundary.
    let results = run_parallel(n, |i| {
        let (a, b) = (starts[i], starts[i + 1]);
        parse_chunk(&body[a..b], body_off + a, counts[i].1 + 16)
    })?;
    results.into_iter().collect() // first error in file order wins: deterministic
}

/// Concatenates chunk columns and merges dictionaries in chunk order, which
/// preserves the global "first appearance" order of a sequential read.
fn merge(chunks: Vec<ChunkOut>) -> Result<Table, CsvError> {
    let total: usize = chunks.iter().map(|c| c.id.len()).sum();
    let mut t = Table {
        id: Vec::with_capacity(total),
        user_id: Vec::with_capacity(total),
        amount_cents: Vec::with_capacity(total),
        category: Vec::with_capacity(total),
        category_dict: Vec::new(),
    };
    let mut global: FnvMap<Vec<u8>, u16> = FnvMap::default();
    for c in chunks {
        // consuming iteration: each chunk is freed as soon as it is merged
        let mut remap = Vec::with_capacity(c.dict.len());
        for word in c.dict {
            let code = if let Some(&g) = global.get(&word) {
                g
            } else {
                let g = u16::try_from(t.category_dict.len())
                    .map_err(|_| CsvError::TooManyCategories)?;
                let s = String::from_utf8(word.clone()).map_err(|_| CsvError::InvalidUtf8)?;
                t.category_dict.push(s);
                global.insert(word, g);
                g
            };
            remap.push(code);
        }
        t.id.extend_from_slice(&c.id);
        t.user_id.extend_from_slice(&c.user_id);
        t.amount_cents.extend_from_slice(&c.cents);
        t.category.extend(c.code.iter().map(|&l| remap[l as usize]));
    }
    Ok(t)
}

pub fn read_csv_optimal_with(path: &Path, opts: &Options) -> Result<Table, CsvError> {
    let chunks = {
        let data = read_all(path)?;
        parse_parallel(&data, opts)?
    }; // `data` (file sized) is freed here, before the merge allocates the final columns
    merge(chunks)
}

pub fn read_csv_optimal(path: &Path) -> Result<Table, CsvError> {
    read_csv_optimal_with(path, &Options::default())
}

// =============================================================================
// SECTION 5: COMPARISON (correctness, timing, peak memory)
// =============================================================================
//
// Harness pieces: a tracking allocator (peak heap bytes), a tiny xorshift PRNG
// (no `rand` crate), a realistic CSV generator (quoted commas, escaped quotes,
// embedded newlines), an independent reference parser, edge-case checks,
// multi-thread determinism checks and the timing table.

// ---- peak-memory tracking allocator -----------------------------------------
// Counts bytes requested from the heap. It does not see page-cache or stack
// memory; it is exactly the "how much heap does this approach need" number.

struct Tracking;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// Every method forwards to `System` with the same arguments and only updates
// atomic counters, so all `GlobalAlloc` contracts are inherited from `System`.
// Explicit `unsafe {}` blocks inside the `unsafe fn`s keep this warning-free on
// edition 2024 (`unsafe_op_in_unsafe_fn`) and are harmless on edition 2021.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged, so the caller's contract holds for `System`.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` come from an earlier `alloc`/`realloc` of this allocator (which forwards to `System`).
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same contract as the caller's; arguments are forwarded unchanged.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let d = new_size - layout.size();
                let now = CURRENT.fetch_add(d, Relaxed) + d;
                PEAK.fetch_max(now, Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

// ---- deterministic input generation -----------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

struct TempFile(PathBuf);
impl TempFile {
    fn new(tag: &str) -> Self {
        TempFile(std::env::temp_dir().join(format!("triad_csv_{}_{}.csv", std::process::id(), tag)))
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Writes `rows` records with ids 1..=rows (the integrity check relies on that).
/// ~30% of notes are quoted: 15% with a comma, 8% with escaped quotes, 7% with a newline.
fn generate_csv(path: &Path, rows: usize, seed: u64) -> io::Result<u64> {
    const CATS: [&str; 8] = [
        "food", "travel", "rent", "health", "fun", "tools", "misc", "books",
    ];
    let mut rng = XorShift(seed | 1);
    let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
    writeln!(w, "id,user_id,note,amount,category")?;
    for i in 0..rows {
        let (a, b) = (rng.next(), rng.next());
        let user = a % 1_000_000;
        let cents = b % 10_000_000;
        let cat = CATS[((a >> 40) % 8) as usize];
        let note: Cow<str> = match (b >> 40) % 100 {
            0..=69 => Cow::Owned(format!("order {} shipped", a % 100_000)),
            70..=84 => Cow::Borrowed("\"Smith, John\""),
            85..=92 => Cow::Borrowed("\"said \"\"hello\"\" twice\""),
            _ => Cow::Borrowed("\"line one\nline two\""),
        };
        writeln!(
            w,
            "{},{},{},{}.{:02},{}",
            i + 1,
            user,
            note,
            cents / 100,
            cents % 100,
            cat
        )?;
    }
    w.flush()?;
    drop(w);
    Ok(fs::metadata(path)?.len())
}

// ---- independent reference parser -------------------------------------------
// Deliberately naive and obviously correct: a char-level RFC 4180 state machine
// over the whole text. Only used on small inputs as the source of truth.

fn reference_parse(text: &str) -> Result<Table, CsvError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_q = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_q = false;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' => in_q = true,
                ',' => fields.push(std::mem::take(&mut field)),
                '\r' if chars.peek() == Some(&'\n') => {}
                '\n' => {
                    fields.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut fields));
                }
                _ => field.push(c),
            }
        }
    }
    let bad = |reason: &'static str| CsvError::Malformed {
        byte_offset: 0,
        reason,
    }; // offsets not tracked
    if in_q {
        return Err(bad("unterminated quoted field"));
    }
    if !field.is_empty() || !fields.is_empty() {
        fields.push(field);
        records.push(fields);
    }
    let mut it = records.into_iter();
    let header = it.next().ok_or(CsvError::BadHeader)?;
    if !header.iter().map(String::as_str).eq(HEADER) {
        return Err(CsvError::BadHeader);
    }
    let mut b = TableBuilder::default();
    for rec in it {
        if rec.len() != NFIELDS {
            return Err(bad("wrong number of fields"));
        }
        let id = rec[0].parse::<u64>().map_err(|_| bad("invalid id"))?;
        let user = rec[1].parse::<u32>().map_err(|_| bad("invalid user_id"))?;
        let cents = parse_cents(rec[3].as_bytes()).ok_or_else(|| bad("invalid amount"))?;
        if rec[4].is_empty() {
            return Err(bad("empty category"));
        }
        b.push(id, user, cents, &rec[4])?;
    }
    Ok(b.finish())
}

fn same_outcome(expected: &Result<Table, CsvError>, got: &Result<Table, CsvError>) -> bool {
    match (expected, got) {
        (Ok(a), Ok(b)) => a == b,
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

/// (rows that exactly match the truth row with the same id, rows returned).
fn integrity(truth: &Table, got: &Table) -> (usize, usize) {
    let mut intact = 0;
    for i in 0..got.len() {
        let idx = (got.id[i] as usize).wrapping_sub(1); // ids are 1..=N by construction
        if idx < truth.len()
            && truth.user_id[idx] == got.user_id[i]
            && truth.amount_cents[idx] == got.amount_cents[i]
            && truth.category_name(idx) == got.category_name(i)
        {
            intact += 1;
        }
    }
    (intact, got.len())
}

type Reader = fn(&Path) -> Result<Table, CsvError>;
const APPROACHES: [(&str, Reader); 3] = [
    ("1-wrong", read_csv_wrong),
    ("2-common", read_csv_common),
    ("3-optimal", read_csv_optimal),
];

// ---- correctness ------------------------------------------------------------

fn run_correctness() -> Result<(), CsvError> {
    println!("== Correctness ==");
    const H: &str = "id,user_id,note,amount,category\n";
    let crlf = "id,user_id,note,amount,category\r\n1,10,hi,1.50,food\r\n2,11,yo,2.00,rent\r\n";
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("header only", H.into()),
        ("empty file", Vec::new()),
        (
            "no trailing newline",
            format!("{H}1,10,hello,1.50,food").into(),
        ),
        ("CRLF line endings", crlf.into()),
        (
            "UTF-8 BOM",
            format!("\u{feff}{H}1,10,hi,1.50,food\n").into(),
        ),
        (
            "quoted comma",
            format!("{H}1,10,\"Smith, John\",12.34,travel\n").into(),
        ),
        (
            "quoted newline",
            format!("{H}1,10,\"two\nlines\",12.34,food\n2,11,plain,0.99,rent\n").into(),
        ),
        (
            "escaped quotes",
            format!("{H}1,10,\"say \"\"hi\"\"\",5.00,food\n").into(),
        ),
        (
            "quoted numbers",
            format!("{H}\"1\",\"10\",x,\"2.50\",\"food\"\n").into(),
        ),
        (
            "amount without decimals",
            format!("{H}1,10,x,42,food\n").into(),
        ),
        (
            "19.99 stays 1999 cents",
            format!("{H}1,10,x,19.99,food\n").into(),
        ),
        (
            "unterminated quote",
            format!("{H}1,10,\"oops,1.00,food\n").into(),
        ),
        ("too few fields", format!("{H}1,10,x,1.00\n").into()),
        (
            "too many fields",
            format!("{H}1,10,x,1.00,food,extra\n").into(),
        ),
        ("non-numeric id", format!("{H}abc,10,x,1.00,food\n").into()),
        (
            "blank line inside",
            format!("{H}1,10,x,1.00,food\n\n2,11,y,2.00,rent\n").into(),
        ),
    ];

    println!(
        "{:<26} | {:<11} | {:<7} | {:<7} | {:<7}",
        "case", "expected", "wrong", "common", "optimal"
    );
    let mut hard_failures = 0;
    for (idx, (name, bytes)) in cases.iter().enumerate() {
        let tmp = TempFile::new(&format!("edge{idx}"));
        fs::write(tmp.path(), bytes)?;
        let text = String::from_utf8_lossy(bytes).into_owned();
        let expected = reference_parse(&text);
        let exp_label = match &expected {
            Ok(t) => format!("ok ({} rows)", t.len()),
            Err(_) => "error".to_string(),
        };
        let mut cells = Vec::new();
        for (k, (_, reader)) in APPROACHES.iter().enumerate() {
            let pass = same_outcome(&expected, &reader(tmp.path()));
            if !pass && k > 0 {
                hard_failures += 1;
            }
            cells.push(if pass { "pass" } else { "FAIL" });
        }
        println!(
            "{:<26} | {:<11} | {:<7} | {:<7} | {:<7}",
            name, exp_label, cells[0], cells[1], cells[2]
        );
    }

    // Generated file: heavy on quoted commas / quotes / newlines.
    let rows = 20_000;
    let tmp = TempFile::new("corr");
    generate_csv(tmp.path(), rows, 7)?;
    let expected = reference_parse(&fs::read_to_string(tmp.path())?)?;
    let (intact, returned) = integrity(&expected, &read_csv_wrong(tmp.path())?);
    println!(
        "\ngenerated {rows} rows: wrong returned {returned} rows, only {intact} match the truth \
         ({} garbage/missing)",
        rows.max(returned) - intact
    );
    if read_csv_common(tmp.path())? != expected {
        hard_failures += 1;
        println!("common  : MISMATCH vs reference");
    } else {
        println!("common  : identical to reference");
    }
    // Force many tiny chunks so boundaries land inside quoted multi-line fields.
    for threads in [1, 2, 3, 5, 8, 13] {
        let opts = Options {
            threads,
            min_chunk_bytes: 2048,
        };
        let ok = read_csv_optimal_with(tmp.path(), &opts)? == expected;
        if !ok {
            hard_failures += 1;
        }
        println!(
            "optimal : threads={threads:<2} min_chunk=2 KiB -> {}",
            if ok {
                "identical to reference"
            } else {
                "MISMATCH"
            }
        );
    }
    if hard_failures > 0 {
        return Err(CsvError::Internal(
            "common/optimal disagree with the reference",
        ));
    }
    println!("=> common and optimal match the reference; wrong does not.");
    Ok(())
}

// ---- benchmark --------------------------------------------------------------

const REPS: usize = 3;
const TIME_BUDGET_S: f64 = 12.0; // skip an approach if its projected time (with warm-up + reps) exceeds this
const MEM_BUDGET: usize = 1_500_000_000; // skip an approach if its projected peak heap exceeds this

struct Sample {
    best: Duration,
    peak: usize,
}

struct Hist {
    rows: usize,
    secs: f64,
    peak: usize,
}

/// One warm-up run, then `REPS` timed runs; reports best time and worst peak heap.
fn measure(mut f: impl FnMut() -> Result<Table, CsvError>) -> Result<(Sample, Table), CsvError> {
    drop(black_box(f()?)); // warm-up: page cache, allocator, branch predictors
    let mut best = Duration::MAX;
    let mut peak = 0usize;
    let mut last = None;
    for _ in 0..REPS {
        let base = CURRENT.load(Relaxed);
        PEAK.store(base, Relaxed);
        let t0 = Instant::now();
        let table = black_box(f()?);
        let dt = t0.elapsed();
        peak = peak.max(PEAK.load(Relaxed).saturating_sub(base));
        best = best.min(dt);
        last = Some(table);
    }
    Ok((
        Sample { best, peak },
        last.ok_or(CsvError::Internal("REPS must be >= 1"))?,
    ))
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn print_row(
    rows: usize,
    file_bytes: u64,
    name: &str,
    s: Option<&Sample>,
    common_secs: Option<f64>,
    status: &str,
) {
    let file_mib = file_bytes as f64 / (1024.0 * 1024.0);
    match s {
        Some(s) => {
            let secs = s.best.as_secs_f64();
            let speedup = common_secs.map_or("n/a".to_string(), |c| format!("{:.1}x", c / secs));
            println!(
                "{:>10} {:>9.1} | {:<9} | {:>9.1} | {:>8.0} | {:>9} | {:>9.1} | {}",
                rows,
                file_mib,
                name,
                secs * 1000.0,
                file_mib / secs,
                speedup,
                mib(s.peak),
                status
            );
        }
        None => {
            let msg = if status.is_empty() {
                "skipped (too slow / too much memory)"
            } else {
                status
            };
            println!(
                "{:>10} {:>9.1} | {:<9} | {:>9} | {:>8} | {:>9} | {:>9} | {}",
                rows, file_mib, name, "-", "-", "-", "-", msg
            );
        }
    }
}

fn run_benchmark(sizes: &[usize]) -> Result<(), CsvError> {
    println!(
        "\n== Benchmark: best of {REPS} after 1 warm-up, warm page cache, {} thread(s) available ==",
        Options::default().threads
    );
    println!(
        "{:>10} {:>9} | {:<9} | {:>9} | {:>8} | {:>9} | {:>9} | status",
        "rows", "file MiB", "approach", "time ms", "MiB/s", "vs common", "peak MiB"
    );
    let mut hist: [Option<Hist>; 3] = [None, None, None];
    let mut last_optimal: Option<Table> = None;
    let mut wrong_error: Option<String> = None;

    for &rows in sizes {
        let tmp = TempFile::new(&format!("bench{rows}"));
        let file_bytes = generate_csv(tmp.path(), rows, 42)?;

        let mut samples: [Option<(Sample, Table)>; 3] = [None, None, None];
        for (k, (_, reader)) in APPROACHES.iter().enumerate() {
            if let Some(h) = &hist[k] {
                let scale = rows as f64 / h.rows as f64;
                if h.secs * scale * (REPS + 1) as f64 > TIME_BUDGET_S
                    || h.peak as f64 * scale > MEM_BUDGET as f64
                {
                    continue; // projected to blow the budget: report "skipped" instead of hanging
                }
            }
            if k == 0 && wrong_error.is_some() {
                continue; // it already crashed on a smaller input; do not retry
            }
            let (s, t) = match measure(|| reader(tmp.path())) {
                Ok(x) => x,
                // The wrong approach may fail outright on corrupted data (for example
                // garbage "categories" overflowing the dictionary). That is a result,
                // not a harness failure.
                Err(e) if k == 0 => {
                    wrong_error = Some(format!("FAIL: crashed with error: {e}"));
                    continue;
                }
                Err(e) => return Err(e),
            };
            hist[k] = Some(Hist {
                rows,
                secs: s.best.as_secs_f64(),
                peak: s.peak,
            });
            samples[k] = Some((s, t));
        }
        let [wrong, common, optimal] = samples;

        // Truth for this size: the common approach (validated against the reference
        // above), or a single-threaded optimal read if common was skipped.
        let fallback;
        let truth: &Table = match &common {
            Some((_, t)) => t,
            None => {
                fallback = read_csv_optimal_with(
                    tmp.path(),
                    &Options {
                        threads: 1,
                        ..Options::default()
                    },
                )?;
                &fallback
            }
        };
        if let Some((_, o)) = &optimal {
            if o != truth {
                return Err(CsvError::Internal("optimal disagrees with the baseline"));
            }
        }

        let common_secs = common.as_ref().map(|(s, _)| s.best.as_secs_f64());
        let wrong_status = match &wrong {
            Some((_, t)) => {
                let (intact, returned) = integrity(truth, t);
                format!(
                    "FAIL: {intact}/{} rows intact, {returned} returned",
                    truth.len()
                )
            }
            None => wrong_error.clone().unwrap_or_default(),
        };
        print_row(
            rows,
            file_bytes,
            "1-wrong",
            wrong.as_ref().map(|x| &x.0),
            common_secs,
            &wrong_status,
        );
        print_row(
            rows,
            file_bytes,
            "2-common",
            common.as_ref().map(|x| &x.0),
            common_secs,
            "ok (== reference)",
        );
        print_row(
            rows,
            file_bytes,
            "3-optimal",
            optimal.as_ref().map(|x| &x.0),
            common_secs,
            "ok (== reference)",
        );
        println!("{}", "-".repeat(112));

        if let Some((_, t)) = optimal {
            last_optimal = Some(t);
        }
    }

    println!(
        "Complexity: wrong O(n) time + ~6n allocs + 5-10x file in RAM | common O(n) + ~6n allocs, streaming |"
    );
    println!(
        "            optimal O(n) work, O(n/cores) wall clock, 0 allocs per row, ~file + 22 B/row in RAM"
    );

    if let Some(t) = last_optimal {
        println!(
            "\nUsing the result (optimal table, {} rows, total {:.2}):",
            t.len(),
            t.total_cents() as f64 / 100.0
        );
        for (cat, cents) in t.total_by_category().into_iter().take(3) {
            println!("  {cat:<8} {:>16.2}", cents as f64 / 100.0);
        }
    }
    Ok(())
}

// ---- verdict ----------------------------------------------------------------
// VERDICT
//   * Wrong is not "slow but usable": it returns silently corrupted data on
//     any file with quoted commas/newlines and truncates money. Correctness
//     failures beat any speed number.
//   * Common is the right default for small files or when memory must stay
//     tiny and simplicity wins. It is correct but allocation-bound.
//   * Optimal wins by removing per-row allocations, decoding only the needed
//     columns as bytes, and (with more than one core) parallelizing safely via
//     quote-parity chunking. Expect the gap to GROW with core count; the
//     sandbox numbers show the single-thread gain only when it has 1 core.
//   * On a cold cache every approach becomes disk-bound: then prefer streaming
//     blocks with overlapped I/O, keeping the same scan/parse core.

// =============================================================================
// SECTION 6: ENTRY POINT
// =============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build detected; timings are meaningless. Use --release / -C opt-level=3."
        );
    }
    let max_rows = match std::env::args().nth(1) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 1_000)
            .ok_or("usage: csv_fast_read [max_rows >= 1000]")?,
        None => 2_000_000,
    };
    let mut sizes = Vec::new();
    let mut s = 10_000usize;
    while s < max_rows {
        sizes.push(s);
        s *= 10;
    }
    sizes.push(max_rows);

    run_correctness()?;
    run_benchmark(&sizes)?;
    Ok(())
}
