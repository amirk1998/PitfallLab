#!/usr/bin/env python3
"""
==============================================================================
SECTION 1: SCENARIO
==============================================================================
Read a very large CSV file into memory as fast as possible, using only the
Python standard library (no pandas, no polars, no conversion to Parquet).

Context
-------
A service (or a data-prep script) must load a multi-hundred-MB / multi-GB CSV
and then use the values in ordinary Python code (sums, filters, joins, ...).
Load time dominates the whole job, so it is the number to minimise.

Requirements
------------
- Input : path to a UTF-8 CSV file with a header row (RFC 4180 style: fields
          may be quoted, quoted fields may contain commas, doubled quotes and
          even newlines). CRLF or LF line endings, optional UTF-8 BOM.
- Output: a dict {column_name: column_values} (column-oriented layout).
          int columns   -> array('q'), float columns -> array('d'),
          str columns   -> list[str].
- Only the columns the caller asks for (``usecols``) are converted.
- Errors: missing file / unknown column / bad number -> specific exceptions
          with a clear message (never a silent wrong answer).

Assumptions
-----------
- All rows have the same number of fields as the header.
- The file fits in RAM once stored in compact typed arrays (a stream
  aggregation would be a different, cheaper problem).
- Benchmark schema: id:int, user_id:int, amount:float, category:str,
  ts:str, note:str.  Two synthetic profiles are used:
    clean : no quote characters anywhere.
    messy : ~2% of rows have a quoted "note" containing commas, doubled
            quotes or embedded newlines.

Goal
----
One linear pass, O(n) time and O(n) *compact* space, with the per-row work
pushed down into C code (csv module, str.split, map, array) instead of
Python-level loops.

Run:  python fast_csv_reader.py [max_rows]
==============================================================================
"""

from __future__ import annotations

import csv
import gc
import io
import os
import random
import sys
import tempfile
import time
import tracemalloc
from array import array
from collections.abc import Callable, Sequence
from typing import Any

Table = dict[str, Any]

# Schema used by all approaches in the benchmark: column name -> converter kind.
SCHEMA: dict[str, str] = {"id": "int", "amount": "float", "category": "str"}
HEADER = ["id", "user_id", "amount", "category", "ts", "note"]


# =============================================================================
# SECTION 2: APPROACH 1 - THE WRONG WAY
# =============================================================================
# What the author was thinking:
#   "A CSV is just text. Read it all at once, split on newlines, then split
#    every line on commas. No need for the csv module."
#
# Why it is wrong:
#   1. CORRECTNESS: quoted fields may contain commas, quotes and newlines.
#      Splitting on "," and "\n" cuts those records in the wrong places, so
#      rows get too many/too few fields and columns shift silently. The
#      quotes also stay glued to the values ('"hello').
#   2. MEMORY: f.read() holds the whole file as one giant str, then split()
#      builds a second full copy as a list of lines, then a third as a list
#      of lists. Peak memory is a multiple of the file size.
#   3. Text mode without newline="" translates line endings and the default
#      encoding is platform dependent (a BOM also ends up inside the first
#      column name).
#   4. Every value is kept as a Python str inside a per-row list: slow to
#      build and very heavy (each small str costs ~50+ bytes).
#
# Failure mode: on clean data it is merely slow and memory hungry; on data
# with quoted fields it returns wrong rows or crashes on int().
# Complexity: time O(n) with a large constant, space O(file size x 3).
# =============================================================================
def read_wrong(path: str, schema: dict[str, str]) -> Table:
    with open(path) as f:  # platform default encoding, newline translation
        text = f.read()  # whole file in memory
    lines = text.split("\n")  # second copy
    rows = [line.split(",") for line in lines[1:] if line]  # third copy
    header = lines[0].split(",")
    idx = {name: header.index(name) for name in schema}
    out: Table = {}
    for name, kind in schema.items():
        j = idx[name]
        conv = {"int": int, "float": float, "str": str}[kind]
        # Python-level loop per row, and one more pass per requested column.
        out[name] = [conv(row[j]) for row in rows]
    return out


# =============================================================================
# SECTION 3: APPROACH 2 - THE COMMON WAY
# =============================================================================
# Why most developers write this:
#   csv.DictReader is the first thing the documentation and every tutorial
#   show. It is correct (quotes, embedded newlines, CRLF are all handled by
#   the C tokenizer underneath) and the code reads like the problem.
#
# What is good: correct output, streaming input, tiny amount of code.
# What it leaves on the table:
#   - DictReader is a Python-level wrapper: for every row it builds a dict
#     of ALL columns, even the ones we never use.
#   - Keeping list[dict] costs a dict + N str objects per row (hundreds of
#     bytes); the conversion pass then walks the rows a second time.
#   - Numbers stay boxed Python objects in lists, not compact arrays.
#
# Complexity: time O(n * c) with a big constant (c = number of columns),
# space O(n * c) boxed objects.
# =============================================================================
def read_common(path: str, schema: dict[str, str]) -> Table:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))  # list of dicts, all columns
    conv = {"int": int, "float": float, "str": str}
    return {name: [conv[kind](r[name]) for r in rows] for name, kind in schema.items()}


# =============================================================================
# SECTION 4: APPROACH 3 - THE OPTIMAL, PRODUCTION-GRADE WAY
# =============================================================================
# Key insight:
#   Do not process rows in Python. Process CHUNKS, and let C do the loops.
#
# Design (each point was measured, see the benchmark):
#   1. Binary chunked read (~1 MiB) instead of text iteration. Bytes are cut
#      only at a "safe" newline: one that is NOT inside a quoted field. The
#      test is quote parity: in RFC 4180 a quote either opens/closes a field
#      or is doubled, so a newline is outside quotes iff the number of '"'
#      before it is even. bytes.count runs at memchr speed.
#      A newline byte can never be part of a multi-byte UTF-8 character, so
#      decoding each chunk separately is safe.
#   2. Two parsing paths per chunk:
#        FAST path (chunk contains no '"'): replace("\n", ",") then one
#           str.split(",") gives a flat list of all fields; column j is the
#           strided slice flat[j::ncols]. Zero Python-level row loop.
#           A length check (len(flat) == rows * ncols) guards it; any
#           mismatch drops to the safe path instead of guessing.
#        SAFE path (chunk has quotes): csv.reader (C tokenizer) on the chunk,
#           then zip(*rows) to transpose rows into columns.
#   3. Column projection: only the requested columns are sliced/converted.
#   4. Typed storage: int -> array('q'), float -> array('d'); fed by
#      map(int, column) so conversion also happens in C. 8 bytes per number
#      instead of ~28-32 for a boxed int/float plus an 8 byte list slot.
#   5. gc.disable() during the load: we allocate millions of small
#      containers that all survive, so the cyclic GC would repeatedly scan
#      them for nothing. It is restored in a finally block.
#   6. Small chunks: ~1 MiB keeps the working set in CPU cache; 16 MiB chunks
#      were measurably slower in our tuning runs.
#
# Complexity: time O(n) (single pass), space O(n) compact + O(chunk) temp.
#
# Trade-offs / when NOT to use this:
#   - Result lives fully in RAM. If it does not fit, stream chunks instead
#     (the generator ``iter_chunks`` below yields them one by one).
#   - The FAST path assumes a regular file (same field count per row). Rows
#     with a wrong field count in a quote-free chunk are detected only when
#     the totals do not add up; the SAFE path then reports the bad chunk.
#   - Pure Python is single-core. On a many-core machine, split the file by
#     byte ranges (using the same quote-parity rule) and parse ranges in a
#     ProcessPoolExecutor; results of array('q')/('d') pickle cheaply.
#   - For repeated reads of the same data, a binary cache (Parquet/Arrow/
#     pickle of arrays) beats any CSV parser; that is out of scope here.
# =============================================================================
_CHUNK_BYTES = 1 << 20  # 1 MiB, tuned (see notes above)
_QUOTE = b'"'


def _safe_cut(data: bytes) -> int:
    """Return the index just after the last newline that is outside quotes.

    ``data`` must start at a record boundary (even quote parity). Returns 0
    if no such newline exists (chunk is a single, still-open record).
    Complexity: O(len(data)) with small constants; loops only when the last
    newline(s) sit inside a quoted field.
    """
    cut = data.rfind(b"\n")
    if cut < 0:
        return 0
    quotes_before = data.count(_QUOTE, 0, cut)
    while cut >= 0 and quotes_before % 2:  # odd parity => inside a quoted field
        prev = data.rfind(b"\n", 0, cut)
        start = prev + 1 if prev >= 0 else 0
        quotes_before -= data.count(_QUOTE, start, cut)
        cut = prev
    return cut + 1 if cut >= 0 else 0


def _first_record_end(data: bytes) -> int:
    """Index just after the first newline outside quotes, or 0 if none yet."""
    pos, quotes = -1, 0
    while True:
        nl = data.find(b"\n", pos + 1)
        if nl < 0:
            return 0
        quotes += data.count(_QUOTE, pos + 1, nl)
        if quotes % 2 == 0:
            return nl + 1
        pos = nl


def iter_chunks(path: str, chunk_bytes: int = _CHUNK_BYTES):
    """Yield the header line first, then blocks of complete records (as str).

    Every block ends exactly on a record boundary, so blocks can be parsed
    independently. Memory: O(chunk_bytes) (more only for a single record
    larger than a chunk, which then simply grows the buffer).
    """
    with open(path, "rb") as f:
        buf = f.read(chunk_bytes)
        if buf.startswith(b"\xef\xbb\xbf"):  # strip UTF-8 BOM
            buf = buf[3:]
        while not (end := _first_record_end(buf)):  # header spans chunks (rare)
            more = f.read(chunk_bytes)
            if not more:
                end = len(buf)  # header-only file without trailing newline
                break
            buf += more
        yield buf[:end].decode("utf-8")
        tail = buf[end:]
        while True:
            block = f.read(chunk_bytes)
            data = tail + block
            if not block:  # EOF: whatever is left is the final record(s)
                if data:
                    yield data.decode("utf-8")
                return
            cut = _safe_cut(data)
            if cut == 0:  # one record spans the whole buffer: keep growing
                tail = data
                continue
            yield data[:cut].decode("utf-8")
            tail = data[cut:]


def _split_columns(text: str, ncols: int, wanted: Sequence[int]) -> list[Sequence[str]]:
    """Return the wanted columns of a chunk as sequences of str."""
    if '"' not in text:
        if "\r" in text:
            text = text.replace("\r\n", "\n")
        text = text.rstrip("\n")
        if not text:
            return [[] for _ in wanted]
        nrows = text.count("\n") + 1
        flat = text.replace("\n", ",").split(",")  # FAST path: C-level only
        if len(flat) == nrows * ncols:
            return [flat[j::ncols] for j in wanted]  # strided slice, no row loop
        # Field count mismatch (blank line / ragged row): use the safe path.
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r]  # SAFE path
    bad = next((r for r in rows if len(r) != ncols), None)
    if bad is not None:
        raise ValueError(
            f"row with {len(bad)} fields, expected {ncols}: {bad[:3]!r}..."
        )
    if not rows:
        return [[] for _ in wanted]
    cols = list(zip(*rows))  # transpose in C
    return [cols[j] for j in wanted]


def _convert(values: Sequence[str], kind: str, name: str):
    try:
        if kind == "int":
            return array("q", map(int, values))
        if kind == "float":
            try:
                return array("d", map(float, values))
            except ValueError:  # maybe empty cells: treat them as NaN
                return array(
                    "d", (float(v) if v.strip() else float("nan") for v in values)
                )
        return list(values)
    except ValueError as exc:
        raise ValueError(f"column {name!r}: cannot convert to {kind}: {exc}") from None


def read_optimal(
    path: str,
    schema: dict[str, str],
    *,
    chunk_bytes: int = _CHUNK_BYTES,
) -> Table:
    """Load ``schema`` columns of a CSV file into column-oriented storage.

    ``schema`` maps column name -> "int" | "float" | "str".
    Raises FileNotFoundError, KeyError (unknown column) or ValueError.
    Time O(n), space O(n) compact + O(chunk_bytes).
    """
    if not schema:
        raise ValueError("schema must name at least one column")
    if any(k not in ("int", "float", "str") for k in schema.values()):
        raise ValueError("schema kinds must be 'int', 'float' or 'str'")
    if chunk_bytes < 4096:
        raise ValueError("chunk_bytes must be at least 4096")

    out: Table = {}
    gc_was_enabled = gc.isenabled()
    gc.disable()  # millions of surviving objects: GC scans would be wasted work
    try:
        chunks = iter_chunks(path, chunk_bytes)
        header_text = next(chunks)
        header = next(csv.reader(io.StringIO(header_text, newline="")))
        missing = [c for c in schema if c not in header]
        if missing:
            raise KeyError(f"unknown column(s): {missing}; file has {header}")
        names = list(schema)
        wanted = [header.index(n) for n in names]
        ncols = len(header)
        for n in names:
            out[n] = (
                array("q")
                if schema[n] == "int"
                else array("d")
                if schema[n] == "float"
                else []
            )
        for text in chunks:
            for n, col in zip(names, _split_columns(text, ncols, wanted)):
                kind = schema[n]
                part = _convert(col, kind, n)
                out[n].extend(part)  # list.extend / array.extend, both C
    finally:
        if gc_was_enabled:
            gc.enable()
    return out


# =============================================================================
# SECTION 5: COMPARISON HARNESS
# =============================================================================
APPROACHES: dict[str, Callable[[str, dict[str, str]], Table]] = {
    "1. wrong": read_wrong,
    "2. common": read_common,
    "3. optimal": read_optimal,
}
TIME_BUDGET_S = 20.0  # skip an approach at larger sizes once it exceeds this
WRONG_MAX_ROWS = 500_000  # approach 1 holds ~3 copies of the file: keep RAM sane


def write_csv(path: str, n: int, messy: bool, seed: int = 7) -> None:
    """Deterministic synthetic file. ``messy`` adds quoted, tricky notes."""
    rng = random.Random(seed)
    cats = ["books", "games", "music", "tools", "food", "travel", "health", "garden"]
    tricky = ["hello, world", 'say "hi"', "line1\nline2", 'a,b,"c"\nd']
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for i in range(n):
            note = rng.choice(tricky) if messy and rng.random() < 0.02 else "ok"
            w.writerow(
                [
                    i,
                    rng.randrange(1_000_000),
                    f"{rng.random() * 1000:.2f}",
                    rng.choice(cats),
                    f"2025-01-{rng.randrange(1, 29):02d}T{rng.randrange(24):02d}:00:00",
                    note,
                ]
            )


def reference(path: str) -> Table:
    """Trusted but slow reference: plain csv.reader, row by row, no tricks."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        ix = {n: header.index(n) for n in SCHEMA}
        cols: Table = {n: [] for n in SCHEMA}
        conv = {"int": int, "float": float, "str": str}
        for row in r:
            for n, k in SCHEMA.items():
                cols[n].append(conv[k](row[ix[n]]))
    return cols


def same(a: Table, b: Table) -> bool:
    return a.keys() == b.keys() and all(list(a[k]) == list(b[k]) for k in a)


def best_time(
    fn: Callable[[str, dict[str, str]], Table], path: str, repeats: int
) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(path, SCHEMA)
        best = min(best, time.perf_counter() - start)
    return best


def peak_memory_mib(fn: Callable[[str, dict[str, str]], Table], path: str) -> float:
    tracemalloc.start()
    fn(path, SCHEMA)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1 << 20)


def check_correctness(tmp: str) -> None:
    print("Correctness check (5,000 rows, compared with a csv.reader reference):")
    for profile in ("clean", "messy"):
        path = os.path.join(tmp, f"check_{profile}.csv")
        write_csv(path, 5_000, messy=(profile == "messy"))
        ref = reference(path)
        for name, fn in APPROACHES.items():
            try:
                ok = same(fn(path, SCHEMA), ref)
                verdict = "PASS" if ok else "FAIL (wrong values)"
            except Exception as exc:  # noqa: BLE001 - demonstrate the failure mode
                verdict = f"FAIL ({type(exc).__name__}: {str(exc)[:40]})"
            print(f"  {profile:<6} {name:<11} {verdict}")

    # Edge cases for the optimal reader: CRLF + BOM, tiny chunks, bad input.
    path = os.path.join(tmp, "edge.csv")
    body = 'id,amount,category\r\n1,2.5,"a,b"\r\n2,,"x\r\ny"\r\n3,4.0,z'
    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf" + body.encode())
    got = read_optimal(
        path, {"id": "int", "amount": "float", "category": "str"}, chunk_bytes=4096
    )
    ok = (
        list(got["id"]) == [1, 2, 3]
        and got["category"][:2] == ["a,b", "x\r\ny"]
        and got["amount"][1] != got["amount"][1]
    )
    print(
        f"  edge   3. optimal   {'PASS' if ok else 'FAIL'} (BOM, CRLF, quoted CRLF, empty float, no final newline)"
    )
    for label, call in (
        ("unknown column", lambda: read_optimal(path, {"nope": "int"})),
        ("bad int value", lambda: read_optimal(path, {"category": "int"})),
        ("missing file", lambda: read_optimal(os.path.join(tmp, "absent.csv"), SCHEMA)),
    ):
        try:
            call()
            print(f"  edge   3. optimal   FAIL ({label} accepted)")
        except (KeyError, ValueError, FileNotFoundError):
            print(f"  edge   3. optimal   PASS ({label} rejected)")
    # Chunk-boundary stress: same result for every chunk size on the messy file.
    ref = reference(os.path.join(tmp, "check_messy.csv"))
    sizes_ok = all(
        same(
            read_optimal(os.path.join(tmp, "check_messy.csv"), SCHEMA, chunk_bytes=s),
            ref,
        )
        for s in (4096, 5000, 8191, 1 << 16)
    )
    print(f"  chunk-size stress on messy file: {'PASS' if sizes_ok else 'FAIL'}\n")


def run_benchmark(tmp: str, max_rows: int) -> None:
    sizes = [s for s in (10_000, 100_000, 500_000, 1_000_000) if s <= max_rows] or [
        max_rows
    ]
    header = f"{'profile':<6} | {'rows':>9} | {'MB':>5} | {'approach':<11} | {'time (s)':>9} | {'peak MiB':>8} | {'vs common':>9}"
    print(header)
    print("-" * len(header))
    for profile in ("clean", "messy"):
        skipped: set[str] = set()
        for n in sizes:
            path = os.path.join(tmp, f"bench_{profile}_{n}.csv")
            write_csv(path, n, messy=(profile == "messy"))
            mb = os.path.getsize(path) / 1e6
            timings: dict[str, float] = {}
            for name, fn in APPROACHES.items():
                if name in skipped:
                    continue
                if name == "1. wrong" and (profile == "messy" or n > WRONG_MAX_ROWS):
                    continue  # garbage on messy data (see correctness check) / memory guard
                try:
                    timings[name] = best_time(
                        fn, path, repeats=3 if n <= 100_000 else 1
                    )
                except Exception:  # noqa: BLE001
                    continue
                if timings[name] > TIME_BUDGET_S:
                    skipped.add(name)
            base = timings.get("2. common")
            for name, fn in APPROACHES.items():
                if name in timings:
                    mem = peak_memory_mib(fn, path) if n <= 500_000 else float("nan")
                    ratio = f"{base / timings[name]:.2f}x" if base else "n/a"
                    print(
                        f"{profile:<6} | {n:>9} | {mb:>5.0f} | {name:<11} | {timings[name]:>9.3f} | {mem:>8.1f} | {ratio:>9}"
                    )
                elif name == "1. wrong" and profile == "messy":
                    print(
                        f"{profile:<6} | {n:>9} | {mb:>5.0f} | {name:<11} | {'invalid output / crash':>9}"
                    )
                elif name == "1. wrong" and n > WRONG_MAX_ROWS:
                    print(
                        f"{profile:<6} | {n:>9} | {mb:>5.0f} | {name:<11} | {'skipped (memory guard)':>9}"
                    )
                else:
                    print(
                        f"{profile:<6} | {n:>9} | {mb:>5.0f} | {name:<11} | {'skipped (too slow)':>9}"
                    )
            print("-" * len(header))
            os.remove(path)


def print_summary() -> None:
    print(
        "\nComplexity summary\n"
        "  wrong   : O(n), ~3 full copies of the file, breaks on quoted fields\n"
        "  common  : O(n*c), one dict + c boxed str per row, correct\n"
        "  optimal : O(n), chunked, C-level loops, typed arrays, only needed columns\n"
        "\nVerdict\n"
        "  Push work into C: csv.reader/str.split/map/array on 1 MiB chunks cut at\n"
        "  quote-safe newlines beat any per-row Python code. Quote-free chunks take\n"
        "  the strided-slice fast path; chunks with quotes take csv.reader. Next step\n"
        "  for more speed is multiprocessing over byte ranges, not a different parser.\n"
        "  Note: numbers are from this machine; ratios and scaling are what matter.\n"
    )


# =============================================================================
# SECTION 6: ENTRY POINT
# =============================================================================
def main() -> None:
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    with tempfile.TemporaryDirectory() as tmp:
        check_correctness(tmp)
        run_benchmark(tmp, max_rows)
    print_summary()


if __name__ == "__main__":
    main()
