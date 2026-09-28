# PitfallLab Scenarios

The catalog of every scenario in the repository: exact name, focus, how the three tiers differ,
measured results, and where the code lives.

**Legend:** ✅ available &nbsp;·&nbsp; ⏳ planned &nbsp;·&nbsp; 💡 idea (not started)

**Focus tags:** `Performance` · `Security` · `Correctness`

---

## 1. Catalog

| ID  | Scenario            | Focus                     | Category           | Core technique                                                  | Python | Go  | Rust |
| :-: | ------------------- | ------------------------- | ------------------ | --------------------------------------------------------------- | :----: | :-: | :--: |
| 001 | **Fast CSV Reader** | Performance · Correctness | File I/O · Parsing | Quote-parity chunking, allocation-free fast path, typed columns |   ✅   | ✅  |  ⏳  |

---

## 2. Scenario cards

### 001 · Fast CSV Reader

> Load a very large CSV into memory as fast as possible using only the standard library
> (no pandas, no polars, no Parquet conversion; in Go, no third-party module).

|              |                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Focus**    | Performance · Correctness (Go version adds input hardening for untrusted files)                                                 |
| **Category** | File I/O · Parsing                                                                                                              |
| **Input**    | UTF-8 CSV with header; quoted fields may hold commas, doubled quotes and newlines; CRLF or LF; optional BOM                     |
| **Output**   | Column store: typed numeric arrays/slices for int and float columns, string list for text; only requested columns are converted |
| **Goal**     | One linear pass, per-row work pushed out of interpreted/allocating code, compact memory                                         |
| **Python**   | [`python/001_fast_csv_reader/fast_csv_reader.py`](python/001_fast_csv_reader/fast_csv_reader.py)                                |
| **Go**       | [`go/001_fast_csv_reader/fast_csv_reader.go`](go/001_fast_csv_reader/fast_csv_reader.go)                                        |
| **Run**      | `python python/001_fast_csv_reader/fast_csv_reader.py 1000000` · `cd go && go run ./001_fast_csv_reader 1000000`                |

**The three tiers**

|    Tier     | Python                                                                                                      | Go                                                                                                                                                                       | Why it fails or leaves speed on the table                                                                |
| :---------: | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
|  1 · Wrong  | `read().split("\n")`, then `split(",")`                                                                     | `os.ReadFile` + `strings.Split` twice, `strconv` errors ignored                                                                                                          | Quoted fields shred rows (Go: panic or silent zeros), about 2-3 copies of the file in RAM, no size limit |
| 2 · Common  | `csv.DictReader` into a list of dicts                                                                       | `encoding/csv` `ReadAll()` then convert                                                                                                                                  | Correct, but every column of every row becomes an object/string first; no BOM handling; single core      |
| 3 · Optimal | 1 MiB chunks cut at quote-safe newlines, strided-slice fast path or `csv.reader`, typed `array`s, GC paused | Same chunking, allocation-free per-line fast path, RFC 4180 state machine for quoted records, interned text, per-chunk typed slices, optional ordered goroutine pipeline | Single core in Python; result must fit in RAM                                                            |

**Measured results: Python** (sandbox, 1 core, 1M rows ≈ 50 MB)

| Profile                | Common | Optimal |  Speedup  | Peak memory at 500k rows (Common → Optimal) |
| ---------------------- | -----: | ------: | :-------: | :-----------------------------------------: |
| clean (no quotes)      | 2.58 s |  0.64 s | **~4.0×** |                309 → 45 MiB                 |
| messy (2% quoted rows) | 2.18 s |  0.95 s | **~2.3×** |                309 → 49 MiB                 |

**Measured results: Go** (same sandbox, 1 core, `Workers: 1`, 1M rows ≈ 50 MB, three repeated runs)

| Profile                |        Common |       Optimal |     Speedup     | Allocated (Common → Optimal) | Live result heap |
| ---------------------- | ------------: | ------------: | :-------------: | :--------------------------: | :--------------: |
| clean (no quotes)      | 0.73 – 0.81 s | 0.17 – 0.18 s | **~4.0 – 4.4×** |         423 → 67 MiB         |   79 → 31 MiB    |
| messy (2% quoted rows) | 0.64 – 0.80 s | 0.17 – 0.18 s | **~3.8 – 4.3×** |         423 → 68 MiB         |   79 → 31 MiB    |

The wrong approach returns invalid output on the messy profile in both languages (Python raises
`ValueError`, Go panics with an index-out-of-range). Timings come from a shared sandbox and are noisy (one earlier run of the same Go code on the messy
profile showed only ~2.7×); the ratios and scaling trend are what matter. The Go parallel
pipeline (`Workers > 1`) was verified for correctness and data-race freedom (`go run -race`), but
its speed-up could not be measured because the sandbox has a single CPU; run the Go file on a
multi-core machine and the extra `3b. optimal xN` row appears in the table automatically.

**Key ideas**

- **Quote-parity cut:** a newline is outside a quoted field exactly when the number of `"` before it is even, so the input can be cut into chunks that are parsed independently.
- **Fast path:** lines without quotes skip the general tokenizer. Python slices columns with a stride (`flat[j::ncols]`); Go hops comma to comma with `bytes.IndexByte` and converts only the requested fields.
- **Typed storage:** compact numeric arrays instead of boxed values; text columns interned (Go, bounded) so repeated categories cost one lookup.
- **Ordered pipeline (Go):** a producer feeds N workers, results are collected in chunk order, in-flight chunks are capped, and the first error in file order wins, so output is identical to the sequential run.

**Trade-offs and limits**

- Python is single-core; the Go version scales with cores when the disk is not the bottleneck.
- Strict RFC 4180 (Go rejects bare quotes) and strict number parsing; clean data upstream if needed.
- The selected columns must fit in RAM; otherwise stream the chunks to a consumer.
- For repeated reads of the same data, a binary cache beats any CSV parser.

---

## 3. Backlog

Candidate scenarios, in no particular order. They become real entries once they get an ID.

| Idea                                | Focus                     | What the tiers would teach                                                |
| ----------------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| 💡 Password hashing                 | Security                  | Plain/fast hash vs salted SHA-256 vs a memory-hard KDF (Argon2id, scrypt) |
| 💡 SQL query building               | Security                  | String concatenation vs manual escaping vs parameterized queries          |
| 💡 Safe archive extraction          | Security                  | Naive extract vs path-prefix check vs full zip-slip and symlink hardening |
| 💡 Token comparison and generation  | Security                  | `==` and `random` vs constant-time compare and a CSPRNG                   |
| 💡 Untrusted deserialization        | Security                  | `pickle`/`gob` on input vs schema-validated JSON with size limits         |
| 💡 Regex on hostile input (ReDoS)   | Security · Performance    | Backtracking patterns vs linear-time matching and input caps              |
| 💡 Top-K frequent items in a stream | Performance               | Full sort vs heap selection vs approximate sketches                       |
| 💡 Duplicate file detection         | Performance               | Hash everything vs group by size, then partial hash, then full hash       |
| 💡 Rate limiter                     | Performance · Correctness | Naive counter vs sliding window vs token bucket, with thread safety       |
| 💡 LRU cache                        | Performance · Correctness | List scan vs ordered dict vs hash map + linked list                       |

---

## 4. Adding or updating an entry

1. Take the next free ID and add a row to the **Catalog** table, including its **Focus** tag.
2. Add a **Scenario card** with the same structure as 001 (summary table, three tiers, measured results, key ideas, trade-offs).
3. Mark each language ✅ only when the file exists and its harness ran cleanly.
4. Record only numbers you actually measured, and say where and how they were measured.
