<h1 align="center">PitfallLab</h1>

<p align="center">
  <b>Learn engineering by stepping into the pitfall first.</b><br>
  Real scenarios solved three ways (Wrong, Common, Optimal) in Python, Go and Rust,
  and proven with measured runs.
</p>

---

## What is PitfallLab?

Every scenario is a real engineering problem. It is solved on **three tiers**, inside one
runnable file per language:

| Tier | Name        | Purpose                                                                                                                           |
| :--: | ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
|  1   | **Wrong**   | The plausible mistake people actually make, with a named and demonstrable flaw.                                                   |
|  2   | **Common**  | What most developers would write: correct and readable, but not tuned or hardened.                                                |
|  3   | **Optimal** | The production-grade solution: key insight first, validated input, bounded resources, documented complexity, explicit trade-offs. |

Every file ends with a **harness** that proves the claims: correctness checks against a trusted
reference, edge cases and error handling, timings over growing inputs, memory figures, a
comparison table and a written verdict. Numbers in this repository come from real runs;
absolute times depend on the machine, so trust the ratios and the scaling trend.

## Focus areas

Scenarios are tagged by what the "optimal" tier improves:

| Focus           | Typical pitfalls it covers                                                       |
| --------------- | -------------------------------------------------------------------------------- |
| **Performance** | Quadratic blow-ups, needless copies, per-row overhead, lock contention           |
| **Security**    | Injection, path traversal, unsafe deserialization, timing leaks, unbounded input |
| **Correctness** | Edge cases, race conditions, non-determinism, silent data corruption             |

## Where to find things

| Path                                                 | What it contains                                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`Scenarios.md`](Scenarios.md)                       | The catalog: exact scenario names, focus, the three tiers, complexity, measured results, status per language |
| [`python/`](python/), [`go/`](go/), [`rust/`](rust/) | The code, one sub-directory per scenario                                                                     |

## Repository layout

```
pitfalllab/
├── README.md
├── Scenarios.md                    <- scenario catalog
├── .gitignore
├── python/
│   └── 001_fast_csv_reader/
│       └── fast_csv_reader.py
├── go/
│   ├── go.mod                      <- module pitfalllab (one for all Go scenarios)
│   └── 001_fast_csv_reader/
│       └── fast_csv_reader.go
└── rust/
```

Conventions:

- One directory per language, one sub-directory per scenario: `NNN_snake_case_name/`.
- **The number `NNN` is the scenario ID and is shared across languages.**
  `python/001_...` and `go/001_...` are the same problem in two languages.
- Everything inside code files (identifiers, comments, printed output) is English.
- Extra material (sample data, notes) lives next to the code, inside the scenario folder.

## Scenario index

| ID  | Scenario                      | Focus                     | Python                                                              | Go                                                              | Rust    |
| :-: | ----------------------------- | ------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------- | ------- |
| 001 | Fast CSV Reader (stdlib only) | Performance · Correctness | [fast_csv_reader.py](python/001_fast_csv_reader/fast_csv_reader.py) | [fast_csv_reader.go](go/001_fast_csv_reader/fast_csv_reader.go) | planned |

Details for each scenario are in [`Scenarios.md`](Scenarios.md).

## Quick start

| Language     | Run                                                    | Notes                                                                                                |
| ------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Python 3.10+ | `python python/NNN_name/file.py [size]`                | Standard library only unless the file says otherwise                                                 |
| Go 1.21+     | `cd go && go run ./NNN_name [size]`                    | One `go.mod` in `go/`; each scenario is its own `package main`. Add `-race` for concurrent scenarios |
| Rust         | `rustc -O rust/NNN_name/main.rs -o /tmp/pl && /tmp/pl` | Move to `cargo` inside the scenario folder if it needs dependencies                                  |

Examples:

```bash
python python/001_fast_csv_reader/fast_csv_reader.py 1000000
cd go && go run ./001_fast_csv_reader 1000000
```

## Adding a scenario

1. Take the next free ID (`002`, `003`, ...) and a short `snake_case` name.
2. Create `<language>/NNN_name/` and place the single code file inside.
3. Add a row to the index above and a full card to [`Scenarios.md`](Scenarios.md).
4. To port a scenario to another language, reuse the same `NNN_name`.
5. Mark a language as available only when the file exists and its harness ran cleanly.
