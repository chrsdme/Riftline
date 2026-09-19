# Documentation

Start with the repository [README](../README.md), then use these Riftline documents:

| Document | Purpose |
|---|---|
| [ARCHITECTURE](ARCHITECTURE.md) | hard-feasibility plane, future performance plane, and runtime placement shape |
| [STATUS](STATUS.md) | public source checkpoint, current qualification, capabilities, and limitations |
| [MEMORY_MODEL](MEMORY_MODEL.md) | per-device admission ledger and heterogeneous KV-growth slopes |
| [PLACEMENT_PLANNER](PLACEMENT_PLANNER.md) | placement specification, candidate enumeration, and automatic-selection boundary |
| [BENCHMARKS](BENCHMARKS.md) | benchmark claims and their scope |
| [RESEARCH_NOTES](RESEARCH_NOTES.md) | pinned research references, comparative runtimes, and design lessons |
| [Acknowledgements](../ACKNOWLEDGEMENTS.md) | upstream contributors and projects that materially informed Riftline |
| [CLI](cli.md) | inherited command-line reference |
| [Serving](serving.md) | inherited serving reference |
| [Performance](performance.md) | inherited performance notes |

The detailed references under `docs/maintainer/` are retained because they document source-level NInfer internals used by the codebase. They do not imply that every upstream or earlier TP2 feature is qualified for Riftline's heterogeneous reference configuration.
