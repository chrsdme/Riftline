# Contributing

Riftline is experimental systems software. Contributions should be narrow, reproducible, and explicit about hardware and model artifacts.

Before submitting a change:

- run focused checks for the affected code path;
- report GPU model, driver, CUDA version, model artifact, context length, placement, and command line for runtime or performance issues;
- keep benchmark claims tied to reproducible provenance;
- do not submit generated model weights, `.ninfer` artifacts, caches, raw logs, credentials, or private machine paths;
- state any relevant check that was not run.

The source uses C++20, CUDA, CMake, and the inherited NInfer style. Build and test commands in this repository should be treated as target-specific until validated on the hardware being reported.
