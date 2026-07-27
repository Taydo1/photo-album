# Global Software Engineering Rules

The following rules apply across software engineering and optimization tasks for this codebase and future iterations:

## 1. Measure First – Baseline Profiling Before Optimization
- Never micro-optimize or restructure code without empirical baseline measurements.
- Establish granular timing breakdowns (warm-up run vs average throughput, sub-operation timings) before modifying implementation logic.

## 2. Single-Level Concurrency Architecture
- Avoid nesting concurrent task abstractions (e.g. spawning `Task.Run` calls inside `Parallel.ForEachAsync` worker loops).
- Maintain a single, flat outer concurrency boundary and execute inner pipeline steps sequentially on the assigned worker thread to prevent thread pool starvation and cache thrashing.

## 3. Transactional Batching for High-Volume I/O
- Do not execute database or disk writes per individual item in high-throughput bulk pipelines.
- Buffer results and execute writes in explicit transactional batches (e.g., 50–100 items per batch) to minimize I/O lock overhead and disk flushing.

## 4. Layered Hardware Acceleration with Software Fallback
- Always wrap hardware-accelerated pipelines (GPU ONNX inference, hardware decoders, DirectML) in a robust try-catch fallback to a CPU/software implementation.
- Ensure application stability across varying target hardware configurations and media types.

## 5. Explicit Startup Warm-Up for Heavy Runtimes
- ML models, GPU graph compilers, and JIT-compiled pipelines introduce significant first-run latency ("cold start").
- Execute explicit dummy warm-up passes during service initialization (`InitializeAsync`) so user-initiated interactive operations remain responsive.
