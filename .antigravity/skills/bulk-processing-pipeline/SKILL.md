---
name: bulk-processing-pipeline
description: Architecture blueprint for building high-throughput, parallel data and media processing pipelines with hardware acceleration, cache locality, and transactional batching.
---

# Bulk Processing Pipeline Skill

Use this skill when designing or refactoring high-throughput processing pipelines for media, files, or large datasets.

## Pipeline Architecture Steps

### 1. Initialization & Warm-Up Phase
- Initialize hardware providers (e.g. GPU inference sessions, DirectML, native decoders).
- Perform a single dummy execution pass during service initialization (`InitializeAsync`) to warm up JIT compilation, GPU command queues, and graph optimizations before user interaction.

### 2. Flat Bounded Concurrency
- Use a single, bounded outer parallel loop (e.g., `Parallel.ForEachAsync` with controlled `MaxDegreeOfParallelism`).
- Do **not** spawn nested asynchronous `Task.Run` calls inside the loop per subtask.

### 3. Sequential Local Worker Execution
- On each parallel worker thread, execute metadata decoding, visual processing, and inference sequentially.
- Keep intermediate byte buffers and matrices allocated on the local worker thread to preserve CPU L2/L3 cache locality and eliminate thread pool context switches.

### 4. Layered Fallbacks
- Wrap hardware-accelerated and native calls in fallback try-catch blocks to software/CPU implementations (e.g., CPU decoding or software face detection).

### 5. Transactional Batch Persistence
- Accumulate processed items in thread-safe buffers.
- Write results to storage or SQLite databases in explicit transactional batches (e.g., 50–100 items per transaction) to eliminate per-item disk flush overhead.
