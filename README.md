# bees-frame (nsc_frame)

Deterministic **frame-based** execution law: a backend performs exactly **one** bounded semantic step per call.

## Core ideas
- **One step per call:** no hidden loops.
- **Driver owns the loop:** scheduling is explicit.
- **Backend owns compute:** implement `FrameStepper`.
- **Policy hook:** an `Arbiter` can `Allow`, `Yield`, or `Refuse`.
- **Uniform result envelope:** each step returns a `StepResult`.

This repo intentionally contains **no I/O**, **no UI**, and **no model-specific logic**.

## What’s included
- `nsc_frame` library crate
- a small example demonstrating stepping + arbiter-yield

## Non-goals
- Linux compatibility layer
- LLM backends
- async frameworks or hidden schedulers
