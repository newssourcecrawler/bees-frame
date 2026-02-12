//! nsc_frame
//!
//! Deterministic *frame-based* execution law.
//!
//! - A backend implements [`FrameStepper`] and performs exactly **one** bounded semantic step per call.
//! - The [`Driver`] owns the scheduling loop and calls the backend stepper.
//! - Every step returns a uniform [`StepResult`] envelope (no hidden loops).
//!
//! This crate intentionally contains **no I/O**, **no UI**, and **no model-specific logic**.
//!

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    Prefill,
    Decode,
    Finished,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Advanced,
    Yielded,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    Eos,
    Cancelled,
    BackendError,
}

#[derive(Debug, Clone)]
pub struct Receipt {
    pub kind: &'static str,
    pub value_u64: u64,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub outcome: StepOutcome,
    pub emitted_token: Option<u32>,
    pub stop_reason: Option<StopReason>,
    pub receipts: Vec<Receipt>,
}

impl StepResult {
    pub fn advanced(token: Option<u32>) -> Self {
        Self {
            outcome: StepOutcome::Advanced,
            emitted_token: token,
            stop_reason: None,
            receipts: Vec::new(),
        }
    }
    pub fn finished(reason: StopReason) -> Self {
        Self {
            outcome: StepOutcome::Finished,
            emitted_token: None,
            stop_reason: Some(reason),
            receipts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameCursor {
    pub position: u32,
}

impl Default for FrameCursor {
    fn default() -> Self {
        Self { position: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct FrameLimits {
    pub max_tokens: usize,
}

#[derive(Debug)]
pub struct Frame<M> {
    pub state: FrameState,
    pub cursor: FrameCursor,
    pub limits: FrameLimits,
    pub mem: M,

    // posterity-safe prompt ownership
    pub prompt_token_ids: Vec<u32>,
    pub prompt_index: usize,

    /// Output log (token ids). Keep in the law so tools can inspect generically.
    pub generated_token_ids: Vec<u32>,
    pub tokens_generated: usize,
}

impl<M> Frame<M> {
    pub fn new(mem: M, max_tokens: usize) -> Self {
        Self {
            state: FrameState::Prefill,
            cursor: FrameCursor::default(),
            limits: FrameLimits { max_tokens },
            mem,
            prompt_token_ids: Vec::new(),
            prompt_index: 0,
            generated_token_ids: Vec::new(),
            tokens_generated: 0,
        }
    }

    pub fn cancel(&mut self) {
        self.state = FrameState::Cancelled;
    }
    
    pub fn with_prompt(mem: M, max_tokens: usize, prompt_token_ids: Vec<u32>) -> Self {
        Self {
            state: FrameState::Prefill,
            cursor: FrameCursor::default(),
            limits: FrameLimits { max_tokens },
            mem,
            prompt_token_ids,
            prompt_index: 0,
            generated_token_ids: Vec::new(),
            tokens_generated: 0,
        }
    }
}

/// Policy oracle. Must never execute. Called once per driver step.
pub trait Arbiter<M> {
    fn decide(&mut self, frame: &Frame<M>) -> Decision;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Yield,
    Refuse,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoArbiter;

impl<M> Arbiter<M> for NoArbiter {
    fn decide(&mut self, _frame: &Frame<M>) -> Decision {
        Decision::Allow
    }
}

/// Backend stepper: does exactly one bounded semantic step.
pub trait FrameStepper<M> {
    fn step(&mut self, frame: &mut Frame<M>) -> Result<StepResult, String>;
}

/// Driver owns the loop (scheduling). Backend owns one-step execution.
pub struct Driver<M, S, A = NoArbiter>
where
    S: FrameStepper<M>,
    A: Arbiter<M>,
{
    pub frame: Frame<M>,
    pub stepper: S,
    pub arbiter: A,
}

impl<M, S> Driver<M, S, NoArbiter>
where
    S: FrameStepper<M>,
{
    pub fn new(frame: Frame<M>, stepper: S) -> Self {
        Self {
            frame,
            stepper,
            arbiter: NoArbiter,
        }
    }
}

impl<M, S, A> Driver<M, S, A>
where
    S: FrameStepper<M>,
    A: Arbiter<M>,
{
    pub fn with_arbiter(frame: Frame<M>, stepper: S, arbiter: A) -> Self {
        Self { frame, stepper, arbiter }
    }

    pub fn step(&mut self) -> Result<StepResult, String> {
        match self.frame.state {
            FrameState::Finished => return Ok(StepResult::finished(StopReason::MaxTokens)),
            FrameState::Cancelled => return Ok(StepResult::finished(StopReason::Cancelled)),
            _ => {}
        }

        match self.arbiter.decide(&self.frame) {
            Decision::Allow => self.stepper.step(&mut self.frame),
            Decision::Yield => Ok(StepResult {
                outcome: StepOutcome::Yielded,
                emitted_token: None,
                stop_reason: None,
                receipts: vec![Receipt { kind: "arbiter.yield", value_u64: 1 }],
            }),
            Decision::Refuse => {
                self.frame.cancel();
                Ok(StepResult::finished(StopReason::Cancelled))
            }
        }
    }

    pub fn run_to_completion(&mut self) -> Result<(), String> {
        loop {
            let r = self.step()?;
            match r.outcome {
                StepOutcome::Finished => return Ok(()),
                _ => {}
            }
        }
    }
}

/// A tiny noop backend (public-friendly): proves the law compiles and runs.
#[derive(Debug, Default)]
pub struct NoopStepper;

#[derive(Debug, Default)]
pub struct NoopMem;

impl FrameStepper<NoopMem> for NoopStepper {
    fn step(&mut self, frame: &mut Frame<NoopMem>) -> Result<StepResult, String> {
        match frame.state {
            FrameState::Prefill => {
                frame.state = FrameState::Decode;
                Ok(StepResult::advanced(None))
            }
            FrameState::Decode => {
                if frame.tokens_generated >= frame.limits.max_tokens {
                    frame.state = FrameState::Finished;
                    return Ok(StepResult::finished(StopReason::MaxTokens));
                }
                // “Generate” a deterministic token id (toy).
                let tok = (frame.cursor.position % 256) as u32;
                frame.generated_token_ids.push(tok);
                frame.tokens_generated += 1;
                frame.cursor.position = frame.cursor.position.saturating_add(1);
                Ok(StepResult::advanced(Some(tok)))
            }
            FrameState::Finished => Ok(StepResult::finished(StopReason::MaxTokens)),
            FrameState::Cancelled => Ok(StepResult::finished(StopReason::Cancelled)),
        }
    }
}