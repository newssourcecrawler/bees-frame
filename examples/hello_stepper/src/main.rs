use nsc_frame::{
    Arbiter, Decision, Driver, Frame, FrameState, NoopMem, NoopStepper,
    StepOutcome,
};

/// Simple arbiter that forces a Yield every 3 steps.
#[derive(Debug, Default)]
struct TickArbiter {
    ticks: u32,
}

impl Arbiter<NoopMem> for TickArbiter {
    fn decide(&mut self, _frame: &Frame<NoopMem>) -> Decision {
        self.ticks += 1;
        if self.ticks % 3 == 0 {
            Decision::Yield
        } else {
            Decision::Allow
        }
    }
}

fn main() {
    // Frame with max 8 generated tokens.
    let mem = NoopMem::default();
    let frame = Frame::new(mem, 8);
    let stepper = NoopStepper::default();
    let arbiter = TickArbiter::default();

    let mut driver = Driver::with_arbiter(frame, stepper, arbiter);

    println!("== bees frame demo ==");

    loop {
        let r = driver.step().expect("step failed");

        match r.outcome {
            StepOutcome::Advanced => {
                if let Some(tok) = r.emitted_token {
                    println!("advanced: token={} state={:?}", tok, driver.frame.state);
                } else {
                    println!("advanced: state={:?}", driver.frame.state);
                }
            }
            StepOutcome::Yielded => {
                println!("yielded by arbiter at cursor={}", driver.frame.cursor.position);
            }
            StepOutcome::Finished => {
                println!("finished: state={:?}", driver.frame.state);
                break;
            }
        }
    }

    println!("generated_token_ids = {:?}", driver.frame.generated_token_ids);
}