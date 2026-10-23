//! # human_connectome_rnn
//!
//! Recurrent neural networks constrained by a human structural connectome:
//! wiring from tractography, conduction delays from fibre length, Dale's-law
//! excitatory/inhibitory populations per region, trained with exact BPTT.
//!
//! ```no_run
//! use hcrnn::{Connectome, ConnectomeRnn, RnnConfig, MemoryGuidedTask, TrainConfig};
//! let conn = Connectome::synthetic_human(40, 0.25, 1).unwrap().log_compressed();
//! let (inp, out) = hcrnn::network::default_roles(&conn).unwrap();
//! let cfg = RnnConfig { input_regions: inp, output_regions: out, ..Default::default() };
//! let mut net = ConnectomeRnn::new(&conn, cfg).unwrap();
//! let report = hcrnn::train::train(&mut net, &MemoryGuidedTask::default(), &TrainConfig::default(), |_| {});
//! println!("{:?}", report.final_eval);
//! ```

pub mod connectome;
pub mod network;
pub mod optim;
pub mod rng;
pub mod task;
pub mod train;

pub use connectome::{Connectome, Region};
pub use network::{ConnectomeRnn, MaskMode, RnnConfig};
pub use task::{MemoryGuidedTask, Trial};
pub use train::{EvalMetrics, TrainConfig};
