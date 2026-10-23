//! Training loop, evaluation and a simple per-system analysis of delay activity.

use crate::network::ConnectomeRnn;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::task::{MemoryGuidedTask, Trial};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub clip: f64,
    pub seed: u64,
    pub log_every: usize,
    pub eval_trials: usize,
    pub threads: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            steps: 400,
            batch: 24,
            lr: 3e-3,
            clip: 1.0,
            seed: 1,
            log_every: 50,
            eval_trials: 96,
            threads: crate::network::default_threads(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalMetrics {
    pub loss: f64,
    /// Mean absolute angular error (degrees) of the decoded response; chance is 90.
    pub angular_error_deg: f64,
    /// Fraction of trials decoded within 30 degrees of the cue.
    pub acc_30deg: f64,
}

#[derive(Clone, Debug)]
pub struct LogLine {
    pub step: usize,
    pub train_loss: f64,
    pub grad_norm: f64,
    pub eval: EvalMetrics,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub log: Vec<LogLine>,
    pub final_eval: EvalMetrics,
    pub seconds: f64,
}

pub fn make_trials(task: &MemoryGuidedTask, n: usize, seed: u64) -> Vec<Trial> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| task.sample(&mut rng)).collect()
}

fn wrap_angle(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut d = (a + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
    if d < -std::f64::consts::PI {
        d += two_pi;
    }
    d
}

pub fn evaluate(net: &ConnectomeRnn, trials: &[Trial]) -> EvalMetrics {
    let w = net.effective_weights();
    let (mut loss, mut err, mut hit) = (0.0, 0.0, 0usize);
    for tr in trials {
        let ro = net.forward_with(&w, tr);
        loss += net.trial_loss(&ro, tr);
        // Decode from the last steps of the response window.
        let win = 8.min(tr.t - tr.go);
        let (mut cx, mut cy) = (0.0, 0.0);
        for t in (tr.t - win)..tr.t {
            cx += ro.y[t * 2];
            cy += ro.y[t * 2 + 1];
        }
        let pred = cy.atan2(cx);
        let e = wrap_angle(pred - tr.angle).abs().to_degrees();
        err += e;
        if e < 30.0 {
            hit += 1;
        }
    }
    let n = trials.len().max(1) as f64;
    EvalMetrics { loss: loss / n, angular_error_deg: err / n, acc_30deg: hit as f64 / n }
}

pub fn train(
    net: &mut ConnectomeRnn,
    task: &MemoryGuidedTask,
    cfg: &TrainConfig,
    mut on_log: impl FnMut(&LogLine),
) -> TrainReport {
    let mut rng = Rng::new(cfg.seed);
    let eval_set = make_trials(task, cfg.eval_trials, cfg.seed ^ 0xE7A1_5EED);
    let mut opt = Adam::new(net.params.len(), cfg.lr, cfg.clip);
    let mut grad = vec![0.0; net.params.len()];
    let start = Instant::now();
    let mut log = Vec::new();
    let (mut acc_loss, mut acc_n, mut last_gn) = (0.0, 0usize, 0.0);
    for step in 1..=cfg.steps {
        let batch: Vec<Trial> = (0..cfg.batch).map(|_| task.sample(&mut rng)).collect();
        let loss = net.batch_loss_and_grad(&batch, &mut grad, cfg.threads);
        last_gn = opt.step(&mut net.params, &grad);
        acc_loss += loss;
        acc_n += 1;
        if step % cfg.log_every == 0 || step == cfg.steps {
            let line = LogLine {
                step,
                train_loss: acc_loss / acc_n as f64,
                grad_norm: last_gn,
                eval: evaluate(net, &eval_set),
                seconds: start.elapsed().as_secs_f64(),
            };
            on_log(&line);
            log.push(line);
            acc_loss = 0.0;
            acc_n = 0;
        }
    }
    let _ = last_gn;
    let final_eval = evaluate(net, &eval_set);
    TrainReport { log, final_eval, seconds: start.elapsed().as_secs_f64() }
}

/// For each system: mean (over its regions) across-trial standard deviation of the
/// excitatory firing rate at the end of the delay. Because the stimulus is identical
/// apart from the cue angle, this is a proxy for how much cue information a region
/// is still carrying when the go signal arrives.
pub fn delay_cue_variability_by_system(net: &ConnectomeRnn, trials: &[Trial]) -> Vec<(String, usize, f64)> {
    let w = net.effective_weights();
    let (r, n) = (net.n_regions, net.n_units);
    let mut sum = vec![0.0; r];
    let mut sumsq = vec![0.0; r];
    for tr in trials {
        let ro = net.forward_with(&w, tr);
        for k in 0..r {
            let v = ro.r[tr.go * n + k];
            sum[k] += v;
            sumsq[k] += v * v;
        }
    }
    let m = trials.len().max(1) as f64;
    let sd: Vec<f64> = (0..r).map(|k| ((sumsq[k] / m) - (sum[k] / m).powi(2)).max(0.0).sqrt()).collect();
    let mut systems: Vec<String> = Vec::new();
    for s in &net.region_systems {
        if !systems.contains(s) {
            systems.push(s.clone());
        }
    }
    let mut out: Vec<(String, usize, f64)> = systems
        .into_iter()
        .map(|s| {
            let idx: Vec<usize> = (0..r).filter(|&k| net.region_systems[k] == s).collect();
            let mean = idx.iter().map(|&k| sd[k]).sum::<f64>() / idx.len() as f64;
            (s, idx.len(), mean)
        })
        .collect();
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    out
}
