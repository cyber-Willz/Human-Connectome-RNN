use hcrnn::network::default_roles;
use hcrnn::rng::Rng;
use hcrnn::train::{evaluate, make_trials, train};
use hcrnn::{Connectome, ConnectomeRnn, MaskMode, MemoryGuidedTask, RnnConfig, TrainConfig};

fn small_conn() -> Connectome {
    Connectome::synthetic_human(8, 0.3, 3).unwrap().log_compressed()
}

fn small_cfg(conn: &Connectome, mode: MaskMode) -> RnnConfig {
    let (i, o) = default_roles(conn).unwrap();
    RnnConfig { mode, input_regions: i, output_regions: o, seed: 5, ..Default::default() }
}

#[test]
fn synthetic_connectome_is_valid() {
    let c = Connectome::synthetic_human(40, 0.25, 1).unwrap();
    let n = c.n();
    assert_eq!(n, 80);
    for i in 0..n {
        assert_eq!(c.weight(i, i), 0.0);
        for j in 0..n {
            assert_eq!(c.weight(i, j), c.weight(j, i));
            assert!(c.weight(i, j) >= 0.0);
        }
    }
    assert!(c.is_connected());
    assert!((c.density() - 0.25).abs() < 0.04, "density {}", c.density());
    // left-right homotopic links dominate interhemispheric strength per edge
    assert!(c.interhemispheric_fraction() > 0.05 && c.interhemispheric_fraction() < 0.5);
    // every edge has a positive fibre length
    for i in 0..n {
        for j in 0..n {
            if i != j {
                assert!(c.length(i, j) > 0.0);
            }
        }
    }
}

#[test]
fn degree_preserving_rewire_keeps_degrees() {
    let c = Connectome::synthetic_human(20, 0.25, 2).unwrap();
    let mut rng = Rng::new(9);
    let r = c.degree_preserving_rewire(&mut rng, 20);
    assert_eq!(c.num_edges(), r.num_edges());
    let mut changed = 0;
    for i in 0..c.n() {
        assert_eq!(c.degree(i), r.degree(i), "degree of region {i}");
        for j in 0..c.n() {
            if (c.weight(i, j) > 0.0) != (r.weight(i, j) > 0.0) {
                changed += 1;
            }
        }
    }
    assert!(changed > c.num_edges(), "graph was barely rewired ({changed})");
    let rr = c.density_matched_random(&mut rng);
    assert_eq!(c.num_edges(), rr.num_edges());
}

#[test]
fn constraints_hold_in_every_mode_and_after_training() {
    let c = small_conn();
    for mode in MaskMode::ALL {
        let mut net = ConnectomeRnn::new(&c, small_cfg(&c, mode)).unwrap();
        net.check_constraints().unwrap();
        let task = MemoryGuidedTask::default();
        let cfg = TrainConfig { steps: 5, batch: 4, log_every: 5, eval_trials: 4, threads: 2, ..Default::default() };
        train(&mut net, &task, &cfg, |_| {});
        net.check_constraints().unwrap();
        // spectral radius at init matches the requested gain
        let fresh = ConnectomeRnn::new(&c, small_cfg(&c, mode)).unwrap();
        assert!((fresh.spectral_radius() - 0.9).abs() < 0.05, "{mode:?}: {}", fresh.spectral_radius());
    }
}

#[test]
fn connectome_mask_is_respected_exactly() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let (r, n) = (net.n_regions, net.n_units);
    let m = net.dense_weights();
    for post in 0..n {
        for pre in 0..n {
            let (rp, rq) = (post % r, pre % r);
            if rp != rq {
                let v = m[post * n + pre];
                if c.weight(rp, rq) == 0.0 {
                    assert_eq!(v, 0.0, "weight where the connectome has no tract");
                } else if pre < r {
                    assert!(v > 0.0, "tract present but no excitatory projection");
                } else {
                    assert_eq!(v, 0.0, "inhibitory long-range projection");
                }
            }
        }
    }
}

#[test]
fn delays_follow_fibre_length() {
    let c = small_conn();
    let cfg = small_cfg(&c, MaskMode::Connectome);
    let net = ConnectomeRnn::new(&c, cfg.clone()).unwrap();
    for e in &net.edges {
        let (rp, rq) = (net.region_of(e.post), net.region_of(e.pre));
        if rp == rq {
            assert_eq!(e.delay, 0);
        } else {
            let expect = ((c.length(rp, rq) / (cfg.velocity_mm_per_ms * cfg.dt_ms)).round() as usize).max(1);
            assert_eq!(e.delay, expect);
        }
    }
    let nd = ConnectomeRnn::new(&c, RnnConfig { use_delays: false, ..cfg }).unwrap();
    assert!(nd.edges.iter().all(|e| e.delay == 0));
    assert!(net.max_delay() >= 2);
}

#[test]
fn analytic_gradient_matches_finite_differences() {
    let c = small_conn();
    let mut net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    // perturb so biases/readout/input weights are non-trivial
    let mut rng = Rng::new(11);
    for p in net.params.iter_mut() {
        *p += 0.05 * rng.normal();
    }
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 6, delay_max: 6, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trial = task.sample(&mut rng);
    let w = net.effective_weights();
    let mut grad = vec![0.0; net.params.len()];
    let loss0 = net.loss_and_grad_with(&w, &trial, &mut grad);
    assert!(loss0.is_finite() && loss0 > 0.0);

    let loss_at = |net: &ConnectomeRnn| {
        let ro = net.forward(&trial);
        net.trial_loss(&ro, &trial)
    };
    let mut checked = 0;
    let mut worst: f64 = 0.0;
    // check a spread of parameters from every block, always including the largest gradients
    let mut idx: Vec<usize> = (0..net.params.len()).step_by(7).collect();
    let mut by_mag: Vec<usize> = (0..net.params.len()).collect();
    by_mag.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
    idx.extend(by_mag.into_iter().take(25));
    for &k in &idx {
        let eps = 1e-6;
        let orig = net.params[k];
        net.params[k] = orig + eps;
        let lp = loss_at(&net);
        net.params[k] = orig - eps;
        let lm = loss_at(&net);
        net.params[k] = orig;
        let fd = (lp - lm) / (2.0 * eps);
        let denom = fd.abs().max(grad[k].abs()).max(1e-7);
        let rel = (fd - grad[k]).abs() / denom;
        if fd.abs().max(grad[k].abs()) > 1e-9 {
            worst = worst.max(rel);
            checked += 1;
            assert!(rel < 1e-4, "param {k}: analytic {} vs numeric {fd} (rel {rel})", grad[k]);
        }
    }
    assert!(checked > 30, "only {checked} parameters had non-negligible gradients");
    eprintln!("gradient check: {checked} params, worst relative error {worst:.2e}");
}

#[test]
fn batch_gradient_is_thread_count_invariant() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let task = MemoryGuidedTask { fix_steps: 3, cue_steps: 4, delay_min: 5, delay_max: 8, resp_steps: 6, grace: 1, cue_amp: 1.0 };
    let trials = make_trials(&task, 7, 3);
    let (mut g1, mut g4) = (vec![0.0; net.params.len()], vec![0.0; net.params.len()]);
    let l1 = net.batch_loss_and_grad(&trials, &mut g1, 1);
    let l4 = net.batch_loss_and_grad(&trials, &mut g4, 4);
    assert!((l1 - l4).abs() < 1e-12);
    for (a, b) in g1.iter().zip(&g4) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn params_roundtrip() {
    let c = small_conn();
    let net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let path = std::env::temp_dir().join("hcrnn_roundtrip_test.bin");
    net.save_params(&path).unwrap();
    let mut other = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    other.params.iter_mut().for_each(|p| *p = 0.0);
    other.load_params(&path).unwrap();
    assert_eq!(net.params, other.params);
    let _ = std::fs::remove_file(path);
}

#[test]
fn training_reduces_loss() {
    let c = Connectome::synthetic_human(24, 0.25, 4).unwrap().log_compressed();
    let mut net = ConnectomeRnn::new(&c, small_cfg(&c, MaskMode::Connectome)).unwrap();
    let task = MemoryGuidedTask { fix_steps: 4, cue_steps: 6, delay_min: 8, delay_max: 12, resp_steps: 10, grace: 2, cue_amp: 1.0 };
    let before = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    let cfg = TrainConfig { steps: 300, batch: 16, lr: 5e-3, log_every: 150, eval_trials: 16, threads: 4, ..Default::default() };
    train(&mut net, &task, &cfg, |_| {});
    let after = evaluate(&net, &make_trials(&task, 32, 99)).loss;
    assert!(after < before * 0.6, "loss did not drop enough: {before} -> {after}");
}

#[test]
fn tvb_directory_loads_when_available() {
    let Ok(dir) = std::env::var("HCRNN_TVB66") else { return };
    let c = Connectome::from_tvb_dir(std::path::Path::new(&dir)).unwrap();
    assert_eq!(c.n(), 66);
    assert!(c.is_connected());
    let (i, o) = default_roles(&c).unwrap();
    assert!(!i.is_empty() && !o.is_empty());
    let net = ConnectomeRnn::new(&c.log_compressed(), small_cfg(&c, MaskMode::Connectome)).unwrap();
    net.check_constraints().unwrap();
}
