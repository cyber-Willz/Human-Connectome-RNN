use hcrnn::network::{default_roles, default_threads};
use hcrnn::train::{delay_cue_variability_by_system, evaluate, make_trials, train};
use hcrnn::{Connectome, ConnectomeRnn, MaskMode, MemoryGuidedTask, RnnConfig, TrainConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const HELP: &str = "\
connectome_rnn - recurrent neural networks constrained by a human structural connectome

USAGE:
  connectome_rnn <inspect|train|compare> [options]

DATA (pick one; default is --synthetic 40):
  --tvb DIR              TVB-style folder (weights.txt, tract_lengths.txt, centres.txt),
                         e.g. the human 66-region connectome (see scripts/fetch_tvb66.sh)
  --csv FILE             N x N weight matrix; with --lengths FILE (mm) and --labels FILE
                         (rows: name[,hemi[,system[,x,y,z]]])
  --synthetic N          synthetic human-like connectome, N regions per hemisphere
  --density D            target density for --synthetic (default 0.25)
  --data-seed S          seed for --synthetic (default 1)
  --linear-weights       use raw weights (default: log-compress heavy-tailed weights)

MODEL:
  --mode M               connectome | degree-preserving | density-matched | dense  (train)
  --no-delays            ignore conduction delays
  --velocity V           conduction velocity in mm/ms = m/s (default 5)
  --dt MS --tau MS       Euler step and membrane time constant (default 5, 20)
  --gain G               initial spectral radius (default 0.9)
  --inter-gain G         inter-regional vs local strength (default 1.0)
  --input-regions LIST   comma-separated indices or names (default: visual regions)
  --output-regions LIST  comma-separated indices or names (default: somatomotor regions)

TRAINING:
  --steps N (400)  --batch N (24)  --lr X (0.003)  --clip X (1.0)  --seed N (1)
  --log-every N (50)  --threads N  --out DIR (save params + weights)
";

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: HashSet<String>,
}

const FLAGS: &[&str] = &["no-delays", "linear-weights", "help"];

impl Args {
    fn parse() -> Result<Args, String> {
        let mut cmd = String::new();
        let mut kv = HashMap::new();
        let mut flags = HashSet::new();
        let mut it = std::env::args().skip(1).peekable();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                if FLAGS.contains(&name) {
                    flags.insert(name.to_string());
                } else {
                    let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                    kv.insert(name.to_string(), v);
                }
            } else if cmd.is_empty() {
                cmd = a;
            } else {
                return Err(format!("unexpected argument '{a}'"));
            }
        }
        Ok(Args { cmd, kv, flags })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.contains(k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.get(k) {
            Some(v) => v.parse::<T>().map_err(|_| format!("--{k}: cannot parse '{v}'")),
            None => Ok(default),
        }
    }
}

fn load_connectome(a: &Args) -> Result<Connectome, String> {
    let c = if let Some(d) = a.get("tvb") {
        Connectome::from_tvb_dir(Path::new(d))?
    } else if let Some(w) = a.get("csv") {
        Connectome::from_csv(Path::new(w), a.get("lengths").map(Path::new), a.get("labels").map(Path::new))?
    } else {
        Connectome::synthetic_human(a.num("synthetic", 40usize)?, a.num("density", 0.25f64)?, a.num("data-seed", 1u64)?)?
    };
    Ok(if a.flag("linear-weights") { c } else { c.log_compressed() })
}

fn region_list(s: &str, conn: &Connectome) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for tok in s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()) {
        if let Ok(i) = tok.parse::<usize>() {
            if i >= conn.n() {
                return Err(format!("region index {i} out of range"));
            }
            out.push(i);
        } else if let Some(i) = conn.regions.iter().position(|r| r.name == tok) {
            out.push(i);
        } else {
            return Err(format!("unknown region '{tok}'"));
        }
    }
    Ok(out)
}

fn build_config(a: &Args, conn: &Connectome, mode: MaskMode) -> Result<RnnConfig, String> {
    let (di, do_) = match (a.get("input-regions"), a.get("output-regions")) {
        (Some(_), Some(_)) => (Vec::new(), Vec::new()),
        _ => default_roles(conn)?,
    };
    let input_regions = match a.get("input-regions") {
        Some(s) => region_list(s, conn)?,
        None => di,
    };
    let output_regions = match a.get("output-regions") {
        Some(s) => region_list(s, conn)?,
        None => do_,
    };
    let d = RnnConfig::default();
    Ok(RnnConfig {
        mode,
        dt_ms: a.num("dt", d.dt_ms)?,
        tau_ms: a.num("tau", d.tau_ms)?,
        velocity_mm_per_ms: a.num("velocity", d.velocity_mm_per_ms)?,
        use_delays: !a.flag("no-delays"),
        gain: a.num("gain", d.gain)?,
        inter_gain: a.num("inter-gain", d.inter_gain)?,
        n_in: MemoryGuidedTask::N_IN,
        n_out: MemoryGuidedTask::N_OUT,
        input_regions,
        output_regions,
        seed: a.num("seed", 1u64)?,
        ..d
    })
}

fn train_config(a: &Args) -> Result<TrainConfig, String> {
    let d = TrainConfig::default();
    Ok(TrainConfig {
        steps: a.num("steps", d.steps)?,
        batch: a.num("batch", d.batch)?,
        lr: a.num("lr", d.lr)?,
        clip: a.num("clip", d.clip)?,
        seed: a.num("seed", d.seed)?,
        log_every: a.num("log-every", d.log_every)?,
        eval_trials: d.eval_trials,
        threads: a.num("threads", default_threads())?,
    })
}

fn names(conn: &Connectome, idx: &[usize]) -> String {
    idx.iter().map(|&i| conn.regions[i].name.as_str()).collect::<Vec<_>>().join(", ")
}

fn describe_net(net: &ConnectomeRnn) -> String {
    let inter = net.edges.iter().filter(|e| e.delay > 0 || net.region_of(e.post) != net.region_of(e.pre)).count();
    format!(
        "units: {} ({} E + {} I)   synapses: {} ({} inter-regional)   trainable params: {}   max delay: {} steps ({:.0} ms)   spectral radius: {:.3}",
        net.n_units,
        net.n_regions,
        net.n_regions,
        net.edges.len(),
        inter,
        net.num_params(),
        net.max_delay(),
        net.max_delay() as f64 * net.cfg.dt_ms,
        net.spectral_radius()
    )
}

fn run() -> Result<(), String> {
    let a = Args::parse()?;
    if a.flag("help") || a.cmd.is_empty() || a.cmd == "help" {
        print!("{HELP}");
        return Ok(());
    }
    let conn = load_connectome(&a)?;
    match a.cmd.as_str() {
        "inspect" => {
            println!("{}", conn.summary());
            let (inp, out) = default_roles(&conn).unwrap_or_default();
            if !inp.is_empty() {
                println!("default input regions : {}", names(&conn, &inp));
                println!("default output regions: {}", names(&conn, &out));
            }
            Ok(())
        }
        "train" => {
            let mode = match a.get("mode") {
                Some(m) => MaskMode::parse(m).ok_or_else(|| format!("unknown --mode '{m}'"))?,
                None => MaskMode::Connectome,
            };
            let cfg = build_config(&a, &conn, mode)?;
            let tcfg = train_config(&a)?;
            println!("{}", conn.summary());
            println!("input regions : {}", names(&conn, &cfg.input_regions));
            println!("output regions: {}", names(&conn, &cfg.output_regions));
            let mut net = ConnectomeRnn::new(&conn, cfg)?;
            net.check_constraints()?;
            println!("mode: {}\n{}", mode.name(), describe_net(&net));
            let task = MemoryGuidedTask::default();
            println!("\n step   train-loss   eval-loss   ang.err(deg)   acc<30deg   |grad|   sec");
            let rep = train(&mut net, &task, &tcfg, |l| {
                println!(
                    "{:5}   {:10.4}   {:9.4}   {:12.1}   {:9.2}   {:6.3}   {:5.1}",
                    l.step, l.train_loss, l.eval.loss, l.eval.angular_error_deg, l.eval.acc_30deg, l.grad_norm, l.seconds
                );
            });
            net.check_constraints()?;
            println!(
                "\nfinal: loss {:.4}  angular error {:.1} deg (chance 90)  within 30 deg: {:.0}%  [{:.1}s]  constraints verified",
                rep.final_eval.loss, rep.final_eval.angular_error_deg, rep.final_eval.acc_30deg * 100.0, rep.seconds
            );
            let test = make_trials(&task, 256, 0xC0FFEE);
            let m = evaluate(&net, &test);
            println!("held-out (256 new trials): loss {:.4}  angular error {:.1} deg  within 30 deg: {:.0}%", m.loss, m.angular_error_deg, m.acc_30deg * 100.0);
            println!("\nCue-dependent variability of E rates at end of delay, by system:");
            for (s, n, v) in delay_cue_variability_by_system(&net, &test[..64]) {
                println!("  {:<18} {:3} regions   {:.4}", s, n, v);
            }
            if let Some(dir) = a.get("out") {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                net.save_params(&Path::new(dir).join("params.bin"))?;
                net.export_weights_csv(&Path::new(dir).join("effective_weights.csv"))?;
                println!("\nsaved params.bin and effective_weights.csv to {dir}");
            }
            Ok(())
        }
        "compare" => {
            let tcfg = train_config(&a)?;
            let task = MemoryGuidedTask::default();
            println!("{}\n", conn.summary());
            let mut rows = Vec::new();
            for mode in MaskMode::ALL {
                let cfg = build_config(&a, &conn, mode)?;
                let mut net = ConnectomeRnn::new(&conn, cfg)?;
                net.check_constraints()?;
                eprintln!("[{}] {}", mode.name(), describe_net(&net));
                let rep = train(&mut net, &task, &tcfg, |_| {});
                net.check_constraints()?;
                let test = make_trials(&task, 256, 0xC0FFEE);
                let m = evaluate(&net, &test);
                rows.push((mode, net.edges.len(), net.num_params(), rep.final_eval.loss, m, rep.seconds));
                eprintln!("[{}] done: held-out angular error {:.1} deg", mode.name(), rows.last().unwrap().4.angular_error_deg);
            }
            println!("{:<18} {:>9} {:>9} {:>11} {:>13} {:>10} {:>7}", "wiring", "synapses", "params", "eval-loss", "ang.err(deg)", "acc<30deg", "sec");
            for (mode, ne, np, _l, m, secs) in rows {
                println!("{:<18} {:>9} {:>9} {:>11.4} {:>13.1} {:>9.0}% {:>7.1}", mode.name(), ne, np, m.loss, m.angular_error_deg, m.acc_30deg * 100.0, secs);
            }
            println!("\n(held-out: 256 trials; chance angular error = 90 deg; same seed, task and hyperparameters for every row)");
            Ok(())
        }
        other => Err(format!("unknown command '{other}' (try --help)")),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
