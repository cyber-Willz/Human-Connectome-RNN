# human_connectome_rnn

Recurrent neural networks whose architecture is constrained by a **human structural connectome**. Zero dependencies (std only), exact BPTT, multi-threaded.

## What is constrained

Each brain region becomes an excitatory (E) and an inhibitory (I) population, so R regions give N = 2R units.

| Property | How it is imposed |
|---|---|
| Which regions talk to each other | Inter-regional synapses exist **only** where the connectome has a tract. The trainable set *is* the allowed set, so there is no mask to forget. |
| Long-range projections | E units only; they target both E and I units of the receiving region (feed-forward inhibition). |
| Local circuit | Full EE / EI / IE / II block inside every region. |
| Dale's law | Sign fixed by presynaptic unit; learning changes magnitude only: `w = sign * softplus(theta)`. |
| Conduction delays | `delay = max(1, round(fibre_length_mm / (velocity * dt)))` steps, per region pair (default 5 m/s, dt 5 ms). Gradients flow through the delays. |
| Input / output | Input enters only E units of input regions (default: visual); readout reads only E units of output regions (default: somatomotor). |

Dynamics: `x_{t+1} = (1-a) x_t + a (b + W_in u_t + sum_e w_e r_{t-d_e})`, `r = tanh(x)`, `a = dt/tau`.

`check_constraints()` re-verifies Dale's law and the wiring mask on the realised dense matrix (done before and after every training run in the CLI).

## Data

* **Real human data**: `sh scripts/fetch_tvb66.sh` downloads the 66-region diffusion-MRI group-average connectome from `the-virtual-brain/tvb-data` (check its docs for provenance/citation). Load with `--tvb data/tvb66`. Regions are labelled by Desikan-Killiany gyri; the "system" grouping (Visual, Somatomotor, ...) is a **name-based heuristic**, not a functional parcellation.
* **Your own data**: `--csv weights.csv --lengths lengths_mm.csv --labels labels.csv` (labels: `name[,hemi[,system[,x,y,z]]]`).
* **Synthetic** (`--synthetic N` regions per hemisphere): human-like organisation (7 Yeo-style systems on a brain-shaped geometry, exponential distance rule, within-system clustering, homotopic links, hubs, log-normal weights). A stand-in for pipelines/tests, **not** measured data.

Note: other TVB connectomes (76/80/96 regions) have discrete 0-3 asymmetric weights and look tracer-derived, not diffusion tractography; they were not used here. Matrices are symmetrised on load (tractography has no direction); learned weights may become asymmetric.

## Usage

```sh
cargo build --release
sh scripts/fetch_tvb66.sh

./target/release/connectome_rnn inspect --tvb data/tvb66
./target/release/connectome_rnn train   --tvb data/tvb66 --steps 400 --out run1
./target/release/connectome_rnn compare --tvb data/tvb66 --steps 400 --seed 1   # real vs nulls vs dense
cargo test --release            # set HCRNN_TVB66=data/tvb66 to include the real-data loader test
```

As a library: see the example in `src/lib.rs`.

Task: delayed memory-guided response. A cue angle is shown to visual regions; after a 100-200 ms delay a go signal requires reporting (cos, sin) from somatomotor regions. It needs persistent activity carried across the connectome.

Wiring controls for `compare`: **degree-preserving** rewiring (same degrees and weight multiset, geometry/delays kept), **density-matched** random graph, and **dense** all-to-all with the same total strength.

## Verification performed

* Analytic BPTT gradient vs central finite differences (with delays): worst relative error 5.5e-6 over 61 parameters.
* Mask, Dale's law, delay assignment, rewiring degree-preservation, thread-count invariance of gradients, parameter round-trip, learning smoke test, real-data loading.

## Results so far (not a finding)

Real 66-region human connectome, 400 Adam steps, 3 seeds, held-out angular error in degrees (chance 90):

| wiring | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| connectome | 13.2 | 16.6 | 6.8 |
| degree-preserving | 7.2 | 40.6 | 11.0 |
| density-matched | 14.5 | 6.6 | 11.8 |
| dense | 45.6 | 34.9 | 24.9 |

The real wiring is **not** distinguishable from its nulls here: differences are within seed noise. The dense network is consistently slower to train at this fixed budget, which reflects optimisation speed, not final capacity. Runs are still improving at 400 steps; use more seeds/steps before concluding anything.

## Limitations

* Macroscale population model: units are regional E/I populations, not neurons. The E/I split and E-only long-range projections are modelling assumptions.
* Streamline counts are not synaptic strengths; here they set wiring support and initial magnitudes only (log-compressed by default, `--linear-weights` to disable).
* Conduction velocity is a single assumed global value.
* One task so far.
