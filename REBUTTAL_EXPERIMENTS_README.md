# Rebuttal experiments for AdaST (NeurIPS 2026, submission 9309)

Five experiments answering the three reviews, plus the infrastructure to run
them reproducibly. Everything is resumable: a run that already produced
`result.json` is skipped, so you can Ctrl-C and restart freely.

```bash
bash experiments_rebuttal/run_all_rebuttal.sh 0      # everything, on GPU 0
```

Nothing here modifies existing repo files. New architectures live under
`experiments_rebuttal/arch/`, new configs under `experiments_rebuttal/configs/`,
and results under `experiments_rebuttal/results/`.

---

## Priority order

If you cannot run everything before the deadline, run in this order. Exp 1 and
Exp 2 are what move scores; Exp 4 is the one most likely to backfire (see the
warning below).

| # | Experiment | Answers | Cost |
|---|---|---|---|
| 1 | Multi-seed statistics | fyiD Q1, mgw5 lim. 2 | High (dominates the budget) |
| 2 | Gate recovery on known regimes | fyiD Q2, i2iu con 1 | Low (20 nodes) |
| 3 | Parameter-matched control | fyiD Q3, i2iu con 2 | Medium |
| 5 | Long horizon (48) | mgw5 lim. 3, i2iu con 3 | Medium |
| 4 | Larger-scale synthetic | i2iu con 3, mgw5 lim. 1 | Low — **but read the warning** |

---

## Exp 1 — multi-seed statistics

```bash
python experiments_rebuttal/run_multi_seed.py --experiment exp1_seeds \
    --models AdaST STID HimNet STAEformer \
    --datasets PEMS04 PEMS07 PEMS08 PurpleAir \
    --seeds 0 1 2 --gpus 0
python experiments_rebuttal/aggregate_results.py --experiment exp1_seeds --reference AdaST
```

Writes `summary.{md,tex,csv,json}` with mean ± std and Welch t-tests against
AdaST. Seeds are set through `CFG.ENV.SEED`; configs are patched in memory
rather than copied, so hyperparameters keep a single source of truth.

**With three seeds the t-test is very low powered.** If PEMS04 comes back
overlapping, say so plainly and lead the response with Exp 2 instead. A rebuttal
that concedes one benchmark and lands a new diagnostic reads far better than one
that oversells `p = 0.31`.

## Exp 2 — gate recovery (the strongest single addition)

```bash
python experiments_rebuttal/exp2_gate_recovery/make_regime_dataset.py \
    --name SynthTimeRegime --mode time --num-nodes 20 --num-steps 20000 --seed 0
python experiments_rebuttal/_train_one.py \
    -c experiments_rebuttal/configs/SynthTimeRegime.py --seed 0 --gpus 0 \
    --run-dir experiments_rebuttal/results/exp2_gate_recovery/SynthTimeRegime__seed0
python experiments_rebuttal/exp2_gate_recovery/extract_gates.py \
    --cfg experiments_rebuttal/configs/SynthTimeRegime.py \
    --run-dir experiments_rebuttal/results/exp2_gate_recovery/SynthTimeRegime__seed0
```

The regime switches over time in **random-length blocks (150–400 steps),
deliberately not aligned to the 288-step day**. If blocks were day-aligned, the
time-of-day expert could recover the regime by reading the clock rather than the
coupling structure, and the diagnostic would prove nothing. `--align-blocks-to-day`
exists only as a negative control. `--mode node` gives the node-varying variant.

The generator prints a Table-1 style check. At N=20, T=20000 it produces:

| regime | temporal (lag-1) | spatial (mean) |
|---|---|---|
| TD | +0.736 | +0.000 |
| SD | +0.144 | +0.147 |
| SC | +0.487 | +0.154 |

`extract_gates.py` reports per-regime gate means (raw **and** z-scored),
regime-recovery accuracy with a 1000-fold permutation test, one-vs-rest AUC, and
a plot of gate trajectories with the true regime shaded behind them.

The z-scoring is the direct answer to i2iu's "gates only move between 0.2300 and
0.2310". The gate logits are multiplied by correlation measures around 0.33
before the softmax, which compresses the absolute range. A narrow band is not
the same as an uninformative signal — but you have to *show* that, which is what
the AUC and permutation test do.

## Exp 3 — parameter-matched control

```bash
python experiments_rebuttal/param_match.py --cfg baselines/AdaST/PEMS08.py
```

Prints a parameter table and the exact commands to launch the controls.
`AdaSTPM` imports its blocks unchanged from `baselines/AdaST/arch/adast_arch.py`,
so the switches are the only difference. Verified on PEMS08:

| variant | params | vs AdaST |
|---|---|---|
| AdaST (reference impl.) | 3,878,111 | — |
| AdaSTPM, all switches on | 3,878,111 | +0.0% (refactor is faithful) |
| w/o gate | 3,831,732 | −1.2% |
| w/o heterogeneity | 3,835,975 | −1.1% |
| control (none of them) | 3,789,596 | −2.3% |
| control, `feed_forward_dim=288` | 3,980,444 | **+2.6%** |

**The whole coupling apparatus costs 2.3% of parameters.** That is a strong
one-line rebuttal to fyiD Q3 on its own, before any control is trained. Then
train the `feed_forward_dim=288` control, which holds *more* parameters than
AdaST: a control that loses while over-parameterised is far more convincing than
one that loses while under-parameterised.

## Exp 4 — larger-scale synthetic — **read this before running**

```bash
python experiments_rebuttal/exp4_synthetic_scale/run_scale_study.py \
    --num-nodes 50 --num-steps 4000 --seeds 0 1 2 --epochs 60
python experiments_rebuttal/exp4_synthetic_scale/mismatch_vs_datasize.py \
    --num-nodes 20 --steps 200 500 2000 8000 --seeds 0 1 2
```

A CPU pilot (N=20, T=2000, 1 seed, 12 epochs, MLP backbone only, and a
*reimplementation* of the T/S/ST heads — not your Figure 1 code) found that
**the diagonal-dominance pattern did not reproduce**: the coupled ST
architecture won on all three regimes, with all margins inside 2%.

That is a warning, not a verdict. But it is the same thing i2iu suspected, and
if it holds up in your own code, **do not put the scale-up in the rebuttal** —
a reviewer who sees a failed replication of your own motivating experiment will
go down, not up.

The likely mechanism: at T=200 there are roughly 120 training windows, so the
larger ST model overfits and the mismatch penalty is really a small-sample
effect. `mismatch_vs_datasize.py` measures that directly by sweeping T and
plotting the penalty decay. If the penalty decays, the honest and *sharper*
claim is that mismatch costs the most when data is scarce relative to capacity —
which is the regime real ST benchmarks live in. That reframing is defensible.
Asserting scale-invariance and being caught is not.

## Exp 5 — long horizon

```bash
python experiments_rebuttal/run_multi_seed.py --experiment exp5_horizon48 \
    --models AdaST STID STAEformer --datasets PEMS08 \
    --seeds 0 1 2 --output-len 48 --gpus 0
```

Horizon overrides handle the differing key names across baselines
(`out_steps` for AdaST/STAEformer/HimNet, `output_len` for STID). No data
regeneration is needed — `TimeSeriesForecastingDataset` slices arbitrary
input/output lengths from `data.dat`. HimNet works too but is a seq2seq with
teacher forcing and is much slower at 48 steps.

---

## Two bugs in the paper's own data generation

Both are latent at N=10, T=200 and fatal at larger scale. Check your generator
before a reviewer does.

**1. The processes are not stationary as written.** Twelve AR coefficients drawn
from U(0, 0.5) sum to ≈3 in expectation. The row-normalised transition matrix P
is stochastic, so its spectral radius is exactly 1, which makes the SC process
(AR + full-strength diffusion) have loop gain > 1. My first test run overflowed
float32 within a few hundred steps. Both generators here rescale AR coefficients
to sum 0.9 and scale the diffusion terms so AR + diffusion < 1.

**2. Fixed edge density does not scale.** Appendix A fixes edge density at 0.43,
so mean out-degree grows linearly in N and P averages over ever more neighbours.
Measured spatial signal-to-noise for the SD process:

| N | mean out-degree | SNR at density 0.43 | SNR at fixed degree 4 |
|---|---|---|---|
| 10 | 3.3 | 1.59 | 1.25 |
| 20 | 7.7 | 0.99 | 1.15 |
| 50 | 21.5 | 0.57 | 1.77 |
| 100 | 42.5 | 0.39 | 1.49 |

At N=100 the SD regime is almost pure noise, so *no* architecture can win on it
and diagonal dominance disappears — for reasons that have nothing to do with
your thesis. Both generators default to a fixed expected degree
(`--expected-degree 4`); pass `--expected-degree 0` to reproduce the original
protocol.

---

## Also worth fixing (cheap, and reviewers notice)

- The broken `Appendix ??` reference on line 128 that i2iu flagged.
- The spatial mixer complexity claim. Appendix B calls it "linear complexity
  O(N²)", which is self-contradictory, and fyiD caught it. A learned N×N matrix
  has the same O(N²) parameter count and pairwise mixing as attention; what it
  avoids is the softmax over pairs and the per-batch attention computation.
  State that precisely and let Table 4's measured speedups carry the argument.
- PurpleAir's non-standard preprocessing: say plainly that you resampled it
  yourself and offer the script. It is also where your largest gain (4.3%) comes
  from, so pre-empting the reproducibility question is worth a sentence.
