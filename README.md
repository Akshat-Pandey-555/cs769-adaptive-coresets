# Coreset memory under drift in personalised Bayesian federated learning

CS 769 Optimization in Machine Learning, course project.

**G18** Akshat Pandey (23B2175), Jash Dalal (23B1522)

## Overview

CORESET-pFedBayes (Chanda, Modi and Ramakrishnan, ICLR 2024) trains each client of a personalised Bayesian federated learning system (pFedBayes, Zhang et al., ICML 2022) on a small weighted subset of its data, chosen by Accelerated Iterative Hard Thresholding (A-IHT). The paper assumes each client's data is fixed.

This project asks what happens when it is not static i.e. if each client may keep only a small weighted memory and its data keeps arriving, expiring and changing in class mix, how should that memory be maintained? We compare memories that are selected once, re-selected on a schedule, re-selected every round, re-selected with a limit on how many examples may change, and filled at random.

## Setting

| | |
|---|---|
| Data | CIFAR-10, 10 clients, each holding 3 of the 10 classes |
| Model | Bayesian MLP 3072 → 100 → 10, one Gaussian (mean, spread) per weight |
| Stream | each client holds a window of 1,000 examples; from round 20, 100 arrive and the 100 oldest expire every round |
| Drift | rounds 30–49 the arrival mix moves from uniform to 10% / 30% / 60% over the client's three classes; rounds 65–84 it moves back |
| Memory | 200 weighted examples (20% of the window) |
| Training | 105 rounds; per round 5 local passes in batches of 100; ζ = 10; Adam, learning rate 0.001; 3 Monte Carlo samples per loss |
| Coreset solver | A-IHT on 50 Monte Carlo draws, at most 200 non-zero non-negative weights |
| Budget | every arm runs the same number of training steps; only the memory policy differs |

## Method

Each client holds three Gaussian posteriors over the network weights:

| Name | Trained on | Role |
|---|---|---|
| `q` | the whole window | personal model; the one evaluated |
| `q_w` | the weighted memory | coreset model |
| `v_z` | — (pulled towards `q_w`) | local copy of the global model; the only thing sent to the server |

Per batch, a client takes one step on each, in order:

1. `q`: minimise −log-likelihood of the batch + (ζ b / n) · KL(q ‖ v_z)
2. `q_w`: the same on the batch, with each example's log-likelihood multiplied by its memory weight
3. `v_z`: minimise (ζ b / n) · KL(q_w ‖ v_z) with `q_w` held fixed (for plain pFedBayes, `q` replaces `q_w`)

The server sets the global model to the mean of the clients' `v_z`.

Coreset selection (paper Eq. 4): draw 50 networks θ₁…θ₅₀ from `q`, form the matrix G (50 × 1000) of each example's centred log-likelihood under each draw, set y = Σⱼ G[:, j], and solve

  min_w ‖y − G w‖²  subject to  ‖w‖₀ ≤ 200,  w ≥ 0

with A-IHT.

### One round (`pFedBayes.train`)

1. `advance_stream` — new examples arrive, old ones expire; surviving memory weights carry over (random-subset arms refill expired slots at random).
2. `send_parameters` — each client's `v_z` is set to the global model.
3. `user.train` — local training as above.
4. `update_memory` — if the arm solves this round, A-IHT proposes new weights; a swap cap, if any, limits how many memory slots may change. The new memory is used from the next round.
5. `aggregate_parameters` — the global model becomes the mean of the clients' `v_z`.
6. `evaluate` — accuracy of each personal model (posterior mean of `q`) on fixed test sets, and of the global model on all ten classes.

## Repository layout

| File | Contents |
|---|---|
| `config.py` | all constants, the two schedules, and the table of arms (`ARMS`) |
| `main.py` | runs one arm for one seed and saves its history |
| `analyze.py` | turns saved histories into the results tables and paired differences |
| `src/model.py` | `federatedBNN` (means and `rho` per weight, σ = softplus(ρ)); `sample`, `net`, `kl` |
| `src/clientModels/clientModelClass.py` | one client: the three posteriors, the three losses, local training, `select_coreset` |
| `src/bayesianCoresets/accelerated_iht.py` | the A-IHT solver |
| `src/serverModels/serverpFedbayes.py` | the server, the round loop, memory updates, aggregation, evaluation |
| `temporal/stream.py` | loading CIFAR-10, each client's window, arrivals and expiries, the drift schedule, the test sets |
| `temporal/memory.py` | how the memory changes between rounds: `carry_over`, `random_replace`, `cap_swaps`, `should_solve` |
| `run_two_cycle.sh`, `run_memory_baselines.sh` | launch the two-cycle runs and the reservoir / class-balanced runs in the background |
| `results/`, `results_two_cycle/` | saved histories (`<arm>_seed<k>.npz`) for every arm reported below |

## Arms (`config.ARMS`)

| Arm | Memory policy |
|---|---|
| `FULL` | no coreset: plain pFedBayes on the whole window (reference) |
| `B` | random subset; expired slots refilled at random |
| `BAL` | class-balanced random subset (67 / 67 / 66 of the client's three classes) |
| `RES` | reservoir sampling over everything seen so far; memory examples protected from expiry |
| `C` | A-IHT re-selects every round |
| `C80`, `C40`, `C20`, `C0` | A-IHT every round, at most 80 / 40 / 20 / 0 voluntary swaps per round |
| `C0U` | `C0` with equal weights on the selected examples |
| `P23` | A-IHT two rounds out of three |
| `D1A` | A-IHT once at round 0, never updated (empties as examples expire) |
| `D1B` | A-IHT once; the selected examples are protected from expiry |
| `D1C` | A-IHT every round, equal weights on the selected examples |

## Installation

Python 3.10 or later. Tested with torch 2.9.0 and numpy 2.2.6 on CPU.

```
pip install -r requirements.txt
```

Download CIFAR-10 (python version) into `data/`:

```
mkdir -p data && cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
cd ..
```

This creates `data/cifar-10-batches-py/`, the default path. Another location can be given with `--data`.

## Running

Run all commands from the repository root.

Quick check (three rounds, about two minutes):

```
python main.py --arm B --rounds 3 --out /tmp/demo
```

It prints one line per round: mean personal accuracy over the ten clients (3-class task, chance 0.33) and global accuracy (10 classes, chance 0.10).

One full run (105 rounds; about 1.5–2.5 hours on a laptop CPU and 1.6 GB of memory):

```
python main.py --arm C20 --seed 1
```

Results are saved to `results/<arm>_seed<k>.npz`. The two-cycle schedule (180 rounds, a second drift in the reverse direction, target mix 60% / 30% / 10%) is selected with `--schedule two_cycle` and saved to `results_two_cycle/`. `./run_two_cycle.sh` starts its five main arms, and `./run_memory_baselines.sh` starts `BAL` and `RES` on both schedules.

Tables from saved results:

```
python analyze.py                       # one-cycle schedule, results/
python analyze.py --schedule two_cycle  # two-cycle schedule, results_two_cycle/
```

With several seeds of the same arm, `analyze.py` averages them and reports 95% intervals for the paired differences.

### Metric and columns

- `acc` (headline): Acc₀.₅ = 0.5 · accuracy on the current arrival mix + 0.5 · accuracy on a fixed test set drawn at the historical mix, averaged over clients and over the window (for one cycle, rounds 95–104). It rewards both following the present and not forgetting.
- `up`, `down`: accuracy on the class whose share rises to 60% and the class whose share falls to 10% in the first drift.
- `glob`: global-model accuracy on a balanced 10-class test set.
- `overlap`: share of the memory kept from the previous round (from round 20).
- `swaps`: examples entering the memory per round that did not replace an expired one (from round 20).

## Results

### One drift cycle, rounds 95–104

| Arm | acc | global | memory kept | swaps / round |
|---|---|---|---|---|
| D1A (static) | 0.541 | 0.149 | 0.11 | –2.1  |
| D1B (static, protected) | 0.598 | 0.270 | 1.00 | 0 |
| D1C (every round, equal weights) | 0.623 | 0.270 | 0.28 | 124 |
| C (every round) | 0.627 | 0.282 | 0.30 | 120 |
| P23 (2 rounds in 3) | 0.634 | 0.288 | 0.52 | 77 |
| C0 | 0.634 | 0.303 | 0.84 | 0 |
| C40 | 0.642 | 0.310 | 0.67 | 40 |
| C80 | 0.643 | 0.285 | 0.49 | 80 |
| C0U | 0.644 | 0.307 | 0.84 | 0 |
| RES (reservoir) | 0.648 | 0.329 | 0.97 | 5 |
| C20 (at most 20 swaps) | 0.651 | 0.319 | 0.76 | 20 |
| BAL (class-balanced random) | 0.661 | 0.331 | 0.83 | 0 |
| **B (random subset)** | **0.662** | 0.326 | 0.83 | 0 |
| FULL (no memory limit) | 0.688 | 0.352 | – | – |



### Two drift cycles (`acc` by window)

| Arm | first drift, drifted (50–64) | after first (95–104) | second drift, drifted (125–139) | after second (170–179) |
|---|---|---|---|---|
| D1B | 0.596 | 0.574 | 0.602 | 0.578 |
| C | 0.616 | 0.617 | 0.639 | 0.642 |
| RES | 0.641 | 0.631 | 0.650 | 0.631 |
| C20 | 0.634 | 0.638 | 0.655 | 0.652 |
| B | 0.649 | 0.649 | 0.668 | 0.658 |
| BAL | 0.651 | 0.652 | 0.673 | 0.661 |
| FULL | 0.669 | 0.669 | 0.687 | 0.684 |

Accuracy on the class that becomes rare during each drift (drifted windows): A-IHT every round 0.17 / 0.20; random subset 0.27 / 0.29; class-balanced 0.31 / 0.33; reservoir 0.36 / 0.44.

The two-cycle run uses a different historical test mix from the one-cycle run, so its `acc` values for rounds 95–104 differ from the table above. Over 180 rounds a client needs more images of a class than CIFAR-10 has, so after round 156 images that have already expired from a client's window may arrive again.

## Findings

1. **Keeping the memory current matters most.** A memory selected once and never updated is 9 points below one re-selected every round (`C − D1A` = +0.086).
2. **A-IHT does not beat a random subset under drift.** Random subset is ahead of A-IHT every round by 3.5 points, with no solver cost, and stays ahead through a second, reversed drift.
3. **Churn is the main reason.** Unrestricted A-IHT replaces about 120 of its 200 examples every round (30% of the memory survives; random keeps 83%). Limiting the swaps closes most of the gap: 0.627 uncapped, 0.643 at 80 swaps, 0.651 at 20. Equal weights change little (`C − D1C` = +0.004).
4. **The churn is a property of the selection problem.** The regression matches 50 Monte Carlo draws using up to 200 of 1,000 examples, so many different subsets fit almost exactly; which one is returned depends on that round's draws. The same churn appears before any data changes.
5. **Standard continual-learning memories do not beat random overall.** Class balancing matches random and helps the rare class; reservoir sampling helps the rare class most but follows the current mix least.


## Limitations

- One dataset, one small model, one memory size (20%), one set of hyperparameters
- The arms are fixed policies; no drift detection or adaptive update rule is attempted.

## References

1. P. Chanda, S. Modi, G. Ramakrishnan. *Bayesian Coreset Optimization for Personalized Federated Learning.* ICLR 2024.
2. X. Zhang, Y. Li, W. Li, K. Guo, Y. Shao. *Personalized Federated Learning via Variational Bayesian Inference.* ICML 2022.
3. Original code: `github.com/prateekiiest/coresetFedML`.
