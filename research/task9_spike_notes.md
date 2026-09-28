# Task 9 evaluation spike: notes

Written 2026-09-27. Numbers come from `python -m research.evaluation_spike` (seed 42);
tables are in `outputs/tables/spike_*.csv` and the figure is `outputs/figures/spike_masking_effect.png`.

## 1. Do real schedules appear in the data?

Lab-visit days for all 100,000 patients, 2018 to 2024. A patient "follows" a schedule when they
have at least two gaps and every gap is within 30 days of the interval.

| Schedule | Follows it | Run of 2+ near gaps | Run of 3+ | Visits a full schedule needs | Patients with that many |
|---|---|---|---|---|---|
| 3-month | 82 (0.08%) | 629 | 21 | 30 | 0 |
| 6-month | 53 (0.05%) | 491 | 22 | 16 | 0 |
| 12-month | 52 (0.05%) | 302 | 11 | 8 | 402 |

Off-policy evaluation needs patients whose observed care resembles the schedule being scored.
There are almost none, and no patient has enough visits for a full 3- or 6-month schedule.

## 2. How far does risk move when one lab visit is hidden?

GRU (Task 7), test split, 8,342 patients with at least one pre-cutoff lab visit.

| Hidden visit | Median abs change | 90th pct | Share over 0.05 | Share crossing 0.5 | AUROC |
|---|---|---|---|---|---|
| Last | 0.013 | 0.057 | 12.8% | 3.1% | 0.853 (from 0.854) |
| Random | 0.011 | 0.055 | 11.8% | 2.9% | 0.853 |

The median change is about a third of the model's own MC-dropout SD (0.036). One lab visit moves
the prediction less than the model's noise, so masking can rank schedules only weakly here.

## 3. What would a small Delphi-style simulator take?

Source: github.com/gerstung-lab/Delphi (read 2026-09-27). nanoGPT-based transformer; inputs are
(patient id, age in days, token) rows; age enters through a sinusoidal age encoding instead of
positions; two heads predict the next token and, through an exponential waiting-time loss, when it
happens. "No event" tokens are inserted every few years. Only discrete tokens are supported, but
our labs are already binned low/normal/high, so they fit. The demo config is 12 layers, width 120,
vocab 1,270, 5,000 iterations; the README reports about 10 minutes on one GPU for the demo and
1 GPU-hour (V100) for Delphi-2M on 400K UK Biobank patients. Code is MIT; the released weights are
CC BY-NC-ND, which does not matter because we would train our own. A synthetic example dataset ships
in `data/ukb_simulated_data/`.

Estimated effort (a judgement, not measured):

| Step | Estimate |
|---|---|
| Rebuild sequences over the full 2018-2024 window and write Delphi's binary format | 0.5-1 day |
| Train the demo-sized model on this Mac (MPS or CPU; speed untested) and tune | 1-2 days |
| Change `generate()` so a schedule can force a lab visit at a chosen age and sample only its result | 1-2 days |
| Check simulated trajectories against the data (prevalence, gaps, progression rate) | 1-2 days |
| Total | about 4-7 working days |

Main caveat: a simulator learns whatever this generator did. Lab timing here is random and one lab
visit barely moves risk, so a simulator trained on this data will probably show little difference
between schedules. That would describe the synthetic generator, not care. The simulator becomes
worth building on data where timing and results carry signal (MIMIC-IV or EHRSHOT).
