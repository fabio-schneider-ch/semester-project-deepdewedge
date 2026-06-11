# Semester Project Plan — DeepDeWedge

---

## Working Files (Maintain from Day 1)

You are expected to maintain two files throughout the project. Both are for your own benefit — one is also shared with your supervisor during check-ins.

### `log.md` — Daily Lab Notebook *(share with supervisor at check-ins)*

Keep this updated every day. It does not need to be polished — honest and specific is more useful than neat.

```
## YYYY-MM-DD

### What I did
- [bullet points of completed tasks]

### What I tried / what failed
- Tried X → got error Y → resolved by Z
- Tried X → did not work, not sure why yet

### Technical details & thoughts
- [equations, parameter choices, observations, hypotheses]
- [anything you would want to remember tomorrow]
```

> **Rule:** If you spent more than 30 minutes on something, it goes in the log — especially if it failed.

---

### `errors/` — Error Screenshot Folder *(for your own reference)*

When you hit an error:
1. Take a screenshot of the full terminal / notebook output
2. Save it as `errors/YYYY-MM-DD_short-description.png`
3. Add one line in `log.md` under "What I tried" describing what caused it

> You do not need to document every warning — only errors that stopped your progress or took time to debug.

---

## How to Get Help

Use this hierarchy in order. Move to the next level only if the current one does not resolve the issue within ~20 minutes.

| Priority | Source | When to use |
|----------|--------|-------------|
| 1 | **Claude / Gemini / GPT** | Error messages, syntax questions, "how does X work", understanding code |
| 2 | **Web search** | Looking for examples, papers, known issues with specific tools |
| 3 | **Supervisor (me)** | Conceptual confusion about the research direction, results that don't make sense, decisions that affect the whole project |
| 4 | **Author's GitHub** | Bugs that look like they might be in the DeepDeWedge code itself → open an issue or check existing issues at the repo |

> When you ask me for help, paste the relevant section of your `log.md` alongside your question. It saves both of us time.

---

## Research Question

> **How robust is DeepDeWedge to incorrect assumptions about the missing wedge angle?**
>
> In real cryo-ET experiments, the exact tilt range is not always precisely known.
> This project systematically investigates what happens when the `mw_angle` parameter
> is set incorrectly — both during training and during inference — and quantifies the
> impact using Fourier Shell Correlation (FSC).

---

## Deliverables Overview


| #   | Deliverable                                                | Due           |
| --- | ---------------------------------------------------------- | ------------- |
| 1   | Reproducible Jupyter notebooks for all experiments         | End of Week 3 |
| 2   | FSC curves comparing all angle conditions                  | End of Week 2 |
| 3   | Fourier-space fill quality analysis (power spectra)        | End of Week 2 |
| 4   | 1–2 page written summary (methods + results + conclusions) | Day 15        |
| 5   | High-resolution figures ready for the report               | Day 15        |


---
iE
## Week 1 — Understanding & Reproduction

**Goal:** Run the full pipeline end-to-end. Build physical intuition for the missing wedge.  
**Key output:** You can explain *why* the method works from a signal-processing perspective.

---

### Day 1 — Physics Background + Environment Setup

**Morning (4 h)**

- Read the DeepDeWedge paper (Wiedemann & Heckel, *Nature Communications* 2024)
  - Focus on: Fig. 1 (overview), the missing wedge explanation, and the Noise2Noise connection
- Read about cryo-ET imaging geometry: why a ±60° tilt range creates a wedge-shaped gap in Fourier space
- Sketch by hand what the missing wedge looks like in 3D Fourier space at different tilt angles

**Afternoon (4 h)**

- Set up the conda environment and install the `ddw` package
- Create your working files — do this before anything else:
  ```
  mkdir errors/
  touch log.md
  ```
  Write your first log entry: what you read this morning, your initial understanding of the missing wedge
- Run **Step 0** (`Step0_DataDownload.ipynb`): download the Tomo110 dataset
- Visualize 3 orthogonal slices (XY, XZ, YZ) of the raw tomogram using `plot_tomo_slices()`
- Note: XZ slice should look noticeably blurrier than XY — this is the missing wedge effect

**End-of-day checkpoint:** Can you explain why the XZ plane looks different from XY? Write the answer in `log.md`.

---

### Day 2 — Data Preparation + Missing Wedge Visualization

**Morning (4 h)**

- Run **Step 1** (`Step1_DataPreparation.ipynb`): extract sub-tomograms with default config
- Visualize 5 extracted sub-tomogram pairs (even frame vs odd frame) side by side
- Read `ddw/utils/missing_wedge.py` in full — understand how the binary mask is generated

**Afternoon (4 h) — Key visualization exercise**

- Write a notebook cell that generates and plots missing wedge masks for:
`mw_angle = [30°, 40°, 50°, 60°, 70°]`
Show both a 2D slice through the mask and the 3D isosurface
- For the 50° mask: compute what fraction of Fourier space is missing
- Read `ddw/utils/fourier.py`: understand how the mask is applied to a volume

**End-of-day checkpoint:** 5-panel figure showing how the wedge shape changes with angle.

---

### Day 3 — Training Pipeline + Launch First Training Run

**Morning (4 h)**

- Read `ddw/utils/subtomo_dataset.py`:
  - Why are sub-tomograms extracted at √2 × target size?
  - How does random 3D rotation work, and why does it help?
  - What exactly gets fed into the model vs. what is the training target?
- Read `ddw/utils/masked_loss.py`:
  - Why is the loss computed **only** over the missing wedge region?
  - What would happen if you computed loss everywhere?

**Afternoon (4 h)**

- Modify `config.yaml` for a fast training run:
  ```yaml
  fit_model:
    unet_params_dict:
      chans: 32          # reduced from 64
    num_epochs: 200      # reduced from 1000
    batch_size: 5
  ```
- Launch training: `ddw fit-model --config config.yaml`
- While it runs: read `ddw/utils/unet.py` — sketch the U-Net architecture (encoder depth, skip connections)
- Monitor `metrics.csv`: watch fitting loss and val loss decrease

**End-of-day checkpoint:** Training is running. You can draw the U-Net architecture from memory.

---

### Day 4 — Training Deep Dive + Model Architecture

**Morning (4 h) — Training is still running**

- Read `ddw/fit_model.py` in full:
  - How are checkpoints saved? (best by train loss, best by val loss, every N epochs)
  - What is `update_subtomo_missing_wedges_every_n_epochs`? Why is iterative refinement needed?
- Plot the loss curve as training progresses — does validation loss track training loss?

**Afternoon (4 h)**

- Read `ddw/utils/subtomos.py`:
  - How does the sliding window extraction work?
  - What is the linear ramp weighting during reassembly? Why not just average?
- Read `ddw/utils/normalization.py`: when and why do we recompute normalization at inference time?
- Write a short note (half page) explaining the full data flow: raw tomo → subtomos → model → refined tomo

**End-of-day checkpoint:** Training complete. You have a trained checkpoint.

---

### Day 5 — Inference + Baseline Comparison

**Morning (4 h)**

- Run **Step 3** using the **provided pre-trained checkpoint** (fitted_model.ckpt)
— this is your experiment baseline
- Compare input vs. refined tomogram:
  - Side-by-side slices: XY, XZ, YZ planes
  - Qualitative: is the blurring in XZ reduced?

**Afternoon (4 h) — Power spectrum analysis**

- Compute and plot the 3D power spectrum of both input and refined tomogram
- Visualize the 2D slice through the power spectrum at kz=0 (should show the wedge gap)
- After refinement: is the wedge region in Fourier space filled in?
- Implement the FSC function (you will use this throughout Week 2):
  ```python
  def compute_fsc(vol1, vol2):
      """
      Compute Fourier Shell Correlation between two volumes.
      Returns: shell_frequencies, fsc_values
      """
      # Hint: FFT both volumes, compute normalized cross-correlation per shell
  ```

**End-of-day checkpoint:**

- Side-by-side figure: input vs. refined (XY and XZ slices)
- Power spectrum before/after showing wedge filling
- Working FSC function

---

## Quick Reference: Key Parameters


| Parameter         | Location                        | Role                                       |
| ----------------- | ------------------------------- | ------------------------------------------ |
| `mw_angle`        | `config.yaml` → shared          | Half-angle of missing wedge (°)            |
| `chans`           | `config.yaml` → fit_model       | U-Net base channels (32 = fast, 64 = full) |
| `num_epochs`      | `config.yaml` → fit_model       | Training duration                          |
| `subtomo_size`    | `config.yaml` → shared          | Must be divisible by 2³ = 8                |
| `subtomo_overlap` | `config.yaml` → refine_tomogram | Overlap during inference reassembly        |


## Key Files to Read (in order)

```
ddw/utils/missing_wedge.py      ← Day 2
ddw/utils/fourier.py            ← Day 2
ddw/utils/subtomo_dataset.py    ← Day 3
ddw/utils/masked_loss.py        ← Day 3
ddw/utils/unet.py               ← Day 3-4
ddw/fit_model.py                ← Day 4
ddw/utils/subtomos.py           ← Day 4
ddw/refine_tomogram.py          ← Day 5
```

