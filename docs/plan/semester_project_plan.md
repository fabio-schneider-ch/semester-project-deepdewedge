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

## Week 2 — Experiments: Missing Wedge Angle Sensitivity

**Goal:** Systematically answer the research question with 3 sub-experiments.  
**Key output:** FSC curves and power spectra for all conditions.

---

### Experiment Design

```
Sub-experiment 1: Inference-time angle error
   Fixed model (trained at 50°) → refine with mw_angle = 30/40/50/60/70°
   Question: Does using the wrong angle at inference time hurt?

Sub-experiment 2: Training-time angle error  
   Train 3 models: mw_angle = 40°, 50°, 60° (all other params identical)
   Evaluate all with correct angle (50°)
   Question: Does training with the wrong angle hurt more or less than inference error?

Sub-experiment 3: Fourier-space fill quality
   For the best and worst conditions from Exp 1+2:
   Compute difference map (refined − input) in Fourier space
   Question: Are the filled frequencies physically reasonable?
```

---

### Day 6 — Sub-Experiment 1: Inference-Time Angle Error

**Morning (4 h)**

- Set up 5 inference runs using the provided checkpoint, varying only `mw_angle`:
  ```
  run_mw30/config.yaml   mw_angle: 30
  run_mw40/config.yaml   mw_angle: 40
  run_mw50/config.yaml   mw_angle: 50  ← correct
  run_mw60/config.yaml   mw_angle: 60
  run_mw70/config.yaml   mw_angle: 70
  ```
- Launch all 5 runs (inference is fast, ~minutes each)

**Afternoon (4 h)**

- For each result, compute FSC against the ground truth (tomo_all_frames.rec)
- Plot: 5 FSC curves on one figure, highlight the correct-angle (50°) curve
- Extract FSC-0.5 resolution for each condition → one summary table


| mw_angle | FSC-0.5 resolution (Å) | Notes                         |
| -------- | ---------------------- | ----------------------------- |
| 30°      |                        | too narrow — fills too little |
| 40°      |                        |                               |
| 50°      |                        | ← correct                     |
| 60°      |                        |                               |
| 70°      |                        | too wide — fills too much     |


**End-of-day checkpoint:** FSC comparison figure for Sub-Exp 1.

---

### Day 7 — Analyze Sub-Experiment 1 + Prepare Sub-Experiment 2

**Morning (4 h)**

- Qualitative comparison: side-by-side XZ slices for all 5 conditions
- Interpret: at what angle error does quality visibly degrade?
- Compute XZ/XY power spectrum ratio for each condition
(a perfect isotropic result would have ratio ≈ 1)

**Afternoon (4 h)**

- Set up 3 training jobs for Sub-Experiment 2 (angle error during training):
  ```
  train_mw40/config.yaml   mw_angle: 40, chans: 32, num_epochs: 200
  train_mw50/config.yaml   mw_angle: 50, chans: 32, num_epochs: 200  ← correct
  train_mw60/config.yaml   mw_angle: 60, chans: 32, num_epochs: 200
  ```
- Launch all 3 training jobs (if multi-GPU available: run in parallel)
- While waiting: write up Sub-Exp 1 findings in the notebook (2–3 paragraphs)

**End-of-day checkpoint:** 3 training jobs running. Sub-Exp 1 analysis complete.

---

### Day 8 — Analyze Power Spectra + Monitor Training

**Morning (4 h) — Training is running**

- Deep analysis of power spectra from Sub-Exp 1:
  - For `mw_angle=30°`: the model under-fills the wedge → residual anisotropy visible
  - For `mw_angle=70°`: the model over-fills the wedge → introduces artifacts?
  - Visualize these as 2D Fourier slices, not just 1D FSC curves
- Write physical interpretation: what does each failure mode mean?

**Afternoon (4 h)**

- If any training runs from Day 7 are complete: run inference and compute FSC
- Compare loss curves: do models trained with wrong angle converge differently?
- Start drafting the experiment section of the written summary

**End-of-day checkpoint:** Power spectrum figures for Sub-Exp 1 ready.

---

### Day 9 — Sub-Experiment 2: Training-Time Angle Error

**Morning (4 h)**

- All 3 models should be trained → run inference for all with `mw_angle=50°`
- Compute FSC for each → add to summary table:


| Trained mw_angle | Inferred mw_angle | FSC-0.5 (Å) |
| ---------------- | ----------------- | ----------- |
| 40°              | 50°               |             |
| **50°**          | **50°**           | ← baseline  |
| 60°              | 50°               |             |


**Afternoon (4 h)**

- Compare Sub-Exp 1 vs Sub-Exp 2: which error source is more damaging?
  - Inference-time error vs training-time error
- Hypothesis to test: inference-time angle determines what gets filled,
training-time angle determines how well the model learns to fill it
- Generate combined figure: FSC curves for both experiments in one panel

**End-of-day checkpoint:** Full FSC comparison across both experiments.

---

### Day 10 — Sub-Experiment 3: Fourier Fill Quality + Supervisor Sync

**Morning (4 h)**

- For best condition (50°/50°) and worst condition: compute Fourier difference maps
  ```python
  diff_fourier = fft(refined_tomo) - fft(input_tomo)
  # Visualize: which k-space regions gained power?
  ```
- Check: does the power increase concentrate inside the missing wedge region?
- For the over-filled case (mw_angle=70°): does power increase outside the wedge too?

**Afternoon (4 h)**

- **Meet with supervisor**: present preliminary results
  - Show FSC curves from Sub-Exp 1 and 2
  - Show Fourier difference maps
  - Discuss: is there a surprising or particularly interesting finding to highlight?
- Agree on which 3–4 figures will go into the report

**End-of-day checkpoint:** All 3 sub-experiments complete. Direction for Week 3 confirmed.

---

## Week 3 — Deepening + Delivery

**Goal:** Extract one clear, physically meaningful conclusion. Prepare all deliverables.

---

### Day 11 — Deepen the Most Interesting Finding

**Morning (4 h)**  
*Based on Day 10 discussion — choose one:*

**Option A** — If Sub-Exp 1 showed a clear asymmetry (e.g., over-filling is worse than under-filling):

- Test at finer angle resolution: 45°, 47°, 50°, 53°, 55° (is there a cliff?)
- Find the acceptable error tolerance: ±N° before FSC degrades by >10%

**Option B** — If the Fourier fill analysis revealed artifacts outside the wedge:

- Quantify: what fraction of the recovered power is inside vs. outside the true wedge?
- Compute a "fill precision" metric across all conditions

**Afternoon (4 h)**

- Run any additional experiments needed
- Start assembling the final results notebook (clean, commented, reproducible)

---

### Day 12 — Results Notebook + Figure Polish

**Morning (4 h)**

- Finalize the results notebook:
  - Cell 1: Load all experimental results
  - Cell 2: FSC comparison figure (publication-quality)
  - Cell 3: Power spectrum / Fourier fill figure
  - Cell 4: Summary table with all conditions
- Ensure every figure has: axis labels, units, legend, title

**Afternoon (4 h)**

- Generate all figures at 300 dpi, save as PNG and PDF
- Review: do the figures tell a coherent story without reading any text?
- Clean up all config files and scripts so experiments are reproducible

---

### Day 13 — Written Summary

**Goal:** 1–2 pages that the supervisor can directly use for the report.

**Structure:**

```
1. Methods (0.5 page)
   - Brief description of DeepDeWedge pipeline
   - How mw_angle is used in training vs. inference
   - Description of the 3 sub-experiments
   - FSC as evaluation metric

2. Results (0.5–1 page)
   - Sub-Exp 1: inference-time angle sensitivity (with FSC table)
   - Sub-Exp 2: training-time angle sensitivity (with FSC table)
   - Sub-Exp 3: Fourier fill quality analysis
   - Key finding: [your conclusion here]

3. Conclusion (0.25 page)
   - Practical guidance: how much angle error is acceptable?
   - Recommendation for real experiments
```

- Write draft by end of day, send for review

---

### Day 14 — Buffer + Repository Cleanup

**Morning (4 h)**

- Address any feedback on the written summary
- Final check: can someone else reproduce your results from scratch?
  - All random seeds set?
  - Config files saved for each experiment?
  - Intermediate data files present?

**Afternoon (4 h)**

- Organize the repository:
  ```
  tutorial/
  ├── experiments/
  │   ├── subexp1_inference_angle/   (5 config files + results)
  │   ├── subexp2_training_angle/    (3 config files + results)
  │   └── subexp3_fourier_fill/      (analysis notebook)
  ├── figures/                       (all high-res figures)
  └── results_summary.ipynb          (main results notebook)
  ```
- Write a short `README` for the experiments folder

---

### Day 15 — Final Delivery

- Send supervisor:
  - Written summary (final version)
  - All high-res figures (labeled and ready to drop into slides/report)
  - Link to organized repository / experiment folder
  - Brief verbal walkthrough of the main finding

---

## Week 4 — Report + Presentation

**You have in hand:**

- Written summary with methods, results, and conclusions → draft directly into report
- High-res figures → drop into slides and report
- Reproducible notebooks → demonstrate live if needed

**Suggested report structure:**

1. Introduction: cryo-ET, missing wedge problem
2. Methods: DeepDeWedge pipeline, experimental design
3. Results: Sub-Exp 1 → 2 → 3, building to main conclusion
4. Discussion: practical implications, limitations
5. Conclusion

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

