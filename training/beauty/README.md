# SCUT-FBP5500 Beauty Regressor — Training Kit

End-to-end recipe to train the ResNet-50 regressor that powers
`BeautyAnalyzer` in `face-service/analyzers/beauty_analyzer.py`.

You have A40 / L40 GPU access, so the full run finishes in well under
an hour. Cost on most clouds is < $2.

## What the model does

Takes a 224×224 face crop and outputs a single floating-point number
in the SCUT-FBP5500 native range of 1.0 (least attractive) to 5.0
(most attractive). The dataset averages each image's score across 60
human raters, so the regression target is already smoothed.

The trained model becomes the dominant signal in the AestheticAnalyzer
"chopped score" — see [face-service/analyzers/aesthetic_analyzer.py](../../face-service/analyzers/aesthetic_analyzer.py)
for the blend math.

## Step 1: Get the data

The dataset isn't redistributable in this repo. Grab it from the
official source:

```bash
git clone https://github.com/HCIILAB/SCUT-FBP5500-Database-Release.git data/SCUT-FBP5500
```

After cloning you should have:

```
data/SCUT-FBP5500/
├── Images/                   # 5,500 .jpg files
├── train_test_files/
│   └── split_of_60%training and 40%testing/
│       ├── train.txt         # "Images/AF1.jpg 3.2"
│       └── test.txt
└── ...
```

`train.py` reads from these paths by default; override with
`--data-root` if you put them elsewhere.

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Includes torch, torchvision, timm, albumentations, scikit-learn, tqdm.

## Step 3: Train

### Local / interactive

```bash
python train.py \
  --data-root data/SCUT-FBP5500 \
  --epochs 25 \
  --batch-size 64 \
  --lr 1e-4 \
  --backbone resnet50 \
  --out checkpoints/beauty_regressor.pt
```

On an A40 / L40 expect ~90 s/epoch → ~40 minutes total.

### Hyak (SLURM on klone)

From a login node (`klone-login03`), after cloning the repo and dataset.

**`conda: command not found` on login nodes is normal.** Hyak's conda
module only loads on compute nodes. Use `setup_hyak.sh`:

```bash
cd ~/HCP/training/beauty

# 1) Get a short interactive shell on a GPU node
srun --account=intelligentsystems -p gpu-rtx6k --gres=gpu:rtx6k \
  --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# 2) On the compute node (prompt will NOT be klone-login03):
module load conda
bash setup_hyak.sh --run
exit

# 3) Back on the login node — clone data if needed, then submit
git clone https://github.com/HCIILAB/SCUT-FBP5500-Database-Release.git ~/HCP/data/SCUT-FBP5500
mkdir -p logs checkpoints
sbatch train.slurm
```

Env is stored under `/gscratch/intelligentsystems/$USER/conda_envs/beauty-train`
(not home — home is only 10 GB).

Check queue / logs:

```bash
squeue -u $USER
tail -f logs/beauty-train-<JOBID>.out
```

**Which partition?** Run `hyakalloc`. As of a recent snapshot, the fastest start was
`intelligentsystems` + `gpu-rtx6k` (8 GPUs free). If that queue is busy, try
`cse` + `gpu-a100` (1 GPU free):

```bash
sbatch --account=cse --partition=gpu-a100 --gres=gpu:a100 train.slurm
```

Avoid `gpu-l40s` on those accounts when hyakalloc shows **0 FREE GPUs** — the
job will sit in `PD` until someone finishes.

**How long?** ~45–60 minutes wall time on a single RTX 6000 Ada / A40 / L40
(~90 s/epoch × 25 epochs). The SLURM script requests **1:30:00** for headroom.
Queue wait is usually minutes on rtx6k when GPUs are free; can be hours on
saturated partitions.

What it does:
- Loads pre-trained ResNet-50 (ImageNet) via `timm.create_model`.
- Replaces the classification head with a single linear output.
- Standard data augmentation: random horizontal flip, color jitter,
  random resized crop at 224.
- Adam + cosine LR schedule.
- Logs train/val MSE + Pearson r to stdout each epoch.
- Saves the best-val-Pearson checkpoint.

Expected metrics on the standard test split:
- **MAE ≤ 0.27** (in the 1.0–5.0 score range)
- **Pearson r ≥ 0.88** between predicted and ground-truth scores

The published academic best is r=0.8997 with ResNeXt-50; ResNet-50
lands a couple of points below that.

## Step 4: Eval

```bash
python eval.py \
  --data-root data/SCUT-FBP5500 \
  --checkpoint checkpoints/beauty_regressor.pt
```

Prints test-set MAE, MSE, Pearson r, Spearman ρ. Use this to sanity
check before deploying.

## Step 5: Integrate into face-service

The simplest path: copy the checkpoint into the face-service models
directory and rebuild.

```bash
cp checkpoints/beauty_regressor.pt ../../face-service/models/beauty_regressor.pt
```

`BeautyAnalyzer` finds it via the default `BEAUTY_WEIGHTS_PATH`
(`models/beauty_regressor.pt` relative to the service's working dir).
Restart the face-service container and the next `/analyze` call will
include `beauty_score` and `beauty_score_norm` in its response.

### Alternative: host on Hugging Face Hub

```bash
python export.py \
  --checkpoint checkpoints/beauty_regressor.pt \
  --hf-repo-id your-username/scut-fbp5500-resnet50
```

Then set `BEAUTY_HF_REPO_ID=your-username/scut-fbp5500-resnet50` in
the face-service environment. `BeautyAnalyzer` will pull it on first
load and cache.

## Notes / gotchas

- **Dataset is research-only.** Your trained weights belong to you,
  but redistributing the original `.jpg` files is restricted.
- **Class balance.** The dataset is balanced across gender + race
  (Asian Female / Asian Male / Caucasian Female / Caucasian Male).
  Beware that the rater pool was Chinese university students — the
  scores reflect their aggregate preferences, not a universal truth.
- **Face crops.** SCUT-FBP5500 is already face-cropped. At inference
  time we feed `BeautyAnalyzer` the InsightFace bbox crop so the
  framing roughly matches training.
- **Out-of-distribution photos** (children, very elderly, heavily
  occluded faces, unusual angles) will produce noisy scores that
  saturate near 2.5–3.5. The clamp in `BeautyAnalyzer` keeps the
  output in [1, 5].

## What to do if accuracy is poor

In order of effort:

1. Run more epochs (try 40). Watch for val-MAE overfitting.
2. Bigger backbone: swap `--backbone resnet50` for `convnext_small`
   or `resnext50_32x4d`. ConvNeXt is a strong drop-in.
3. Ensemble: train 3 seeds, average predictions at inference.
4. Larger input: 320 instead of 224 (`--img-size 320`). Often worth
   another 0.5–1 point of Pearson r at 2× compute.
