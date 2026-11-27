# Running

Tests run locally in Docker. Training and evaluation run on Colab (L4 or A100), with the project folder on Google Drive.

## 1. Tests (local, Docker)

```bash
docker build -t hcdpo .
docker run --rm hcdpo
```

Expected: `17 passed`. The first build downloads CPU torch and the other packages; later builds reuse them.

Optional tiny end-to-end run on the CPU (downloads the 0.5B model and the dataset into the container; takes a few minutes):

```bash
docker run --rm hcdpo python -m hcdpo.data --config configs/debug.yaml
docker run --rm hcdpo sh -c "python -m hcdpo.data --config configs/debug.yaml && python -m hcdpo.train --config configs/debug.yaml"
```

## 2. Colab setup

1. Put this folder on Drive at `MyDrive/Projects/hc-dpo`.
2. Add a Colab secret `WANDB_API_KEY` (key icon in the left sidebar) with notebook access on.
3. Open `notebooks/colab_runner.ipynb`, choose an L4 or A100 runtime, and run the setup cell. It mounts Drive, sets the key and installs the packages. Rerun it after every disconnect.

Every run below is safe to rerun: finished runs are skipped and an interrupted run resumes from its latest checkpoint and continues the same W&B run.

## 3. DPO sweep

```bash
bash scripts/run_dpo_sweep.sh
```

Prepares the data, evaluates the base model, then trains and evaluates beta in {0.05, 0.1, 0.25, 0.5}. Each run writes `metrics.json` and `generations.jsonl` to `outputs/<run_name>/`.

## 4. HC-DPO sweep

Pick four epsilons that span the `kl_mean` values of the DPO runs, then:

```bash
EPSILONS="1 2 4 8" bash scripts/run_hcdpo_sweep.sh
```

## 5. Seeds

Take the DPO beta with the highest `reward_mean`, and that run's `kl_mean` as the epsilon:

```bash
bash scripts/run_seeds.sh <beta> <epsilon>
```

## 6. Analysis

Open `notebooks/analysis.ipynb` on Colab (or anywhere with pandas and matplotlib), set `OUTPUT_DIR`, and run it. It reads the `metrics.json` and `train_log.jsonl` files and does not need W&B.

## Single commands

```bash
python -m hcdpo.data --config configs/dpo.yaml
python -m hcdpo.evaluate --config configs/dpo.yaml --base
python -m hcdpo.train --config configs/dpo.yaml --set beta=0.1 seed=42
python -m hcdpo.evaluate --config configs/dpo.yaml --set beta=0.1 seed=42
python -m hcdpo.train --config configs/hcdpo.yaml --set epsilon=2.0 seed=42
python -m hcdpo.evaluate --config configs/hcdpo.yaml --set epsilon=2.0 seed=42
```

Evaluate with the same `--config` and `--set` as the training run. Add `wandb.enabled=false` to `--set` to run without W&B.

## Memory

`configs/dpo.yaml` and `configs/hcdpo.yaml` use batch size 8 with 4 accumulation steps (effective 32), which fits an L4. If a run goes out of memory, add `batch_size=4 grad_accum_steps=8` to the sweep command.
