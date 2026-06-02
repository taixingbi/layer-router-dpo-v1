# layer-router-train-v1

**QLoRA DPO and SFT training** for the HuntAI intent router (offline batch job on LAN GPU nodes). Does not run as a cluster service.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — fast and fits 16GB / RTX 3090.

## Layout

| Path | Role |
|------|------|
| `app/main.py` | End-to-end CLI: fetch → load → train (default), `merge` |
| `app/pipeline.py` | Fetch dataset, validate load, orchestrate training |
| `app/method_config.py` | Method-aware paths, checkpoint names, HF repo defaults |
| `app/load_jsonl.py` | JSONL → TRL DPO or SFT rows |
| `app/train_dpo.py` | QLoRA + `DPOTrainer` |
| `app/train_sft.py` | QLoRA + `SFTTrainer` |
| `app/export_merge.py` | Optional merge for full-weight vLLM deploy |
| `data/dpo/`, `data/sft/` | Cached JSONL per method (gitignored) |
| `checkpoints/` | Training output (gitignored) |

## Dataset

Training JSONL is pulled automatically on first run into `./data/{method}/`:

| Method | Orchestrator output |
|--------|---------------------|
| **DPO** (default) | [router-eval/dpo-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/dpo-router/output) |
| **SFT** | [router-eval/sft-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/sft-router/output) |

Or use a monorepo sibling: `../layer-orchestrator-v1/router-eval/{dpo,sft}-router/output/*.jsonl`.

## Train on GPU node (16GB)

```bash
git clone <layer-router-train-v1-url> && cd layer-router-train-v1
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # TRAIN_METHOD, HF_TOKEN, etc.
```

### DPO (default)

```bash
python -m app.main
python -m app.main --method dpo
```

### SFT

```bash
python -m app.main --method sft
TRAIN_METHOD=sft python -m app.main
```

One command: **fetch** JSONL (if missing) → **load** / validate counts → **train**. Defaults: `Qwen/Qwen2.5-1.5B-Instruct`, `max-length=1024`, output under `checkpoints/router-{method}-qwen2.5-1.5b-<timestamp>/`.

After `pip install -e .`, run `layer-router-train` (alias: `layer-router-dpo`).

**OOM on 16GB:** lower `--max-length` / `--gradient-accumulation-steps` / `--num-train-epochs`.

### Deploy to EC2 GPU

Push to `main` or run [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) manually.

**Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`

**Variables:** `DEPLOY_BUCKET`, `EC2_IAM_INSTANCE_PROFILE`, `EC2_INSTANCE_TYPE` (default `g5.xlarge`), `TRAIN_METHOD` (`dpo` or `sft`), `BASE_MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct`), `HF_REPO_FEATURE` (default `router`), `HF_REPO_VERSION` (default `0.00`; e.g. `0.01`, `1.00`), `HF_REPO_ID` (optional full override), `AUTO_TERMINATE_EC2` (default `true`)

The workflow ensures a GPU instance (default `g5.xlarge`, tag `ec2-gpu-layer-router-train-v1`), syncs app + deploy files to S3, and runs `deploy/remote-deploy.sh` via SSM. After training, the LoRA adapter uploads to Hugging Face Hub (default `{user}/layer-router-{method}-v1`).

**Note:** EC2 tag changed from `ec2-gpu-layer-router-dpo-v1` to `ec2-gpu-layer-router-train-v1`. Terminate any old instance manually if still running.

### Flags (`python -m app.main`)

| Flag / env | Meaning |
|------------|---------|
| `--method` / `TRAIN_METHOD` | `dpo` or `sft` (default `dpo`) |
| `--train-jsonl`, `--val-jsonl` | Dataset paths (default `./data/{method}/*.jsonl`) |
| `--base-model` | HuggingFace id (`BASE_MODEL`); also drives Hub repo model slug |
| `--output-dir` | Checkpoint root (default auto timestamp under `checkpoints/`) |
| `--max-length` | `MAX_LENGTH` |
| `--num-train-epochs` | `NUM_TRAIN_EPOCHS` |
| `--gradient-accumulation-steps` | `GRAD_ACCUM` |
| `--per-device-train-batch-size` | `PER_DEVICE_TRAIN_BATCH_SIZE` |
| `--hf-repo-id` | `HF_REPO_ID` — full Hub repo override |
| `HF_REPO_FEATURE` | Hub repo segment (default `router`) |
| `HF_REPO_VERSION` | Hub repo version segment (default `0.00`; use `0.01`, `1.00`, …) |
| `--beta` | DPO only (`DPO_BETA`) |

## Hugging Face Hub

Default Hub repos (owner = `HF_TOKEN` user, override with `HF_REPO_ID`):

- DPO: `{user}/router-qwen2.5-1.5b-dpo-0.00`
- SFT: `{user}/router-qwen2.5-1.5b-sft-0.00`

Pattern: `{HF_REPO_FEATURE}-{model-slug}-{method}-{HF_REPO_VERSION}` where `model-slug` is derived from `BASE_MODEL` (e.g. `Qwen/Qwen2.5-7B-Instruct` → `qwen2.5-7b`). Bump `HF_REPO_VERSION` to `0.01` or `1.00` for the next release repo.

GitHub Actions variables (Settings → Variables):

| Variable | Example |
|----------|---------|
| `EC2_INSTANCE_TYPE` | `g5.xlarge` or `g5.2xlarge` |
| `TRAIN_METHOD` | `sft` |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_REPO_FEATURE` | `router` |
| `HF_REPO_VERSION` | `0.00` or `0.01` |
| `HF_REPO_ID` | `taixingbi/router-qwen2.5-1.5b-sft-0.00` (optional full override) |

Do **not** set `HF_REPO_MODEL` in GitHub — the Hub slug is derived from `BASE_MODEL`. Delete that variable if it exists from an older setup.

```bash
export HF_TOKEN=hf_...
TRAIN_METHOD=sft python -m app.main
# uploads to taixingbi/router-qwen2.5-1.5b-sft-0.00

BASE_MODEL=Qwen/Qwen2.5-7B-Instruct HF_REPO_VERSION=0.01 TRAIN_METHOD=sft python -m app.main
# uploads to taixingbi/router-qwen2.5-7b-sft-0.01
```

## Tests

```bash
pip install pytest
pytest tests/ -v -k "not production"
```

## See also

- [router-eval/dpo-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/dpo-router/output) — DPO JSONL
- [router-eval/sft-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/sft-router/output) — SFT JSONL
