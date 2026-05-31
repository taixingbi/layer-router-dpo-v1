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
| **DPO** (default) | [aval/dpo-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/aval/dpo-router/output) |
| **SFT** | [aval/sft-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/aval/sft-router/output) |

Or use a monorepo sibling: `../layer-orchestrator-v1/aval/{dpo,sft}-router/output/*.jsonl`.

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

One command: **fetch** JSONL (if missing) → **load** / validate counts → **train**. Defaults: `Qwen/Qwen2.5-1.5B-Instruct`, `max-length=1024`, output under `checkpoints/router-{method}-qwen25-1.5b-<timestamp>/`.

After `pip install -e .`, run `layer-router-train` (alias: `layer-router-dpo`).

**OOM on 16GB:** lower `--max-length` / `--gradient-accumulation-steps` / `--num-train-epochs`.

### Deploy to EC2 GPU

Push to `main` or run [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) manually.

**Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`

**Variables:** `DEPLOY_BUCKET`, `EC2_IAM_INSTANCE_PROFILE`, `TRAIN_METHOD` (`dpo` or `sft`, default `dpo`), `HF_REPO_ID` (optional), `AUTO_TERMINATE_EC2` (default `true`)

The workflow ensures a `g5.xlarge` GPU instance (`ec2-gpu-layer-router-train-v1`), syncs app + deploy files to S3, and runs `deploy/remote-deploy.sh` via SSM. After training, the LoRA adapter uploads to Hugging Face Hub (default `{user}/layer-router-{method}-v1`).

**Note:** EC2 tag changed from `ec2-gpu-layer-router-dpo-v1` to `ec2-gpu-layer-router-train-v1`. Terminate any old instance manually if still running.

### Flags (`python -m app.main`)

| Flag / env | Meaning |
|------------|---------|
| `--method` / `TRAIN_METHOD` | `dpo` or `sft` (default `dpo`) |
| `--train-jsonl`, `--val-jsonl` | Dataset paths (default `./data/{method}/*.jsonl`) |
| `--base-model` | HuggingFace id (`BASE_MODEL`) |
| `--output-dir` | Checkpoint root (default auto timestamp under `checkpoints/`) |
| `--max-length` | `MAX_LENGTH` |
| `--num-train-epochs` | `NUM_TRAIN_EPOCHS` |
| `--gradient-accumulation-steps` | `GRAD_ACCUM` |
| `--per-device-train-batch-size` | `PER_DEVICE_TRAIN_BATCH_SIZE` |
| `--hf-repo-id` | `HF_REPO_ID` — upload adapter after training |
| `--beta` | DPO only (`DPO_BETA`) |

## Hugging Face Hub

Default repos (override with `HF_REPO_ID`):

- DPO: `your-username/layer-router-dpo-v1`
- SFT: `your-username/layer-router-sft-v1`

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=your-username/layer-router-sft-v1
TRAIN_METHOD=sft python -m app.main
```

## Tests

```bash
pip install pytest
pytest tests/ -v -k "not production"
```

## See also

- [aval/dpo-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/aval/dpo-router/output) — DPO JSONL
- [aval/sft-router/output](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/aval/sft-router/output) — SFT JSONL
