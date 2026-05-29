# layer-router-dpo-v1

**QLoRA DPO training** for the HuntAI intent router (offline batch job on LAN GPU nodes). Consumes JSONL from [layer-orchestrator-v1 `dpo-router/output`](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router/output); does not run as a cluster service.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — router-only DPO; fast and fits 16GB / RTX 3090.

## Layout

| Path | Role |
|------|------|
| `app/main.py` | End-to-end CLI: fetch → load → train (default), `merge` |
| `app/pipeline.py` | Fetch dataset, validate load, orchestrate training |
| `app/load_jsonl.py` | JSONL → TRL `prompt` / `chosen` / `rejected` |
| `app/train_dpo.py` | QLoRA + `DPOTrainer` |
| `app/export_merge.py` | Optional merge for full-weight vLLM deploy |
| `data/` | Training JSONL (gitignored; default `./data/train.jsonl`, `val.jsonl`) |
| `checkpoints/` | Training output (gitignored) |

## Dataset

Training JSONL is pulled automatically on first run from [layer-orchestrator-v1 `dpo-router/output`](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router/output) into `./data/`. Or use a monorepo sibling: `../layer-orchestrator-v1/dpo-router/output/*.jsonl`.

## Train on GPU node (16GB)

On Debian/Ubuntu, install the venv module once (version must match `python3 --version`, e.g. 3.12):

```bash
sudo apt install -y python3.12-venv
```

```bash
git clone <layer-router-dpo-v1-url> && cd layer-router-dpo-v1
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional: CUDA_VISIBLE_DEVICES, HF_TOKEN, etc.
```

1. **Free GPU** — do not train on the same card as vLLM at ~70% VRAM. Train on `173` while inference stays on `176`, or set `CUDA_VISIBLE_DEVICES` in `.env`.
2. Output: `checkpoints/router-dpo-*/adapter/` (local) or Hugging Face Hub when `HF_REPO_ID` is set.

See [deploy-vllm-inference.md](../huntai-k3s/docs/deploy-vllm-inference.md) for LoRA serving.

### Train (1.5B / RTX 3090)

```bash
source .venv/bin/activate
python -m app.main
```

One command: **fetch** JSONL (if missing) → **load** / validate counts → **train** DPO. Defaults: `Qwen/Qwen2.5-1.5B-Instruct`, `max-length=1024`, output under `checkpoints/router-dpo-qwen25-1.5b-<timestamp>/`.

Override example:

```bash
python -m app.main --output-dir checkpoints/my-run --num-train-epochs 1
```

After `pip install -e .`, run `layer-router-dpo` (same as above).

**OOM on 16GB:** free the GPU from other processes, or lower `--max-length` / `--gradient-accumulation-steps` / `--num-train-epochs`.

### Deploy to EC2 GPU

Push to `main` or run [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) manually (same pattern as [ec2-gpu-vllm-inference](https://github.com/taixingbi/ec2-gpu-vllm-inference/blob/main/.github/workflows/deploy.yml)).

**Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`, `HF_REPO_ID` (e.g. `your-username/layer-router-dpo-v1`)

**Variables:** `DEPLOY_BUCKET`, `EC2_IAM_INSTANCE_PROFILE`, `AWS_REGION` (optional), `AWS_AMI_ID` (optional), `AWS_SECURITY_GROUP_ID` or `AWS_SECURITY_GROUP_NAME`, `EC2_KEY_PAIR` (optional)

`HF_REPO_ID` may also be set as a repo **variable** instead of a secret (repo id is not sensitive).

The workflow ensures a `g5.xlarge` GPU instance (`ec2-gpu-layer-router-dpo-v1`), syncs app + deploy files to S3, and runs `deploy/remote-deploy.sh` via SSM to install the venv and start DPO training. After training, the LoRA adapter is uploaded to `HF_REPO_ID` on Hugging Face Hub. A local copy also lands under `/home/ubuntu/layer-router-dpo-v1/checkpoints/` on the instance.

### Flags (`python -m app.main`)

| Flag / env | Meaning |
|------------|---------|
| `--train-jsonl`, `--val-jsonl` | Dataset paths (default `./data/*.jsonl`) |
| `--base-model` | HuggingFace id (`BASE_MODEL`) |
| `--output-dir` | Checkpoint root (default auto timestamp under `checkpoints/`) |
| `--max-length` | `MAX_LENGTH` (default 2048) |
| `--num-train-epochs` | `NUM_TRAIN_EPOCHS` |
| `--gradient-accumulation-steps` | `GRAD_ACCUM` |
| `--per-device-train-batch-size` | `PER_DEVICE_TRAIN_BATCH_SIZE` |
| `--hf-repo-id` | `HF_REPO_ID` — upload adapter to this Hub repo after training |

## Hugging Face Hub

After training, set `HF_REPO_ID` and `HF_TOKEN` to publish the LoRA adapter:

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=your-username/layer-router-dpo-v1
python -m app.main
```

The repo will contain `adapter/` (weights + tokenizer), `train_meta.json`, and a generated `README.md`. Use with vLLM `--enable-lora` or `PeftModel.from_pretrained(base, HF_REPO_ID)`.

## Deploy and verify

1. **LoRA (recommended):** use the Hub repo (`HF_REPO_ID`) or copy `adapter/` locally, enable vLLM `--enable-lora` / `--lora-modules`.
2. **Merged:**

```bash
python -m app.main merge \
  --adapter-dir checkpoints/.../adapter \
  --output-dir /data/models/merged \
  --base-model Qwen/Qwen2.5-1.5B-Instruct
```

3. **Test:** `router_model` on `POST /v1/orchestrator/eval/router`, then gold eval on orchestrator.
4. **Promote:** set `LLM_MODEL` when match rate improves.

## Tests

```bash
pip install pytest
pytest tests/ -v -k "not production"
```

## See also

- [layer-orchestrator-v1/dpo-router](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router) — dataset build
- [dpo-router/output on GitHub](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router/output) — committed JSONL
