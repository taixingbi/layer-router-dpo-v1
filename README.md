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
2. Output: `checkpoints/router-dpo-*/adapter/`

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

### Docker (GPU node)

Image is built and pushed to Docker Hub on every push to `main` (see [`.github/workflows/docker-push.yml`](.github/workflows/docker-push.yml), same pattern as [layer-orchestrator-v1](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/.github/workflows)).

**Secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` in repo Settings → Secrets and variables → Actions.

**Run training** (needs [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v layer-router-dpo-hf:/cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  taixingbi/layer-router-dpo-v1:latest
```

**Merge adapter:**

```bash
docker run --rm --gpus all \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  taixingbi/layer-router-dpo-v1:latest merge \
  --adapter-dir /app/checkpoints/.../adapter \
  --output-dir /app/checkpoints/merged \
  --base-model Qwen/Qwen2.5-1.5B-Instruct
```

Replace `taixingbi` with your Docker Hub username if you fork the repo.

**OOM on 16GB:** free the GPU from other processes, or lower `--max-length` / `--gradient-accumulation-steps` / `--num-train-epochs`.

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

## Deploy and verify

1. **LoRA (recommended):** copy `adapter/` to e.g. `/data/models/router-dpo-v1`, enable vLLM `--enable-lora` / `--lora-modules`.
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
