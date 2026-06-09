# layer-router-train-v1

**QLoRA DPO and SFT training** for the HuntAI intent router (offline batch job on LAN GPU nodes). Does not run as a cluster service.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — fast and fits 16GB / RTX 3090.

## Layout

| Path | Role |
|------|------|
| `app/train/` | QLoRA pipeline: fetch → load → DPO/SFT train, merge |
| `app/build/` | DPO/SFT JSONL builders (`router-build`) |
| `app/eval/` | Golden batch eval vs orchestrator (`router-eval`) |
| `app/tests/` | Pytest (CPU-only; no GPU) |
| `data/dpo/`, `data/sft/` | Cached JSONL per method (gitignored) |
| `data/golden-test/` | Gold CSVs + eval results |
| `data/output/dpo`, `data/output/sft` | Committed training JSONL |
| `checkpoints/` | Training output (gitignored) |
| `deploy/` | EC2 GPU bootstrap (`remote-deploy.sh`, `requirements-gpu.txt`) |

## Dataset

Training JSONL is pulled automatically on first run into `./data/{method}/`:

| Method | Dataset output |
|--------|----------------|
| **DPO** (default) | [data/output/dpo](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/dpo) |
| **SFT** | [data/output/sft](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/sft) |

Build locally with `python -m app.build {dpo,sft}` (auto-uses orchestrator venv when train venv lacks deps), or let training fetch JSONL from GitHub on first run.

## Router eval & datasets

Build JSONL for the **intent router LLM only** (`run_intent_rewrite_router` in layer-orchestrator-v1). Does not train RAG, GitHub MCP, or answer models. Shared gold logic: `app/build/gold.py`. Outputs: `data/output/dpo/`, `data/output/sft/`.

**Prerequisite:** sibling `layer-orchestrator-v1` (`ORCHESTRATOR_ROOT` if layout differs).

### Build DPO JSONL

From repo root (synthetic **rejected** if no eval results):

```bash
python -m app.build dpo
```

After golden eval, rebuild so **rejected** comes from real mismatches in `data/golden-test/result/*.csv`:

```bash
ROUTER_PROMPT_VERSION=router-v2.00 python -m app.eval
python -m app.build dpo
```

Live eval for **rejected** (no result CSV needed):

```bash
ORCHESTRATOR_URL=http://192.168.86.179:30184 \
  python -m app.build dpo --fetch-live --orchestrator-url "$ORCHESTRATOR_URL"
```

Include seed-FAQ / injection gold (normally skipped):

```bash
python -m app.build dpo --include-seed-faq --include-hack
```

Post-train regression check:

```bash
ROUTER_PROMPT_VERSION=router-v2.00 python -m app.eval
```

### DPO JSONL record shape

Each line matches what the router LLM sees in production:

```json
{
  "prompt": [
    {"role": "system", "content": "<app/prompts/router-v2.00.txt rendered>"},
    {"role": "user", "content": "History:\n(none)\n\nLatest question:\n..."}
  ],
  "chosen": "{\"rewritten_question\":\"...\",\"route\":\"rag_private_kb\",...}",
  "rejected": "{\"rewritten_question\":\"...\",\"route\":\"help\",...}",
  "meta": {
    "question": "...",
    "expected_route": "rag_private_kb",
    "source_file": "router_rag_private_kb.csv",
    "rejected_source": "result_csv | live_eval | synthetic",
    "router_prompt_version": "router-v2.00"
  }
}
```

- **chosen** — gold `expected_route` via `app.build.gold`
- **rejected** — eval mismatch from result CSV, live `/v1/orchestrator/eval/router`, or synthetic opposite route

### Gold CSV mapping (legacy → canonical)

| Gold `expected_route` | Chosen `route` |
|-----------------------|----------------|
| `rag` / `rag_private_kb` | `rag_private_kb` |
| `tool` / `github_search` / `web_search` | tool route (or `expected_tool` column) |
| `direct_reply` / `greeting` / `identity` / `help` / `capabilities` | matching static route |
| `clarify` | `clarify` |
| `reject` | `reject` |

Optional columns: `expected_tool`, `history_json`, `conversation_id`, `history`.

By default **skips** `internal/router_*.csv` (small-talk seed) and `router_reject.csv` (injection guard) — no router LLM in prod on those paths.

### CLI

| Command | Purpose |
|---------|---------|
| `python -m app.eval` or `router-eval` | Batch golden-test vs orchestrator |
| `python -m app.build dpo` or `router-build dpo` | Build DPO JSONL |
| `python -m app.build sft` or `router-build sft` | Build SFT JSONL |

### Build SFT JSONL

Gold completions only (no rejected pairs). Builder: `app/build/build_sft.py`.

```bash
python -m app.build sft
python -m app.build sft --include-seed-faq --include-hack
```

SFT record shape: `messages` with `system` / `user` / `assistant` (assistant content is router JSON). Assistant JSON matches DPO **chosen**.

See [data/golden-test/readme.md](data/golden-test/readme.md).

## Train on GPU node (16GB)

```bash
git clone <layer-router-train-v1-url> && cd layer-router-train-v1
python3.11 -m venv .venv && source .venv/bin/activate   # use 3.11 if `python3` is 3.12 without python3.12-venv
pip install -e ".[dev]"
cp .env.example .env   # TRAIN_METHOD, HF_TOKEN, etc.
```

### DPO (default)

```bash
python -m app.train.main
python -m app.train.main --method dpo
```

### SFT

```bash
python -m app.train.main --method sft
TRAIN_METHOD=sft python -m app.train.main
```

One command: **fetch** JSONL (if missing) → **load** / validate counts → **train**. Defaults: `Qwen/Qwen2.5-1.5B-Instruct`, `max-length=1024`, output under `checkpoints/router-{method}-qwen2.5-1.5b-<timestamp>/`.

After `pip install -e .`, run `layer-router-train` (alias: `layer-router-dpo`).

**OOM on 16GB:** lower `--max-length` / `--gradient-accumulation-steps` / `--num-train-epochs`.

### Deploy to EC2 GPU (AWS)

Push to `main` or run [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) manually.

**Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`

**Variables:** `DEPLOY_BUCKET`, `EC2_IAM_INSTANCE_PROFILE`, `EC2_INSTANCE_TYPE` (default `g5.xlarge`), `TRAIN_METHOD` (`dpo` or `sft`), `BASE_MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct`), `HF_REPO_FEATURE` (default `router`), `HF_REPO_VERSION` (default `0.00`; e.g. `0.01`, `1.00`), `HF_REPO_ID` (optional full override), `AUTO_TERMINATE_EC2` (default `true`)

The workflow ensures a GPU instance (tag `ec2-gpu-layer-router-train-v1`), syncs the repo to S3, and runs `deploy/remote-deploy.sh` via SSM (`python -m app.train.main`). After training, the LoRA adapter uploads to Hugging Face Hub.

### Flags (`python -m app.train.main`)

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

Do **not** set `HF_REPO_MODEL` — the Hub slug is derived from `BASE_MODEL`.

**GitHub Actions variables** (for EC2 deploy workflow): `TRAIN_METHOD`, `BASE_MODEL`, `HF_REPO_FEATURE`, `HF_REPO_VERSION`, `HF_REPO_ID`, `EC2_INSTANCE_TYPE`.

```bash
export HF_TOKEN=hf_...
TRAIN_METHOD=sft python -m app.train.main
# uploads to taixingbi/router-qwen2.5-1.5b-sft-0.00

BASE_MODEL=Qwen/Qwen2.5-7B-Instruct HF_REPO_VERSION=0.01 TRAIN_METHOD=sft python -m app.train.main
# uploads to taixingbi/router-qwen2.5-7b-sft-0.01
```

## Tests

```bash
pip install pytest
pytest app/tests/ -v -k "not production"
```

## See also

- [data/golden-test/readme.md](data/golden-test/readme.md) — golden batch eval
- [data/output/dpo](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/dpo) — DPO JSONL
- [data/output/sft](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/sft) — SFT JSONL
