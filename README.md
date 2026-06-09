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
| `data/dpo/`, `data/sft/` | Cached JSONL per method (from fetch or train) |
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

**Gold CSVs:** `app.build` and `app.eval` auto-fetch from [layer-orchestrator-v1 `router-eval/golden-test/data`](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/golden-test/data) when `data/golden-test/data/` is empty. Local copy or sibling checkout is used if present. Override with `ORCHESTRATOR_GOLD_REPO`, `ORCHESTRATOR_REF`, `ORCHESTRATOR_GOLD_SUBDIR`.

**Orchestrator code (build only):** sibling `layer-orchestrator-v1` for router prompts and gold logic (`ORCHESTRATOR_ROOT` if layout differs).

### Golden batch eval

For each row in `data/golden-test/data/**/*.csv`, calls `POST /v1/orchestrator/eval/router`, writes per-suite results under `data/golden-test/result/` (flat basename, e.g. `router_greeting.csv`), then builds `result/router-eval-report-<ROUTER_PROMPT_VERSION>.md`.

**Requirements:** `pip install -e ".[dev]"` (stdlib HTTP; no curl/jq) and a running orchestrator at `ORCHESTRATOR_URL` (default `http://192.168.86.179:30184`).

| Path | Role |
|------|------|
| `data/golden-test/data/tools/*.csv` | Tool-route gold (`rag_private_kb`, `web_search`, …) |
| `data/golden-test/data/internal/*.csv` | Internal-intent gold (`greeting`, `identity`, `help`, …) |
| Header | `question,expected_route` (required). Optional: `conversation_id`, `history` (JSON `{role, content}` array) |
| Threading | Default `conversation_id` per file: `conv-gold-<basename>`; `X-Session-Id: ses-gold-<basename>` |
| `result/<name>.csv` | Per input basename: `question`, `expected_route`, `actual_route`, `route_match`, `rewritten_question`, `actual_answer` |
| `result/router-eval-report-<version>.md` | Summary, match rate, bad items (`route_match` = false) |

```bash
python -m app.eval

CONCURRENCY=20 ROUTER_PROMPT_VERSION=router-v2.00 python -m app.eval

# Score a trained LoRA (separate result dir per adapter)
ROUTER_MODEL=router-qwen2.5-7b-sft-v1.00 \
  python -m app.eval \
  --result-dir data/golden-test/result/sft-v1.00 \
  --router-prompt-version router-v2.00
```

**Progress** (stderr) — one line per gold file, then match-rate table on stdout. Full report: `data/golden-test/result/router-eval-report-<ROUTER_PROMPT_VERSION>.md`. Generated `result/*.csv` and reports are gitignored under `data/golden-test/.gitignore`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_DIR` | `data/golden-test/data` | Input `*.csv` directory |
| `RESULT_DIR` | `data/golden-test/result` | Output directory |
| `ORCHESTRATOR_URL` | `http://192.168.86.179:30184` | Orchestrator base URL |
| `CONCURRENCY` | `4` | Parallel HTTP requests per file |
| `ROUTER_PROMPT_VERSION` | `router-v2.00` | `router_prompt_version` on each eval request |
| `ROUTER_MODEL` | _(unset)_ | Optional vLLM model / LoRA id; sent as `router_model` |
| `REPORT_PATH` | `…/router-eval-report-<prompt>[-<model>].md` | Markdown report path |

Eval responses include `decision.route_detail` alongside legacy `decision.route`. Optional CSV columns: `expected_route_detail_type`, `expected_tool_name`.

**Gold suites** (`router_<suite>.csv`):

| File | Focus |
|------|--------|
| `data/golden-test/data/internal/router_greeting.csv` | `greeting` — smalltalk seed |
| `data/golden-test/data/internal/router_identity.csv` | `identity` |
| `data/golden-test/data/internal/router_capabilities.csv` | `capabilities` |
| `data/golden-test/data/internal/router_help.csv` | `help` |
| `data/golden-test/data/internal/router_reject.csv` | Injection guard → `reject` |
| `data/golden-test/data/tools/router_rag_private_kb.csv` | Candidate / profile (`rag_private_kb`) |
| `data/golden-test/data/tools/router_github.csv` | HuntAI repos (`github_search`) |
| `data/golden-test/data/tools/router_web_search.csv` | Public web (`web_search`) |

**Small-talk seed (not RAG):** `layer-orchestrator-v1/app/prompts/seed_intents/*.json` — on empty history, normalized question matched exactly to `user_examples` → `direct_reply` with seed `answer` (no router LLM). `__CANDIDATE_NAME__` replaced at runtime.

### CLI

| Command | Purpose |
|---------|---------|
| `python -m app.eval` or `router-eval` | Batch golden-test vs orchestrator |
| `python -m app.build dpo` or `router-build dpo` | Build DPO JSONL |
| `python -m app.build sft` or `router-build sft` | Build SFT JSONL |

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

### Build SFT JSONL

Gold completions only (no rejected pairs). Builder: `app/build/build_sft.py`.

```bash
python -m app.build sft
python -m app.build sft --include-seed-faq --include-hack
```

SFT record shape: `messages` with `system` / `user` / `assistant` (assistant content is router JSON). Assistant JSON matches DPO **chosen**.

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

- [data/output/dpo](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/dpo) — DPO JSONL
- [data/output/sft](https://github.com/taixingbi/layer-router-train-v1/tree/main/data/output/sft) — SFT JSONL
- [layer-orchestrator-v1 golden-test data](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/router-eval/golden-test/data) — upstream gold CSVs
