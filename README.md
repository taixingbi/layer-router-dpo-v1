# layer-router-dpo-v1

**QLoRA DPO training** for the HuntAI intent router (offline batch job on LAN GPU nodes). Consumes JSONL from [layer-orchestrator-v1 `dpo-router/output`](https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router/output); does not run as a cluster service.

**Base model:** `Qwen/Qwen2.5-7B-Instruct` (same as orchestrator `LLM_MODEL` / vLLM).

## Layout

| Path | Role |
|------|------|
| `scripts/load_jsonl.py` | JSONL → TRL `prompt` / `chosen` / `rejected` |
| `scripts/train_dpo.py` | QLoRA + `DPOTrainer` |
| `scripts/export_merge.py` | Optional merge for full-weight vLLM deploy |
| `fetch-dataset.sh` | Download `train.jsonl` / `val.jsonl` from GitHub (GPU node) |
| `run-train.sh` | Training wrapper |
| `run-export-merge.sh` | Merge wrapper |
| `data/` | Downloaded JSONL (gitignored; default train input) |
| `checkpoints/` | Training output (gitignored) |

## Load dataset (different machine than orchestrator)

Training JSONL is **committed** on `main` here:

**https://github.com/taixingbi/layer-orchestrator-v1/tree/main/dpo-router/output**

Files: `train.jsonl`, `val.jsonl`, `build-stats.json`.

On a GPU node that only has `layer-router-dpo-v1` (no sibling `layer-orchestrator-v1` checkout):

```bash
cd layer-router-dpo-v1
bash fetch-dataset.sh
# -> data/train.jsonl, data/val.jsonl, data/build-stats.json
bash run-train.sh
```

`run-train.sh` picks paths in order:

1. `TRAIN_JSONL` / `VAL_JSONL` if set
2. `./data/*.jsonl` (after `fetch-dataset.sh`)
3. `../layer-orchestrator-v1/dpo-router/output/*.jsonl` (monorepo sibling)

### Manual download

```bash
mkdir -p data
REF=main
BASE=https://raw.githubusercontent.com/taixingbi/layer-orchestrator-v1/${REF}/dpo-router/output
curl -fsSL "$BASE/train.jsonl" -o data/train.jsonl
curl -fsSL "$BASE/val.jsonl"   -o data/val.jsonl
curl -fsSL "$BASE/build-stats.json" -o data/build-stats.json
```

Pin another branch or tag: `ORCHESTRATOR_DPO_REF=router-dpo-v2 bash fetch-dataset.sh`

### Rebuild dataset (orchestrator machine)

When gold or eval results change, rebuild and push from a host with `layer-orchestrator-v1`:

```bash
cd layer-orchestrator-v1
ROUTER_PROMPT_VERSION=router-v2.00 bash gold-test/run-router-eval.sh   # optional
PYTHON=./venv/bin/python bash dpo-router/run-build-dpo.sh
git add dpo-router/output && git commit -m "Update router DPO dataset" && git push
```

Then on the GPU node: `bash fetch-dataset.sh` again.

## Train on GPU node (16GB)

```bash
git clone <layer-router-dpo-v1-url> && cd layer-router-dpo-v1
bash fetch-dataset.sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run-train.sh
```

1. **Free GPU** — do not train on the same card as vLLM at ~70% VRAM. Train on `173` while inference stays on `176`, or scale vLLM down on the train host.
2. Output: `checkpoints/router-dpo-*/adapter/`

See [deploy-vllm-inference.md](../huntai-k3s/docs/deploy-vllm-inference.md) for LoRA serving.

### Environment

| Variable | Default |
|----------|---------|
| `TRAIN_JSONL` / `VAL_JSONL` | `./data/*.jsonl` or sibling orchestrator path |
| `ORCHESTRATOR_DPO_DIR` | `../layer-orchestrator-v1/dpo-router` (local build only) |
| `ORCHESTRATOR_DPO_REF` | `main` (for `fetch-dataset.sh`) |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| `OUTPUT_DIR` | `./checkpoints/router-dpo-YYYYMMDD-HHMM` |
| `NUM_TRAIN_EPOCHS` | `2` |
| `GRAD_ACCUM` | `8` |
| `LORA_R` | `32` |

**OOM:** `MAX_LENGTH=1536`, `GRAD_ACCUM=4`, `NUM_TRAIN_EPOCHS=1`.

## Deploy and verify

1. **LoRA (recommended):** copy `adapter/` to e.g. `/data/models/router-dpo-v1`, enable vLLM `--enable-lora` / `--lora-modules`.
2. **Merged:** `bash run-export-merge.sh --adapter-dir checkpoints/.../adapter --output-dir /data/models/merged`
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
