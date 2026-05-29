# layer-router-dpo-v1
# layer-router-dpo-v1

**QLoRA DPO training** for the HuntAI intent router (offline batch job on LAN GPU nodes). Consumes JSONL built by [`layer-orchestrator-v1/dpo-router`](../layer-orchestrator-v1/dpo-router/README.md); does not run as a cluster service.

**Base model:** `Qwen/Qwen2.5-7B-Instruct` (same as orchestrator `LLM_MODEL` / vLLM).

## Layout

| Path | Role |
|------|------|
| `scripts/load_jsonl.py` | JSONL → TRL `prompt` / `chosen` / `rejected` |
| `scripts/train_dpo.py` | QLoRA + `DPOTrainer` |
| `scripts/export_merge.py` | Optional merge for full-weight vLLM deploy |
| `run-train.sh` | Training wrapper |
| `run-export-merge.sh` | Merge wrapper |
| `checkpoints/` | Training output (gitignored) |

## Pipeline

```bash
# 1) Dataset (layer-orchestrator-v1)
cd layer-orchestrator-v1
ROUTER_PROMPT_VERSION=router-v2.00 bash gold-test/run-router-eval.sh   # optional
PYTHON=./venv/bin/python bash dpo-router/run-build-dpo.sh

# 2) Train (this repo, on GPU node 173 or 176)
cd ../layer-router-dpo-v1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run-train.sh
```

Default JSONL paths (sibling checkout):

- `TRAIN_JSONL` → `../layer-orchestrator-v1/dpo-router/output/train.jsonl`
- `VAL_JSONL` → `../layer-orchestrator-v1/dpo-router/output/val.jsonl`

Override with `ORCHESTRATOR_DPO_DIR` or explicit `TRAIN_JSONL` / `VAL_JSONL`.

## GPU node (16GB)

1. **Sync** `layer-router-dpo-v1/` and `layer-orchestrator-v1/dpo-router/output/*.jsonl` to the node.
2. **Free GPU** — do not train while vLLM uses the same card (~70% VRAM). Train on `173` while inference runs on `176`, or scale vLLM down on the train host.
3. **Run** `bash run-train.sh` → `checkpoints/router-dpo-*/adapter/`.

See [deploy-vllm-inference.md](../huntai-k3s/docs/deploy-vllm-inference.md) for LoRA serving.

### Environment

| Variable | Default |
|----------|---------|
| `ORCHESTRATOR_DPO_DIR` | `../layer-orchestrator-v1/dpo-router` |
| `TRAIN_JSONL` / `VAL_JSONL` | `{ORCHESTRATOR_DPO_DIR}/output/*.jsonl` |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| `OUTPUT_DIR` | `./checkpoints/router-dpo-YYYYMMDD-HHMM` |
| `NUM_TRAIN_EPOCHS` | `2` |
| `GRAD_ACCUM` | `8` |
| `LORA_R` | `32` |

**OOM:** `MAX_LENGTH=1536`, `GRAD_ACCUM=4`, `NUM_TRAIN_EPOCHS=1`.

## Deploy and verify

1. **LoRA (recommended):** copy `adapter/` to e.g. `/data/models/router-dpo-v1`, enable vLLM `--enable-lora` / `--lora-modules`.
2. **Merged:** `bash run-export-merge.sh --adapter-dir checkpoints/.../adapter --output-dir /data/models/merged`
3. **Test:** `router_model` on `POST /v1/orchestrator/eval/router`, then `gold-test/run-router-eval.sh`.
4. **Promote:** set `LLM_MODEL` when gold match rate improves.

## Tests

```bash
pip install pytest
pytest tests/ -v -k "not production"
```

## See also

- [layer-orchestrator-v1/dpo-router](../layer-orchestrator-v1/dpo-router/README.md) — dataset build
- [gold-test](../layer-orchestrator-v1/gold-test/readme.md) — router eval
