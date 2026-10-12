# Scaling INFINITY HYPER 5.5: 100M → 1B → 7B

The code for every size is in this repository and tested. Only the hardware and the data are missing.

## Sizes (from `infinity size --config ...`)

| Config | Parameters | Training state (weights + Adam) | Compute-optimal tokens | GPU-hours (A100 80GB, 40% MFU) | Minimum GPUs |
|---|---|---|---|---|---|
| `hyper-5.5-small` | 97.5M | ~1.6 GB | ~2B | ~2.5 (T4: ~16) | 1 |
| `hyper-5.5-1b` | 1.03B | ~16.6 GB | ~21B | ~286 | 1 (8 for days, not weeks) |
| `hyper-5.5-7b` | **7.11B** | ~114 GB | ~142B (strong models: 1–2T) | ~13,500 (H100: ~4,300) | 3+ (realistically 32–64) |

Add `--price-per-gpu-hour <your provider's price>` to the size command to get costs. The tool never assumes prices.

## Architecture at 7B (`configs/hyper-5.5-7b.json`)

32 layers, d_model 4096, 32 query heads with **8 key/value heads (grouped-query attention)**,
SwiGLU d_ff 14336, RoPE theta 500k, 4,096-token context, 32k vocabulary, tied embeddings, gradient
checkpointing, bf16. Every layer is the same code as the tiny model (`infinity/core/model.py`).

## Step 1: 100M on Kaggle (free)

Open `notebooks/kaggle_train_100m.ipynb` in Kaggle and set the accelerator to GPU T4 x2 and internet on,
then choose Run All. The notebook:

1. Downloads licence-filtered code (`infinity data fetch-hf`).
2. Prepares it with sharded, parallel tokenisation (`infinity data prepare-large`).
3. Trains with fp16 loss scaling on T4 and DDP across both GPUs.
4. Stops at 11 hours with a checkpoint.
5. Packages the result for download.

Run it again in a new session to resume. The same cells are smoke-tested locally in `tests/test_scale.py`.

## Step 2–3: 1B / 7B on a GPU cluster

```bash
# data (on a CPU box or the cluster): billions of tokens, licence-filtered, deduplicated, sharded
infinity data fetch-hf --max-mb 200000 --out data/hf/code
infinity data prepare-large --manifest data/manifest.json --out runs/data-large --vocab 32000 --sample-mb 200

# one node, 8 GPUs
torchrun --nproc_per_node 8 -m infinity.core.dist_train --config configs/hyper-5.5-7b.json --data runs/data-large
# several nodes
torchrun --nnodes 8 --node_rank $RANK --nproc_per_node 8 --rdzv_endpoint $MASTER:29500 \
  -m infinity.core.dist_train --config configs/hyper-5.5-7b.json --data runs/data-large
```

- **FSDP2** (`fully_shard`) shards parameters, gradients and optimizer state, with one unit per block, bf16 parameters and fp32 reductions.
- **Checkpoints** are gathered to rank 0 in the normal verified INFINITY format, including optimizer state keyed by parameter name, so a run resumes even on a different number of GPUs.
- **Verified here:** 2-process DDP and FSDP2 runs on CPU (gloo) give identical losses, resume correctly, and produce checkpoints the inference engine loads.
- **Not verified here:** NCCL multi-GPU and multi-node runs, and real throughput. There is no GPU in the development container.

## Then make it an agent

After pretraining, run the tool-use pipeline on the new checkpoint (`infinity sft generate/train`,
`infinity bench run`). See `docs/FINETUNE_REPORT.md`. The 5.8M model learned the action format but could not copy
unseen names. That is the capability more parameters and data are expected to unlock. It has not been demonstrated yet.

## Known limits

- **Start-up memory:** each rank builds the full model before sharding. A 7B start-up needs about 14 GB (bf16) of free memory per GPU. Meta-device initialisation is a planned improvement.
- **Tokenisation speed:** BPE encoding is pure Python, parallelised across processes. It's fine for tens of GB. At TB scale, budget hours of CPU time, or add a compiled tokenizer.
- **Training data:** you must supply the data and review its licences. `fetch-hf` filters by the per-file licence labels a dataset provides.
