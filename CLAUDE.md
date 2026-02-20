# subspec_v2

Speculative decoding research codebase. Supports multiple draft strategies (Classic SD, SubSpec, EAGLE) over FlashInfer-backed paged KV caches. The beam_engine beam search is being ported in as a new draft strategy (`be_classic_sd_fi`).

## Project Structure

```
run/
  main.py                        # Unified CLI entrypoint (--config <yaml> <subcommand>)
  core/presets.py                # Method registry — maps method names to generator + draft_model classes
specdecodes/
  models/
    draft_models/
      classic_sd_fi.py           # Classic SD draft model (FlashInfer, tree-style)
      be_classic_sd_fi.py        # beam_engine beam search port — COW paged KV + trie dedup
      subspec_sd_fi.py           # SubSpec draft model (FlashInfer)
    generators/
      classic_sd_fi.py           # Generator for classic_sd_fi
    utils/
      cpu_tree.py                # Tree / TreeNode — CPU-side draft tree structure
      flashinfer/
        cache_manager.py         # KvCachePool, RequestKvCache, KvCacheBatchPosition
        attention_wrapper.py     # FlashinferAttentionWrapper
configs/
  methods/                       # Per-method YAML templates
  exp_offloading/                # Full experiment configs (offloading variants)
```

## Running

```bash
# Quick sanity check
python -m run.main --config configs/methods/classic_sd_fi.yaml run-test

# Benchmark
python -m run.main --config configs/methods/classic_sd_fi.yaml run-benchmark --benchmarks mt-bench --max-samples 20

# Override device / warmup
python -m run.main --config configs/methods/classic_sd_fi.yaml --device cuda:1 --warmup-iter 0 run-test
```

Key YAML fields:
```yaml
method: classic_sd_fi          # selects generator + draft_model from presets.py
draft_params:
  temperature: 0.2
  max_depth: 32
  topk_len: 6                  # K (beam width / topk fan-out)
compile_mode: max-autotune-no-cudagraphs  # required for flashinfer
```

## Remote Testing

- Host: `brain_l@140.113.24.210`
- Conda env: `subspec`; project root on remote: `~/flashtree/base/subspec_v2`
- Check free GPUs first: `ssh brain_l@140.113.24.210 "bash -i -c nvidia-smi"`
- Run: `ssh brain_l@140.113.24.210 "bash -i -c 'conda activate subspec && cd ~/flashtree/base/subspec_v2 && git pull && CUDA_VISIBLE_DEVICES=<gpu_id> python -m run.main --config configs/methods/<method>.yaml run-test'"`

## Registering a New Method

Add an entry in `run/core/presets.py`:
```python
ModelRegistry.register(
    name="be_classic_sd_fi",
    generator_cls="specdecodes.models.generators.classic_sd_fi:ClassicSDGenerator",
    draft_model_cls="specdecodes.models.draft_models.be_classic_sd_fi:ClassicSDDraftModel",
    default_config={...},
    load_draft_model_fn=flashinfer_load_draft_model,
    load_kv_cache_fn=flashinfer_load_kv_cache,
)
```

## beam_engine Integration (`be_classic_sd_fi.py`)

This is the active development area. The file lives at `specdecodes/models/draft_models/be_classic_sd_fi.py` and replaces the original `speculate()` with beam_engine's COW paged KV beam search while keeping all other methods (`forward`, `init_cuda_graph_runner`, `postspec`, etc.) intact.

### What changed vs `classic_sd_fi.py`

| | `classic_sd_fi.py` | `be_classic_sd_fi.py` |
|---|---|---|
| Tree construction | All K tokens per depth level written into shared KV; tree mask handles attention | K independent KV histories via COW page sharing |
| Forward per step | 1 call, batch=[1, K] tokens with tree attention mask | 1 call, batch=[K, 1] tokens, no mask needed |
| KV writes | All in `request_kv_cache`; rolled back via `decrement()` | Separate page allocations; prompt pages shared read-only |
| postspec | Enabled (extends tree depth) | Disabled (`postspec_count = max_depth`) |

### Key data structures

**`KvCachePool`** (`cache_manager.py`)
- `cache_data`: `[num_layers, max_pages, 2, page_len, num_heads, head_dim]` (single 6D tensor)
- `allocate(n) -> list[int]` — returns n free page indices
- `deallocate(list[int])` — marks pages free

**`RequestKvCache`** (`cache_manager.py`)
- `kv_page_indices`: Python `list[int]` — ordered page indices for the request's KV sequence
- `get_seq_length() -> int` — current token count
- `increment(n)` — allocates pages as needed and advances `kv_len`

**`KvCacheBatchPosition`** (`cache_manager.py`)
- Describes a batch of sequences for FlashInfer kernels
- `positions`: **absolute** token positions (not within-page offsets) — FlashInfer derives page offset internally
- `kv_last_page_len`: must include the token currently being written (`current_pos % page_size + 1`)

**`Tree` / `TreeNode`** (`utils/cpu_tree.py`)
- `Tree(root_token_id, dtype)` — root at depth 0
- `tree.nodes`: plain Python list; `tree.current_size`: int
- `tree.find_child_index(node_idx, token_id) -> int` — returns child index or -1 (used for trie dedup)
- Direct manipulation: `tree.nodes.append(tn); tree.current_size += 1` (bypasses `add_nodes` which has no dedup support)

### COW + ref counting invariant

`node_pages[node_idx]` = ordered list of physical page indices covering the full KV path to `node_idx`.
`page_ref_counts[page]` = number of distinct `node_pages` lists that contain `page`.

- Prompt pages are shared read-only across all initial beams (ref = K).
- Before each decode write, iterate `set(beam_node)` (unique nodes only):
  - `off == 0`: allocate a fresh page (no copy needed).
  - `off > 0` and `ref > 1`: `_copy_block(pool, page, off)` — allocate + copy first `off` slots; decrement old ref; redirect `node_pages[node_idx][pli]` to new page.
- On new node creation: `node_pages[new] = list(node_pages[parent])` + increment all refs.
- On old-node release: decrement all refs; deallocate pages that hit 0.
- Final cleanup: collect unique non-prompt pages across all remaining `node_pages` entries (use a `set()` to avoid double-free) and deallocate.

### Module-level helpers

```python
_copy_block(kvCachePool, src_page, off) -> int
# Allocates a new page, copies cache_data[:, src_page, :, :off] into it. Returns new page index.

_build_beam_batch_position(beam_pages_list, current_pos, page_size, device) -> KvCacheBatchPosition
# Builds KvCacheBatchPosition for K decode beams. positions = [current_pos]*K (absolute).
```
