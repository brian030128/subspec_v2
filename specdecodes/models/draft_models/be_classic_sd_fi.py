import torch
import nvtx

from ..utils.cpu_tree import Tree, TreeNode
from .base import DraftModelBase, TreeData, TreeMaskCache

from ..utils.flashinfer.cache_manager import (
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.be_attention_wrapper import BeFlashinferWrapper


def _copy_block(kvCachePool, src_page, off):
    """COW: allocate a fresh page and copy the first `off` token slots from src_page."""
    new_page = kvCachePool.allocate(1)[0]
    # cache_data shape: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
    kvCachePool.cache_data[:, new_page, :, :off] = kvCachePool.cache_data[:, src_page, :, :off]
    return new_page


def _build_beam_batch_position(beam_pages_list, current_pos, page_size, device):
    """
    Build a KvCacheBatchPosition for K beams each writing one token at current_pos.

    beam_pages_list: list of K page lists (one per beam).
    positions uses absolute sequence position (subspec_v2 convention).
    """
    K = len(beam_pages_list)
    kv_page_indices = []
    kv_page_indptr = [0]
    for pages in beam_pages_list:
        kv_page_indices.extend(pages)
        kv_page_indptr.append(len(kv_page_indices))

    kv_last_page_len_val = current_pos % page_size + 1  # includes slot being written

    return KvCacheBatchPosition(
        seq_indptr=torch.arange(K + 1, dtype=torch.int32, device=device),
        kv_page_indptr=torch.tensor(kv_page_indptr, dtype=torch.int32, device=device),
        kv_page_indices=torch.tensor(kv_page_indices, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([kv_last_page_len_val] * K, dtype=torch.int32, device=device),
        batch_indices=torch.arange(K, dtype=torch.int32, device=device),
        positions=torch.tensor([current_pos] * K, dtype=torch.int32, device=device),
    )


class ClassicSDDraftModel(DraftModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.had_first_speculate = False
        self.postspec_count = 0

    def init_cuda_graph_runner(self, device: torch.device):
        """Capture a CUDA graph for the beam decode forward (K beams, 1 token each)."""
        print("be_classic_sd_fi: Initializing CUDA Graph runner for beam decode...")
        if hasattr(self, "beam_graph"):
            return

        self.model.eval()
        K = self.draft_params.topk_len
        kvCachePool = self.kvCachePool
        max_num_pages = kvCachePool.max_pages
        PAGE_SIZE = kvCachePool.page_len
        dtype = kvCachePool.cache_data[0].dtype

        # ── Model input staging buffers ──
        self.beam_input_ids_buf    = torch.zeros((K, 1), dtype=torch.long, device=device)
        self.beam_position_ids_buf = torch.zeros((K, 1), dtype=torch.long, device=device)

        # ── Batch position staging buffers ──
        self.beam_kv_page_indptr_buf    = torch.zeros(K + 1, dtype=torch.int32, device=device)
        self.beam_kv_page_indices_buf   = torch.zeros(max_num_pages, dtype=torch.int32, device=device)
        self.beam_kv_last_page_len_buf  = torch.zeros(K, dtype=torch.int32, device=device)
        self.beam_batch_indices_buf     = torch.arange(K, dtype=torch.int32, device=device)
        self.beam_positions_buf         = torch.zeros(K, dtype=torch.int32, device=device)

        # ── Persistent KvCacheBatchPosition wrapping staging buffers ──
        self.beam_batch_position = KvCacheBatchPosition(
            seq_indptr       = torch.arange(K + 1, dtype=torch.int32, device=device),
            kv_page_indptr   = self.beam_kv_page_indptr_buf,
            kv_page_indices  = self.beam_kv_page_indices_buf,
            kv_last_page_len = self.beam_kv_last_page_len_buf,
            batch_indices    = self.beam_batch_indices_buf,
            positions        = self.beam_positions_buf,
        )

        # ── Reinit decode wrapper with CUDA graph support ──
        self.flashinferWrapper.init_cuda_graph_decode(K, max_num_pages, device)

        # ── Warmup + Capture ──
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.flashinferWrapper.prepareAttention(
                    'decode', self.beam_batch_position, PAGE_SIZE, "NONE", dtype)
                _ = self(
                    self.beam_input_ids_buf, with_softmax=False,
                    position_ids=self.beam_position_ids_buf,
                    kvCachePool=kvCachePool, batch_position=self.beam_batch_position,
                    mode='decode', flashinferWrapper=self.flashinferWrapper)

            torch.cuda.current_stream().wait_stream(stream)
            self.flashinferWrapper.prepareAttention(
                'decode', self.beam_batch_position, PAGE_SIZE, "NONE", dtype)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=stream):
                self.beam_output_buffer = self(
                    self.beam_input_ids_buf, with_softmax=False,
                    position_ids=self.beam_position_ids_buf,
                    kvCachePool=kvCachePool, batch_position=self.beam_batch_position,
                    mode='decode', flashinferWrapper=self.flashinferWrapper)

        self.beam_graph = cg
        print("be_classic_sd_fi: Finished capturing beam decode CUDA graph")

    def beam_decode_step(self, beam_input_ids, beam_position_ids, batch_position):
        """Copy data into staging buffers, re-plan, and replay the captured graph."""
        # Copy model inputs
        self.beam_input_ids_buf.copy_(beam_input_ids)
        self.beam_position_ids_buf.copy_(beam_position_ids)

        # Copy batch position
        self.beam_kv_page_indptr_buf.copy_(batch_position.kv_page_indptr)
        n_indices = batch_position.kv_page_indptr[-1].item()
        self.beam_kv_page_indices_buf[:n_indices].copy_(batch_position.kv_page_indices[:n_indices])
        self.beam_kv_last_page_len_buf.copy_(batch_position.kv_last_page_len)
        self.beam_positions_buf.copy_(batch_position.positions)

        # Re-plan (outside graph)
        self.flashinferWrapper.prepareAttention(
            'decode', self.beam_batch_position,
            self.kvCachePool.page_len, "NONE",
            self.kvCachePool.cache_data[0].dtype)

        # Replay
        self.beam_graph.replay()
        return self.beam_output_buffer

    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)

        return logits

    @torch.no_grad()
    def update_tree(self, tree_data):
        with nvtx.annotate("tree_finalize"):
            with nvtx.annotate("tree_data/get"):
                data = tree_data.get_data()
            with nvtx.annotate("tree/apply"):
                self.tree.add_nodes(*data)
        return self.tree

    @torch.no_grad()
    def speculate(self, input_ids, request_kv_cache, **kwargs) -> Tree:
        self.had_first_speculate = True

        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        self.request_kv_cache = request_kv_cache

        if not hasattr(self, 'flashinferWrapper'):
            self.flashinferWrapper = BeFlashinferWrapper(
                self.model.config.num_attention_heads,
                self.model.config.num_key_value_heads,
                self.model.config.hidden_size,
                request_kv_cache.kvCachePool.page_len,
            )
        self.kvCachePool = request_kv_cache.kvCachePool

        K = self.draft_params.topk_len
        max_depth = self.draft_params.max_depth
        kvCachePool = request_kv_cache.kvCachePool
        PAGE_SIZE = kvCachePool.page_len

        # --- kv_len init ---
        kv_len = request_kv_cache.get_seq_length()
        if isinstance(kv_len, torch.Tensor):
            kv_len = kv_len.item()

        # --- Prefill (identical to original) ---
        request_kv_cache.increment(input_len)
        batch_position = getKvCacheBatchPosition(
            request_kv_caches=[request_kv_cache],
            mode='tree',
            device=device,
            treeTokens=input_len,
        )
        self.flashinferWrapper.prepareAttention(
            'prefill',
            batch_position,
            PAGE_SIZE,
            "NONE",
            kvCachePool.cache_data[0].dtype,
        )
        position_ids = torch.arange(kv_len, kv_len + input_len, dtype=torch.long, device=device).unsqueeze(0)
        sampled_probs = self(
            input_ids,
            with_softmax=True,
            logits_to_keep=1,
            position_ids=position_ids,
            kvCachePool=kvCachePool,
            batch_position=batch_position,
            mode='prefill',
            flashinferWrapper=self.flashinferWrapper,
        )
        kv_len += input_len
        org_kv_len = kv_len

        # --- Init tree (root = last input token, same as original) ---
        tree = Tree(input_ids[0, -1], dtype)
        prompt_pages = list(request_kv_cache.kv_page_indices)  # Python list copy

        # --- Get K initial tokens from prefill output ---
        first_probs = sampled_probs[0, -1, :]        # [vocab]
        topk_probs, topk_ids = first_probs.topk(K)   # [K]

        # --- Init trie state ---
        # node_pages[node_idx]   = list[int] of physical pages for full KV path to this node
        # page_ref_counts[page]  = number of node_pages lists containing this page
        node_pages = {}       # int -> list[int]
        page_ref_counts = {}  # int -> int

        # Add K depth-1 nodes to tree, all children of root (node 0).
        # Top-K tokens are distinct, so no dedup needed here.
        beam_node = []  # beam_node[k] = tree node index for beam k
        for i in range(K):
            tok = topk_ids[i].item()
            prob = topk_probs[i].item()
            new_idx = tree.current_size
            tn = TreeNode(parent=0, token_id=tok, cumulative_probability=prob, depth=1)
            tree.nodes[0].children.append(new_idx)
            tree.nodes.append(tn)
            tree.current_size += 1
            # Inherit prompt pages; track ref counts
            node_pages[new_idx] = list(prompt_pages)
            for p in prompt_pages:
                page_ref_counts[p] = page_ref_counts.get(p, 0) + 1
            beam_node.append(new_idx)

        tree.available_leaves = list(beam_node)
        cum_log_probs = torch.log(topk_probs).tolist()

        # --- Decode loop ---
        current_pos = kv_len  # absolute position being written this step
        for step in range(max_depth - 1):
            off = current_pos % PAGE_SIZE   # within-page offset being written
            pli = current_pos // PAGE_SIZE  # page list index for this position

            # COW enforcement: one pass over unique live nodes.
            # If two beams share the same trie node, process it only once.
            for node_idx in set(beam_node):
                if off == 0:
                    # Starting a fresh page — allocate one
                    new_page = kvCachePool.allocate(1)[0]
                    node_pages[node_idx].append(new_page)
                    page_ref_counts[new_page] = 1
                else:
                    write_page = node_pages[node_idx][pli]
                    if page_ref_counts[write_page] > 1:
                        # Shared page — copy before writing (COW)
                        new_page = _copy_block(kvCachePool, write_page, off)
                        page_ref_counts[write_page] -= 1
                        if page_ref_counts[write_page] == 0:
                            kvCachePool.deallocate([write_page])
                            del page_ref_counts[write_page]
                        node_pages[node_idx][pli] = new_page
                        page_ref_counts[new_page] = 1

            # Build KvCacheBatchPosition: K beams (may share node_pages if same trie node)
            beam_pages_list = [node_pages[beam_node[k]] for k in range(K)]
            batch_position = _build_beam_batch_position(beam_pages_list, current_pos, PAGE_SIZE, device)

            # Batched decode forward: [K, 1] input
            beam_input_ids = torch.tensor(
                [[tree.nodes[beam_node[k]].token_id] for k in range(K)],
                dtype=torch.long, device=device,
            )
            beam_position_ids = torch.full((K, 1), current_pos, dtype=torch.long, device=device)

            if hasattr(self, 'beam_graph'):
                logits = self.beam_decode_step(beam_input_ids, beam_position_ids, batch_position)
            else:
                self.flashinferWrapper.prepareAttention(
                    'decode',
                    batch_position,
                    PAGE_SIZE,
                    "NONE",
                    kvCachePool.cache_data[0].dtype,
                )
                logits = self(
                    beam_input_ids,
                    with_softmax=False,
                    position_ids=beam_position_ids,
                    kvCachePool=kvCachePool,
                    batch_position=batch_position,
                    mode='decode',
                    flashinferWrapper=self.flashinferWrapper,
                )  # [K, 1, vocab]

            # Score and select top-K next tokens across all beams
            probs = torch.softmax(logits[:, -1, :] / self.draft_params.temperature, dim=-1)  # [K, vocab]
            cum = torch.tensor(cum_log_probs, device=device)
            flat_scores = (cum[:, None] + torch.log(probs + 1e-10)).reshape(-1)
            topk_scores, topk_flat_ids = flat_scores.topk(K)
            vocab_size = probs.shape[-1]
            parent_list = (topk_flat_ids // vocab_size).tolist()
            new_tok_list = (topk_flat_ids % vocab_size).tolist()

            # Build new trie nodes with dedup
            seen_pairs = {}   # (parent_node_idx, token_id) -> new tree node idx
            new_beam_node = []
            for i in range(K):
                parent_node = beam_node[parent_list[i]]
                tok = new_tok_list[i]
                key = (parent_node, tok)
                step_prob = probs[parent_list[i], tok].item()

                if key in seen_pairs:
                    # Trie dedup: reuse existing new node
                    new_node = seen_pairs[key]
                else:
                    existing = tree.find_child_index(parent_node, tok)
                    if existing != -1 and existing in node_pages:
                        # Already in trie (rare edge case)
                        new_node = existing
                    else:
                        # Create new tree node
                        new_node = tree.current_size
                        tn = TreeNode(
                            parent=parent_node,
                            token_id=tok,
                            cumulative_probability=step_prob,
                            depth=tree.nodes[parent_node].depth + 1,
                        )
                        tree.nodes[parent_node].children.append(new_node)
                        tree.nodes.append(tn)
                        tree.current_size += 1
                        # Inherit pages from parent node (ref counts updated below)
                        node_pages[new_node] = list(node_pages[parent_node])
                        for p in node_pages[new_node]:
                            page_ref_counts[p] += 1
                    seen_pairs[key] = new_node
                new_beam_node.append(new_node)

            # Release old live nodes — decrement their page refs
            old_unique = set(beam_node)
            for old_node in old_unique:
                pages_to_free = []
                for p in node_pages[old_node]:
                    page_ref_counts[p] -= 1
                    if page_ref_counts[p] == 0:
                        pages_to_free.append(p)
                        del page_ref_counts[p]
                for p in pages_to_free:
                    kvCachePool.deallocate([p])
                del node_pages[old_node]

            tree.available_leaves = list(set(new_beam_node))
            beam_node = new_beam_node
            cum_log_probs = topk_scores.tolist()
            current_pos += 1

        # --- Cleanup: free all beam-allocated (non-prompt) pages ---
        prompt_pages_set = set(prompt_pages)
        pages_to_free = set()
        for pages in node_pages.values():
            for p in pages:
                if p not in prompt_pages_set:
                    pages_to_free.add(p)
        for p in pages_to_free:
            kvCachePool.deallocate([p])

        # request_kv_cache stays at org_kv_len; beam pages were separate allocations
        # Disable postspec (beam search returns complete tree upfront)
        self.postspec_count = max_depth

        # Save state for postspec interface
        self.tree = tree
        self.tree_data = TreeData()
        self.tree_mask_cache = TreeMaskCache(
            prefix_len=current_pos + 1,
            sample_len=K,
            max_cache_len=None,
            dtype=dtype,
            device=device,
        )

        # Last beam state (for speculate_once to continue from)
        self.token_ids = torch.tensor(
            [[tree.nodes[n].token_id for n in beam_node]],
            dtype=torch.long, device=device,
        )
        self.position_ids = torch.full(
            (1, K), current_pos, dtype=torch.long, device=device,
        )
        self.parent_probs = torch.tensor(
            [cum_log_probs], dtype=dtype, device=device,
        ).exp()

        return tree

    def init_postspec(self):
        self.tree_data = TreeData()
        self.postspec_count = 0

    @torch.no_grad()
    def postspec(self):
        if not self.had_first_speculate:
            return
        if self.postspec_count > (self.draft_params.max_depth - 1):
            return
        with nvtx.annotate("postspec_step", color="blue"):
            self.speculate_once()
        self.postspec_count += 1

    @torch.no_grad()
    def speculate_once(self, **kwargs):
        tree_attention_mask = self.tree_mask_cache.get_tree_mask()
        token_ids = self.token_ids
        parent_probs = self.parent_probs
        position_ids = self.position_ids

        request_kv_cache = self.request_kv_cache

        with nvtx.annotate("draft_forward", color="red"):
            num_tokens = self.draft_params.topk_len

            request_kv_cache.increment(num_tokens)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree',
                device=token_ids.device,
                treeTokens=num_tokens,
            )
            self.flashinferWrapper.prepareAttention(
                'tree',
                batch_position,
                request_kv_cache.kvCachePool.page_len,
                "NONE", #POS_ENCODING_MODE.NONE
                request_kv_cache.kvCachePool.cache_data[0].dtype,
                attention_mask=tree_attention_mask,
            )

            sampled_probs = self(
                token_ids,
                with_softmax=True,
                past_key_values=None,
                position_ids=position_ids,
                kvCachePool=request_kv_cache.kvCachePool,
                batch_position=batch_position,
                mode='tree',
                flashinferWrapper=self.flashinferWrapper,
            )

        with nvtx.annotate("draft_sample", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len
            )
            parent_probs = child_probs

        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)

        # Update internal state
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids += 1


    def update_tree_after_post(self):
        """Return the finalized draft tree after post-speculation."""
        # Update the tree data and mask cache before returning
        self.update_tree(self.tree_data)
        return self.tree
