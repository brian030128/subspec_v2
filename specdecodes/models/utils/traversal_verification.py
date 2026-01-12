import torch
from typing import Optional, Tuple

from .cpu_tree import Tree
from .lossy_tree_verify import lossy_bottom_up_verify


@torch.no_grad()
def traversal_verification_tree(
    *,
    tree: Tree,
    root_ind: int,
    logits: torch.Tensor,
    sample_token_fn,
    verify_step_fn,
    eos_token_id: Optional[int],
    logits_processor,
    do_sample: bool,
    skip_nodes: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Traversal verification algorithm for speculative decoding.

    Implements the algorithm from the pseudocode that iteratively:
    1. Initializes acceptance probabilities for all nodes
    2. Traverses tree in post-order DFS, accepting/rejecting paths
    3. Updates residual distributions when paths are rejected
    4. Returns accepted path plus bonus token

    Args:
        tree: Draft tree object (CPU tree).
        root_ind: Root node index in original tree indexing.
        logits: Target model logits for the decoded tree slice.
        sample_token_fn: Callable to get probabilities from logits.
        verify_step_fn: Not used (kept for signature compatibility).
        eos_token_id: EOS token id.
        logits_processor: HF LogitsProcessorList.
        do_sample: Whether to sample target token.
        skip_nodes: Number of leading nodes skipped (default 0).

    Returns:
        sampled_tokens: (1, L) tensor
        hidden_indices: (L,) tensor of node indices
        (total_len, accept_len): metrics tuple
    """

    eps = 1e-10  # Epsilon for numerical stability

    # ========== PHASE 1: INITIALIZATION ==========

    # 1.1: Get target model distributions (M_b)
    global_p = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
    global_p = global_p.squeeze(0).cpu()  # (num_nodes, vocab_size)

    # Preserve CUDA synchronization behavior
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 1.2: Build tree structure
    node_data = tree.get_tree_data(skip_nodes=skip_nodes)
    num_nodes = len(tree.nodes) - skip_nodes
    vocab_size = global_p.shape[1]

    # 1.3 & 1.4: Add fields to TreeNode and compute distributions
    for node_idx in range(skip_nodes, len(tree.nodes)):
        node = tree.nodes[node_idx]
        local_idx = node_idx - skip_nodes

        # Initialize new fields
        node.p_alpha_ini = 0.0
        node.p_alpha = 0.0
        node.M_b = {}
        node.M_s = {}
        node.is_deleted = False

        # Compute M_b: Target distribution (sparse)
        if local_idx < global_p.shape[0]:
            M_b_full = global_p[local_idx]
            for token_id in range(vocab_size):
                prob = M_b_full[token_id].item()
                if prob > eps:
                    node.M_b[token_id] = prob

        # Compute M_s: Draft distribution from tree structure
        if node.parent is not None:
            parent_node = tree.nodes[node.parent]
            parent_cum_prob = parent_node.cumulative_probability

            if parent_cum_prob > eps:
                # Add all sibling probabilities (including self)
                for sibling_idx in parent_node.children:
                    sibling = tree.nodes[sibling_idx]
                    sibling_prob = sibling.cumulative_probability / parent_cum_prob
                    if sibling_prob > eps:
                        node.M_s[sibling.token_id] = sibling_prob

    # 1.5: Compute initial acceptance probabilities recursively
    def compute_initial_acceptance(node_idx: int):
        """Recursively compute initial acceptance probabilities."""
        node = tree.nodes[node_idx]

        if node.parent is None:
            # Root node always has acceptance probability 1
            node.p_alpha_ini = 1.0
            node.p_alpha = 1.0
        else:
            parent = tree.nodes[node.parent]
            token_id = node.token_id

            # Get M_b(token | parent) and M_s(token | parent)
            M_b_val = parent.M_b.get(token_id, 0.0)
            M_s_val = parent.M_s.get(token_id, 0.0)

            # Compute acceptance rate: M_b / M_s
            if M_s_val > eps:
                acceptance_rate = M_b_val / M_s_val
            else:
                # Token not in draft but appears in target
                acceptance_rate = 0.0

            node.p_alpha_ini = min(parent.p_alpha_ini * acceptance_rate, 1.0)
            node.p_alpha = node.p_alpha_ini

        # Recursively process children
        for child_idx in node.children:
            compute_initial_acceptance(child_idx)

    compute_initial_acceptance(root_ind)

    # ========== PHASE 2: MAIN LOOP - POST-ORDER DFS TRAVERSAL ==========

    def find_first_leaf_postorder(root_idx: int):
        """Find first leaf in post-order DFS traversal.

        Returns: (leaf_idx, path_from_root)
        """
        current = root_idx
        path = []

        while True:
            path.append(current)
            current_node = tree.nodes[current]

            # Filter out deleted children
            active_children = [c for c in current_node.children
                             if not tree.nodes[c].is_deleted]

            if not active_children:
                # Found a leaf (or all children deleted)
                return current, path

            # Go to leftmost (first) child
            current = active_children[0]

    def update_residuals_on_rejection(rejected_idx: int):
        """Update residual distributions when a node is rejected."""
        rejected_node = tree.nodes[rejected_idx]
        parent_idx = rejected_node.parent

        if parent_idx is None:
            # Cannot reject root
            return

        # Mark node as deleted
        rejected_node.is_deleted = True

        parent = tree.nodes[parent_idx]
        rejected_token = rejected_node.token_id

        # Compute residual M'_b at parent
        M_b_new = {}
        norm_b = 0.0

        for token, M_b_val in parent.M_b.items():
            M_s_val = parent.M_s.get(token, 0.0)
            # Residual: [p_α(parent) * M_b(x) - M_s(x)]_+
            residual = max(parent.p_alpha * M_b_val - M_s_val, 0.0)
            if residual > eps:
                M_b_new[token] = residual
                norm_b += residual

        # Normalize M_b_new
        if norm_b > eps:
            for token in M_b_new:
                M_b_new[token] /= norm_b

        # Compute residual M'_s at parent (remove rejected token)
        M_s_new = {}
        norm_s = 0.0

        for token, M_s_val in parent.M_s.items():
            if token != rejected_token:
                M_s_new[token] = M_s_val
                norm_s += M_s_val

        # Normalize M_s_new
        if norm_s > eps:
            for token in M_s_new:
                M_s_new[token] /= norm_s

        # Update parent's distributions
        parent.M_b = M_b_new
        parent.M_s = M_s_new

        # Update parent's p_alpha
        if norm_s > eps and norm_b > eps:
            parent.p_alpha = min(norm_b / norm_s, 1.0)
        else:
            parent.p_alpha = 0.0

        # Recursively update acceptance probabilities for all descendants
        def recompute_acceptance(node_idx: int):
            """Recursively recompute acceptance probabilities."""
            node = tree.nodes[node_idx]

            if node.is_deleted:
                return

            if node.parent is not None:
                parent_node = tree.nodes[node.parent]
                token_id = node.token_id

                M_b_val = parent_node.M_b.get(token_id, 0.0)
                M_s_val = parent_node.M_s.get(token_id, 0.0)

                if M_s_val > eps:
                    rate = M_b_val / M_s_val
                else:
                    rate = 0.0

                node.p_alpha = min(parent_node.p_alpha * rate, 1.0)

            # Recursively update children
            for child_idx in node.children:
                recompute_acceptance(child_idx)

        # Update all children of parent
        for child_idx in parent.children:
            if not tree.nodes[child_idx].is_deleted:
                recompute_acceptance(child_idx)

    # Main verification loop
    max_iterations = 1000
    accepted_path = None

    for iteration in range(max_iterations):
        # Check if root is deleted (all paths exhausted)
        if tree.nodes[root_ind].is_deleted:
            accepted_path = [root_ind]
            break

        # Find first leaf in post-order DFS
        leaf_idx, path = find_first_leaf_postorder(root_ind)
        leaf_node = tree.nodes[leaf_idx]

        # Sample uniform random number
        eta = torch.rand(1).item()

        # Accept/reject decision
        if eta < leaf_node.p_alpha:
            # Accept this path
            accepted_path = path
            break
        else:
            # Reject: update residuals and continue
            update_residuals_on_rejection(leaf_idx)

            # Check if all root's children are deleted
            root_node = tree.nodes[root_ind]
            active_root_children = [c for c in root_node.children
                                   if not tree.nodes[c].is_deleted]

            if not active_root_children:
                # All paths rejected, return root only
                accepted_path = [root_ind]
                break

    # Safety: if loop exhausted, use root
    if accepted_path is None:
        accepted_path = [root_ind]

    # ========== PHASE 3: EXTRACT TOKENS AND SAMPLE BONUS ==========

    sampled_tokens = []
    hidden_indices = []

    # 3.1: Build token sequence from accepted path
    # Skip root (which is just context), extract tokens from children
    for i in range(len(accepted_path) - 1):
        context_idx = accepted_path[i]
        node_idx = accepted_path[i + 1]
        node = tree.nodes[node_idx]

        sampled_tokens.append(node.token_id)
        hidden_indices.append(context_idx)

        # Stop at EOS
        if eos_token_id is not None and node.token_id == eos_token_id:
            break

    accept_len = len(sampled_tokens)

    # 3.2: Sample bonus token
    final_context_idx = accepted_path[-1]
    final_node = tree.nodes[final_context_idx]

    # Check if we should sample bonus
    should_sample_bonus = (
        len(sampled_tokens) == 0 or
        eos_token_id is None or
        sampled_tokens[-1] != eos_token_id
    )

    if should_sample_bonus:
        # Sample from M_b at final context
        M_b_dict = final_node.M_b

        if len(M_b_dict) > 0:
            # Convert dict to tensor for sampling
            tokens = list(M_b_dict.keys())
            probs = torch.tensor([M_b_dict[t] for t in tokens], dtype=torch.float32)

            if do_sample:
                sampled_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                sampled_idx = probs.argmax().item()

            bonus_token = tokens[sampled_idx]
        else:
            # Fallback to original global_p if M_b is empty
            local_idx = final_context_idx - skip_nodes
            if local_idx < global_p.shape[0]:
                dist = global_p[local_idx]
                bonus_token = dist.multinomial(1).item() if do_sample else dist.argmax().item()
            else:
                # Edge case: use first token as fallback
                bonus_token = 0

        sampled_tokens.append(bonus_token)
        hidden_indices.append(final_context_idx)

    # 3.3: Return results in expected format
    sampled_tokens_tensor = torch.tensor(sampled_tokens, dtype=torch.long, device="cpu")
    hidden_indices_tensor = torch.tensor(hidden_indices, dtype=torch.long, device="cpu")

    total_len = len(sampled_tokens)

    return sampled_tokens_tensor.unsqueeze(0), hidden_indices_tensor, (total_len, int(accept_len))
