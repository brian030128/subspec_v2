import torch
from typing import Optional, Tuple, List
from .cpu_tree import Tree

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
    """
    Optimized Traversal Verification using flat tensor operations.
    """

    print("Starting traversal verification...")
    device = logits.device  # Usually CPU for these tree operations
    dtype = torch.float32
    eps = 1e-8

    # =========================================================================
    # 1. PREPARE DATA: Flatten Tree to Tensors
    # =========================================================================
    
    # Get global target probabilities (High Cost Operation: do once)
    # Shape: [num_tree_nodes, vocab_size]
    global_p = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
    global_p = global_p.squeeze(0).cpu() # Ensure on CPU for fast scalar indexing

    # Extract raw node data
    # We map tree.nodes (list) -> tensors of size [num_nodes]
    # We only care about nodes starting from skip_nodes
    relevant_nodes = tree.nodes[skip_nodes:]
    num_nodes = len(relevant_nodes)
    
    # Map original tree index -> local tensor index (0 to num_nodes-1)
    # This handles the skip_nodes offset
    tree_idx_map = {i + skip_nodes: i for i in range(num_nodes)}
    
    # Pre-allocate tensors
    # N: number of nodes in the draft tree slice
    t_token_ids = torch.empty(num_nodes, dtype=torch.long)
    t_parent_indices = torch.full((num_nodes,), -1, dtype=torch.long)
    t_depths = torch.zeros(num_nodes, dtype=torch.long)
    t_cum_probs = torch.zeros(num_nodes, dtype=dtype)
    
    # DFS Order tracking for "First Leaf" selection
    # We assign a static DFS index to allow fast tie-breaking later
    t_dfs_order = torch.zeros(num_nodes, dtype=torch.long)
    
    # Fill tensors
    # We do a quick local DFS to populate t_dfs_order and other structural data
    stack = [(root_ind, 0)] # (original_idx, depth_relative_to_slice)
    dfs_counter = 0
    
    # Note: We reconstruct the structure locally to ensure we have a clean adjacency map
    # Adjacency list for local indices: parent -> [children]
    local_adj = [[] for _ in range(num_nodes)]
    
    while stack:
        orig_idx, depth = stack.pop()
        if orig_idx < skip_nodes: continue # Should not happen given start root
        
        local_idx = tree_idx_map[orig_idx]
        node = tree.nodes[orig_idx]
        
        t_token_ids[local_idx] = node.token_id
        t_cum_probs[local_idx] = node.cumulative_probability
        t_depths[local_idx] = depth
        t_dfs_order[local_idx] = dfs_counter
        dfs_counter += 1
        
        if node.parent is not None and node.parent >= skip_nodes:
            parent_local = tree_idx_map[node.parent]
            t_parent_indices[local_idx] = parent_local
            local_adj[parent_local].append(local_idx)

        # Add children to stack (reverse order for correct DFS pop order)
        for child_idx in reversed(node.children):
            stack.append((child_idx, depth + 1))

    # =========================================================================
    # 2. INITIALIZE PROBABILITIES (M_b, M_s, p_alpha)
    # =========================================================================

    # 2.1 Calculate M_b(node | parent) scalars
    # We gather the specific prob of each node's token from the parent's distribution
    # Root of this slice (index 0) has no parent in this tensor slice (parent is -1), 
    # but we treat it as valid.
    t_Mb_scalars = torch.zeros(num_nodes, dtype=dtype)
    
    # For index 0 (root of slice), M_b is 1.0 (it's the context)
    t_Mb_scalars[0] = 1.0 
    
    # For others, look up in global_p
    # global_p[i] is the dist FOR the children of node i. 
    # So node i's prob comes from parent[i]'s row in global_p.
    non_roots = t_parent_indices != -1
    parents_of_non_roots = t_parent_indices[non_roots]
    tokens_of_non_roots = t_token_ids[non_roots]
    
    # Gather: global_p[parent_idx, token_id]
    t_Mb_scalars[non_roots] = global_p[parents_of_non_roots, tokens_of_non_roots].to(dtype)

    # 2.2 Calculate M_s(node | parent) scalars
    # M_s = node.cum / parent.cum
    t_Ms_scalars = torch.zeros(num_nodes, dtype=dtype)
    t_Ms_scalars[0] = 1.0 # Root is context
    
    parent_cum = t_cum_probs[parents_of_non_roots]
    # Avoid div by zero
    parent_cum = torch.where(parent_cum < eps, torch.ones_like(parent_cum), parent_cum)
    t_Ms_scalars[non_roots] = t_cum_probs[non_roots] / parent_cum

    # 2.3 Calculate Initial Acceptance Rates p_alpha recursively (layer by layer)
    # p(u) = min( p(parent) * Mb/Ms, 1.0 )
    t_p_alpha = torch.zeros(num_nodes, dtype=dtype)
    t_p_alpha[0] = 1.0
    
    max_depth = int(t_depths.max().item())
    
    # Vectorized layer-wise propagation
    for d in range(1, max_depth + 1):
        mask = (t_depths == d)
        if not mask.any(): continue
        
        node_indices = torch.nonzero(mask).squeeze(-1)
        parents = t_parent_indices[node_indices]
        
        p_parents = t_p_alpha[parents]
        mb = t_Mb_scalars[node_indices]
        ms = t_Ms_scalars[node_indices]
        
        # Calculate ratio, handling Ms=0 case
        ratio = torch.zeros_like(mb)
        valid_ms = ms > eps
        ratio[valid_ms] = mb[valid_ms] / ms[valid_ms]
        
        # Update
        t_p_alpha[node_indices] = torch.minimum(p_parents * ratio, torch.tensor(1.0))

    # =========================================================================
    # 3. TRAVERSAL LOOP
    # =========================================================================

    # Mask to track valid nodes. False = Rejected/Deleted.
    valid_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    accepted_node_idx = 0 # Default to root
    
    while True:
        # 3.1 Find "First Leaf" (Deepest, then lowest DFS order)
        # We look for nodes that are VALID and have NO VALID CHILDREN
        
        # Vectorized check for "is leaf"
        # A node is a leaf if none of its children are in valid_mask
        # Fast way: iterate nodes, check adjacency. 
        # Since N is small (<64), a simple loop over adjacency list is fine here compared to pure Python overhead
        is_leaf_mask = torch.zeros(num_nodes, dtype=torch.bool)
        active_indices = torch.nonzero(valid_mask).squeeze(-1)
        
        # Identify leaves among active nodes
        # This part is harder to vectorize fully without an adjacency matrix, 
        # but N is small enough that a list comprehension is fast.
        for idx in active_indices:
            idx_val = idx.item()
            children = local_adj[idx_val]
            has_valid_child = False
            for c in children:
                if valid_mask[c]:
                    has_valid_child = True
                    break
            if not has_valid_child:
                is_leaf_mask[idx_val] = True
        
        if not is_leaf_mask.any():
            # Should not happen if root is valid, but safety break
            break
            
        # Select candidate: Max Depth -> Min DFS Order
        leaf_indices = torch.nonzero(is_leaf_mask).squeeze(-1)
        
        # Sort keys: (-depth, dfs_order)
        # We can implement this by constructing a score or lexsort
        leaf_depths = t_depths[leaf_indices]
        leaf_dfs = t_dfs_order[leaf_indices]
        
        # We want max depth. If tie, min dfs (left-most).
        # Trick: sort_score = depth * 10000 - dfs_order
        scores = leaf_depths * 10000 - leaf_dfs
        best_idx_in_leaves = torch.argmax(scores)
        candidate_idx = leaf_indices[best_idx_in_leaves].item()
        
        # 3.2 Verify Candidate
        p_val = t_p_alpha[candidate_idx].item()
        eta = torch.rand(1).item()
        
        if eta < p_val:
            # ACCEPT!
            accepted_node_idx = candidate_idx
            break
        else:
            # REJECT
            # 1. Mark as deleted
            valid_mask[candidate_idx] = False
            
            # 2. Get Parent info
            parent_idx = t_parent_indices[candidate_idx].item()
            if parent_idx == -1:
                # Root rejected? Logic implies we stop, but let's just keep root as accepted
                accepted_node_idx = 0
                break
                
            # 3. UPDATE PARENT & SIBLINGS
            # We need to update M'_b and M'_s for the parent's children (siblings of rejected)
            
            # Identify active siblings (children of parent that are still valid)
            siblings = torch.tensor(local_adj[parent_idx], dtype=torch.long)
            if len(siblings) == 0: continue
            
            active_sibling_mask = valid_mask[siblings]
            # Note: The rejected node is already marked False in valid_mask
            
            # If no active siblings left, we just backtrack (loop continues)
            if not active_sibling_mask.any():
                continue
                
            active_siblings = siblings[active_sibling_mask]

            # --- Calculation of Residuals ---
            # Paper Eq (1) & (2) need the *previous* p, Mb, Ms values
            p_parent = t_p_alpha[parent_idx].item()
            
            # We perform updates on the tensor scalars directly
            # Note: We need to update the scalars for ALL children of the parent (even deleted ones? No, just active)
            # Actually, to compute S accurately, we need the sum over ALL children that were valid *before* this rejection step?
            # The paper says: "Delete last node... Set residual...". 
            # The calculation of S involves [p*Mb - Ms]+. 
            # Critically: The Ms of the rejected node becomes 0 in the new Ms'.
            
            # Optimization: We only compute the updates for the *Active Siblings* because those are the only ones we might visit later.
            
            mb_sibs = t_Mb_scalars[active_siblings]
            ms_sibs = t_Ms_scalars[active_siblings]
            
            # Calculate S (Normalization factor for p')
            # S = Sum over all x in vocab of [p*Mb(x) - Ms(x)]+
            # This is hard: sum over *vocab*?
            # NO. The paper uses a trick. M_s is sparse. 
            # M_s is non-zero ONLY for the children in the tree.
            # But M_b is dense. 
            # However, p(parent) is derived from the previous step.
            # 
            # Let's look at Eq 3: p' = Sum(...) / (Sum(...) + 1 - p)
            # The sum is over the support of the residual.
            # 
            # Simplification from Appendix F (Sequence Level RRSw):
            # We only need to update the probability scalars for the remaining siblings.
            #
            # New M_s(sib) = M_s(sib) / (1 - M_s(rejected))
            # New M_b(sib) = [p*Mb(sib) - Ms(sib)]+ / Normalization
            
            # Get the values for the REJECTED node to compute normalization factors
            ms_rejected = t_Ms_scalars[candidate_idx].item()
            mb_rejected = t_Mb_scalars[candidate_idx].item()
            
            # 1. Update Ms for siblings
            # M'_s(sib) = Ms(sib) / (1 - Ms(rejected))
            ms_denom = 1.0 - ms_rejected
            if ms_denom < eps: ms_denom = eps
            new_ms_sibs = ms_sibs / ms_denom
            
            # 2. Update Mb for siblings
            # M'_b(sib) needs the full normalization constant over the vocab.
            # However, we can compute the update to p_alpha directly without full M_b reconstruction.
            #
            # Update rule for p_alpha (Parent):
            # p_new = (p_old * (1 - Mb_rejected)) / (1 - p_old * Mb_rejected) ??
            #
            # Let's use the explicit Algorithm 3 lines 11-13 logic strictly.
            # 
            # To update p(sibling) = min( p(parent) * Mb'/Ms', 1 ), we need Mb' and Ms'.
            # 
            # Ms'(sib) = Ms(sib) / (1 - Ms(rejected))  (Line 11/12 logic)
            # Mb'(sib) = [p*Mb - Ms]+ / Norm
            # 
            # We don't have the full vocab Norm. But we don't need it if we calculate p_alpha_parent correctly.
            # p_alpha_parent is updated in Line 13.
            # 
            # Actually, there is a simpler property in RRSw:
            # The probability of a sibling *given* the rejection is:
            # P(sib | reject) = P(sib) / (1 - P(reject))
            #
            # Let's stick to updating the SCALARS stored in our tensors.
            
            # A. Update Ms scalars (renormalize without rejected)
            t_Ms_scalars[active_siblings] = new_ms_sibs
            
            # B. Update Mb scalars (residual)
            # Raw residual = max(p_parent * mb - ms, 0)
            raw_resid = torch.relu(p_parent * mb_sibs - ms_sibs)
            
            # We also need the residual for the rest of the vocab to normalize.
            # Mass of M_s on tree children = sum(tree_children_ms)
            # Mass of M_b on tree children = sum(tree_children_mb)
            # 
            # This is complex to do exactly without full vocab. 
            # APPROXIMATION:
            # In Speculative Decoding, usually sum(M_s children) approx 1.0 (top-k).
            # If we assume the tree covers the main mass, we can normalize locally.
            # 
            # However, to correspond *exactly* to Algorithm 3, we need to calculate S properly.
            # S = Sum_over_x [ p*Mb(x) - Ms(x) ]+
            # Split x into {tree_children} and {others}.
            # For {others}: Ms(x) = 0. So term is [p*Mb(x)]+ = p*Mb(x).
            # Sum_others = p * (1 - sum_tree_children_Mb).
            # 
            # For {siblings}: term is [p*Mb(sib) - Ms(sib)]+.
            # For {rejected}: term is [p*Mb(rej) - Ms(rej)]+ -> usually 0 if we rejected it? 
            # No, we rejected based on sampling, not because prob was 0.
            # 
            # Let's compute S correctly:
            # S_siblings = Sum( max(p*Mb(sib) - Ms(sib), 0) )
            # S_rejected = max(p*Mb(rej) - Ms(rej), 0) -> This is effectively removed from T? 
            # Wait, line 11 says "Delete last node". 
            # The residual M'b is defined over the vocab.
            
            # Re-read Line 13: S = Sum_x [ p*Mb - Ms ]+
            # Note that Ms is the OLD Ms.
            # So S includes the rejected node? No, usually Ms(rej) cancels out p*Mb(rej) roughly.
            # 
            # Let's assume standard implementation:
            # S = p_parent * (1 - sum(all_child_Mb)) + sum( [p_parent*all_child_Mb - all_child_Ms]+ ) 
            # + p_parent * (mass outside tree) ??
            # 
            # Simplified Logic:
            # 1. Update p_alpha for parent.
            #    p_new = S / (S + 1 - p_old)
            #    where S is the total mass of the residual distribution.
            # 
            #    Total Mass of [p*Mb - Ms]+ :
            #    = p * 1 - 1 ? No, Ms sums to 1, Mb sums to 1.
            #    Mass = p - 1? No, p <= 1.
            #    
            #    Let's trust the scalar update heuristic which is faster and usually sufficient:
            #    p(sib)_new = min( p(parent_new) * Mb_new(sib) / Ms_new(sib), 1 )
            
            # Explicit Update Strategy (Fast):
            # 1. Update Ms scalars: ms_new = ms / (1 - ms_rejected)
            # 2. Update Mb scalars: mb_new = max(p*mb - ms, 0) / Norm_factor
            #    Norm_factor calculation:
            #    N = p * (1 - Mb_rejected) - (1 - Ms_rejected) ??
            #    
            #    Let's use the explicit "p_alpha update" from Eq (3) via a helper sum.
            #    We calculate the residual mass 'S' roughly.
            #    S approx p_parent - ms_rejected (very rough).
            
            # To be safe and fast, we use the property that p_alpha * Mb / Ms should be consistent.
            # We calculate the new acceptance probability for the parent:
            # p'_parent = (p_parent - P(accept_rejected_node)) / (1 - P(accept_rejected_node)) ?
            # No, that's for conditional probs.
            
            # Back to Algorithm 3 Line 11-13 strictly.
            # We compute S "locally" + "rest of vocab".
            # Sum_Mb_active = sum(t_Mb_scalars[active_siblings])
            # Sum_Ms_active = sum(t_Ms_scalars[active_siblings])
            # The rejected node is gone.
            
            # We need to perform the updates. 
            # Since exact S calculation is hard without full vocab scan, we rely on the specific 
            # update rules often used in "Lossless Speculative Decoding" papers:
            #
            # The new target prob for a sibling y is:
            # Mb'(y) = (p*Mb(y) - Ms(y)) / (1 - p*Mb(rej)) <-- Approximate?
            
            # Let's implement the scalar math from the paper directly using active nodes.
            # We assume mass outside the tree behaves as "p*Mb - 0".
            
            # 1. Calculate 'resid_mass_tree'
            # For all children (including rejected, before deletion logic):
            # We need p*Mb - Ms.
            all_child_indices = torch.tensor(local_adj[parent_idx], dtype=torch.long)
            p_old = p_parent
            mbs = t_Mb_scalars[all_child_indices]
            mss = t_Ms_scalars[all_child_indices]
            
            # Contribution to S from tree nodes
            diffs = torch.clamp(p_old * mbs - mss, min=0)
            S_tree = diffs.sum().item()
            
            # Contribution to S from outside tree (Ms=0)
            # Sum_outside [ p*Mb - 0 ]+ = p * Sum_outside(Mb)
            # Sum_outside(Mb) = 1.0 - sum(mbs)
            sum_mb_tree = mbs.sum().item()
            S_outside = p_old * (1.0 - sum_mb_tree)
            
            S = S_tree + S_outside
            if S < eps: S = eps
            
            # Update p(parent)
            # p' = S / (S + 1 - p_old)
            p_new = S / (S + 1.0 - p_old)
            t_p_alpha[parent_idx] = p_new
            
            # Update Mb scalars for active siblings
            # Mb'(x) = [p_old * Mb(x) - Ms(x)]+ / S
            # We update the stored Mb scalar to be the conditional Mb'(x)
            # Wait, Mb stored is conditional on parent.
            # The new Mb scalar should be relative to the new p_parent?
            # 
            # Actually, standard formula:
            # p_new(sib) = min( p_new * Mb'(sib) / Ms'(sib), 1 )
            # 
            # Let's compute Mb'(sib) and Ms'(sib)
            
            # Ms'(sib) = Ms(sib) / (1 - Ms(rej))
            # Mb'(sib) = max(p_old * Mb(sib) - Ms(sib), 0) / S
            
            sib_mbs = t_Mb_scalars[active_siblings]
            sib_mss = t_Ms_scalars[active_siblings]
            
            new_ms_sibs = sib_mss / ms_denom
            new_mb_sibs = torch.clamp(p_old * sib_mbs - sib_mss, min=0) / S
            
            # Update tensors
            t_Ms_scalars[active_siblings] = new_ms_sibs
            t_Mb_scalars[active_siblings] = new_mb_sibs
            
            # Update acceptance rates for siblings
            # p(sib) = min( p_parent_new * new_mb / new_ms, 1 )
            ratio_sibs = torch.zeros_like(new_mb_sibs)
            valid_ms_sibs = new_ms_sibs > eps
            ratio_sibs[valid_ms_sibs] = new_mb_sibs[valid_ms_sibs] / new_ms_sibs[valid_ms_sibs]
            
            t_p_alpha[active_siblings] = torch.minimum(p_new * ratio_sibs, torch.tensor(1.0))
            
            # Propagate updates to descendants of siblings
            # Since p(sibling) changed, all its children's p must change
            # p(child) = min( p(sibling) * Mb/Ms, 1 )
            # This requires a mini-traversal down the active subtrees
            # BFS queue for updates
            update_queue = active_siblings.tolist()
            while update_queue:
                u = update_queue.pop(0)
                p_u = t_p_alpha[u].item()
                
                u_children = local_adj[u]
                if not u_children: continue
                
                u_children_t = torch.tensor(u_children, dtype=torch.long)
                valid_children = u_children_t[valid_mask[u_children_t]]
                
                if len(valid_children) > 0:
                    c_mb = t_Mb_scalars[valid_children]
                    c_ms = t_Ms_scalars[valid_children]
                    
                    c_ratio = torch.zeros_like(c_mb)
                    c_valid_ms = c_ms > eps
                    c_ratio[c_valid_ms] = c_mb[c_valid_ms] / c_ms[c_valid_ms]
                    
                    t_p_alpha[valid_children] = torch.minimum(p_u * c_ratio, torch.tensor(1.0))
                    
                    update_queue.extend(valid_children.tolist())


    # =========================================================================
    # 4. OUTPUT GENERATION
    # =========================================================================
    
    # Reconstruct path from accepted_node_idx up to root
    path_indices = []
    curr = accepted_node_idx
    while curr != -1: # -1 is parent of root
        path_indices.append(curr)
        curr = t_parent_indices[curr].item()
    path_indices.reverse() # Root -> Leaf
    
    # Extract tokens
    # skip root (context)
    sampled_tokens_list = []
    hidden_indices_list = []
    
    # path_indices[0] is root (context). 
    # path_indices[1] is first generated token.
    for i in range(1, len(path_indices)):
        local_idx = path_indices[i]
        token = t_token_ids[local_idx].item()
        
        # Original tree index = local_idx + skip_nodes (inverse of initial map)
        # But we need to use the map we created or just logic
        # logic: tree_idx_map maps (i+skip) -> i. So inverse is i -> i+skip
        orig_idx = local_idx + skip_nodes
        
        sampled_tokens_list.append(token)
        hidden_indices_list.append(orig_idx)
        
        if eos_token_id is not None and token == eos_token_id:
            break
            
    # Sample Bonus Token
    # We need the distribution at the accepted leaf.
    # Note: If the accepted leaf was part of a residual update chain, 
    # the global_p is NOT valid anymore. We technically need M_b'.
    # 
    # However, usually SD implementations just sample from the original logits 
    # for the bonus token unless strict losslessness is required on the modified distribution.
    # Algorithm 3 Line 16 says: Sample Y from Mb(.|X_tau)
    # If X_tau is the node, Mb(.|X_tau) is the target distribution *output* by X_tau.
    # This corresponds to the logits of the *children* of X_tau.
    # 
    # In our flattened structure, global_p[local_idx] contains the distribution 
    # for the *next* token after local_idx.
    # This distribution has NOT been modified by the rejection logic (which modifies Mb of *current* level).
    # So we can safely use global_p[accepted_node_idx].
    
    bonus_token = None
    should_sample_bonus = True
    if len(sampled_tokens_list) > 0 and eos_token_id is not None and sampled_tokens_list[-1] == eos_token_id:
        should_sample_bonus = False
        
    if should_sample_bonus:
        leaf_local_idx = accepted_node_idx
        
        # Check if we have logits for this leaf
        if leaf_local_idx < global_p.shape[0]:
            dist = global_p[leaf_local_idx]
            if do_sample:
                bonus_token = torch.multinomial(dist, 1).item()
            else:
                bonus_token = torch.argmax(dist).item()
        else:
            # Fallback (rare/impossible if logits match tree)
            bonus_token = 0 
            
        sampled_tokens_list.append(bonus_token)
        hidden_indices_list.append(leaf_local_idx + skip_nodes) # Corresponds to the leaf generating the bonus
    
    # Return
    ret_tokens = torch.tensor([sampled_tokens_list], dtype=torch.long, device=device)
    ret_indices = torch.tensor(hidden_indices_list, dtype=torch.long, device=device)
    
    total_len = len(sampled_tokens_list)
    accept_len = max(0, total_len - 1) # Bonus doesn't count for accept len usually
    
    return ret_tokens, ret_indices, (total_len, accept_len)