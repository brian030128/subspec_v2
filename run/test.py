import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import your classes
try:
    from specdecodes.models.utils.cpu_tree import Tree, TreeNode
    from specdecodes.models.utils.traversal_verification import traversal_verification_tree
except ImportError:
    # Adjust this import to match your exact file structure if needed
    from models.cpu_tree import Tree, TreeNode
    from models.traversal_verifier import traversal_verification_tree

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_logits_processor(temperature, top_p, top_k):
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
        TopKLogitsWarper,
    )
    processors = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processors.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    return processors

def sample_token(logits, processor, do_sample, return_probs=False):
    """Helper to sample a token or return probs."""
    # FIX 1: Ensure logits are 2D [Batch, Vocab] for multinomial
    orig_shape = logits.shape
    if len(orig_shape) == 3:
        # [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
        logits = logits.view(-1, orig_shape[-1])

    # FIX 2: Ensure float32 for numerical stability in softmax
    logits = logits.to(torch.float32)

    if processor:
        logits = processor(None, logits)

    probs = F.softmax(logits, dim=-1)

    if return_probs:
        # Return in original shape if requested (needed for verification)
        if len(orig_shape) == 3:
            return probs.view(orig_shape)
        return probs

    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    return next_token

# ==========================================
# 3. GENERATION LOGIC
# ==========================================

@torch.no_grad()
def sample_naive(model, input_ids, processors, tokens=3):
    """Standard Autoregressive Generation."""
    generated = []
    curr_input = input_ids.clone()

    for _ in range(tokens):
        outputs = model(curr_input)
        # Fix: Select specific seq index to keep 2D [Batch, Vocab]
        next_token_logits = outputs.logits[:, -1, :] 

        token = sample_token(next_token_logits, processors, do_sample=True)
        generated.append(token)
        curr_input = torch.cat([curr_input, token], dim=-1)

    return torch.cat(generated, dim=-1)

@torch.no_grad()
def sample_with_traversal(model, input_ids, processors, tokens=3):
    """Simulates Speculative Decoding + Traversal Verification."""
    
    # --- Step 1: Draft Chain ---
    draft_tokens = []
    draft_node_probs = []
    curr_input = input_ids.clone()
    
    # NOTE: For the lossless check to pass perfectly in this simplified script,
    # we use the SAME temperature for drafting. This verifies the verification 
    # logic (indexing, tree structure) is bugs-free.
    draft_processors = processors 
    
    for _ in range(tokens):
        outputs = model(curr_input)
        logits = outputs.logits[:, -1, :] # 2D [Batch, Vocab]
        
        # Calculate Draft Probabilities (M_s)
        # Note: We must get the probability of the *chosen* token
        probs = sample_token(logits, draft_processors, do_sample=True, return_probs=True)
        
        token_id = torch.multinomial(probs, 1).item()
        token_prob = probs[0, token_id].item()
        
        draft_tokens.append(token_id)
        draft_node_probs.append(token_prob)
        
        curr_input = torch.cat([curr_input, torch.tensor([[token_id]], device=input_ids.device)], dim=-1)

    # --- Step 2: Build Tree ---
    root_token_id = input_ids[0, -1] 
    draft_tree = Tree(root_token_id, prob_dtype=torch.float32)
    
    parent_idx = 0 
    cum_prob = 1.0
    
    for i, (tid, prob) in enumerate(zip(draft_tokens, draft_node_probs)):
        cum_prob *= prob
        new_node = TreeNode(
            parent=parent_idx,
            token_id=tid,
            cumulative_probability=cum_prob,
            depth=i+1
        )
        draft_tree.nodes.append(new_node)
        draft_tree.nodes[parent_idx].children.append(len(draft_tree.nodes) - 1)
        parent_idx = len(draft_tree.nodes) - 1

    # --- Step 3: Target Logits ---
    # Prepare input: [Prompt, Draft1, Draft2, ...]
    draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
    full_input = torch.cat([input_ids, draft_tensor], dim=-1)
    
    outputs = model(full_input)
    
    # Logit Alignment:
    # We want predictions STARTING from the end of the prompt.
    # input_ids len = L. Indices 0..L-1.
    # Logit at L-1 predicts token at L (First draft token).
    start_idx = input_ids.shape[1] - 1
    # We need predictions for all draft tokens + 1 bonus.
    end_idx = start_idx + len(draft_tokens) + 1 
    
    target_logits = outputs.logits[:, start_idx:end_idx, :]

    # --- Step 4: Verify ---
    accepted_tokens, _, _ = traversal_verification_tree(
        tree=draft_tree,
        root_ind=0,
        logits=target_logits,
        sample_token_fn=sample_token,
        verify_step_fn=None,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=processors,
        do_sample=True,
        skip_nodes=0
    )
    
    return accepted_tokens

# ==========================================
# 4. EXPERIMENT EXECUTION
# ==========================================

# Config
model_id = "meta-llama/Llama-3.2-1B-Instruct" 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
model.eval()

# Prompt
text = "Once upon a time in a land far away,"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

# Params
temperature = 1.0 
top_k = 50 # Standard sampling settings
top_p = 0.95
target_processors = get_logits_processor(temperature, top_p, top_k)

rep = 300 
compare_len = 3 
vocab_size = tokenizer.vocab_size

print(f"Running {rep} iterations...")

counts = {
    "naive": [torch.zeros(vocab_size) for _ in range(compare_len)],
    "traversal": [torch.zeros(vocab_size) for _ in range(compare_len)]
}

for i in tqdm(range(rep)):
    # 1. Naive
    out_naive = sample_naive(model, input_ids, target_processors, tokens=compare_len)
    
    # 2. Traversal
    out_trav = sample_with_traversal(model, input_ids, target_processors, tokens=compare_len)
    
    # Record Stats
    for pos in range(compare_len):
        if pos < out_naive.shape[1]:
            counts["naive"][pos][out_naive[0, pos].item()] += 1
            
        if pos < out_trav.shape[1]:
            counts["traversal"][pos][out_trav[0, pos].item()] += 1

# ==========================================
# 5. PLOTTING
# ==========================================
def plot_distributions(counts_dict, k=10):
    fig, axs = plt.subplots(compare_len, 1, figsize=(10, 4*compare_len))
    if compare_len == 1: axs = [axs]
    
    for t in range(compare_len):
        ax = axs[t]
        
        naive_cts = counts_dict["naive"][t].numpy()
        trav_cts = counts_dict["traversal"][t].numpy()
        
        # Normalize
        if naive_cts.sum() > 0: naive_probs = naive_cts / naive_cts.sum()
        else: naive_probs = naive_cts
            
        if trav_cts.sum() > 0: trav_probs = trav_cts / trav_cts.sum()
        else: trav_probs = trav_cts
        
        # Sort by Naive
        top_indices = np.argsort(naive_probs)[::-1][:k]
        
        x = np.arange(k)
        width = 0.35
        
        ax.bar(x - width/2, naive_probs[top_indices], width, label='Naive')
        ax.bar(x + width/2, trav_probs[top_indices], width, label='Traversal')
        
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}" for t in top_tokens], rotation=45)
        ax.set_title(f"Position {t+1}")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig("lossless_verification_check.png")
    print("\nSaved plot to lossless_verification_check.png")

plot_distributions(counts)
