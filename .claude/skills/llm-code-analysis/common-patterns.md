# Common ML Implementation Patterns

Reference guide for recognizing and understanding common patterns in LLM/ML implementations.

---

## 1. Execution Models

### 1.1 Eager Execution (PyTorch Default)
```python
# Each operation launches a separate kernel
x = linear1(x)      # kernel 1
x = activation(x)   # kernel 2
x = linear2(x)      # kernel 3
```
- **Pros:** Easy to debug, flexible
- **Cons:** Kernel launch overhead, no fusion

### 1.2 Graph-Based Execution (torch.compile, JAX)
```python
@torch.compile
def fused_mlp(x, w1, w2):
    return linear2(activation(linear1(x, w1)), w2)
```
- **Pros:** Automatic fusion, optimization
- **Cons:** Compilation overhead, tracing limitations

### 1.3 Megakernel Execution
```python
# Single kernel handles multiple operations
# Tasks scheduled to SMs via work queue
megakernel[grid](work_queue, scoreboard, ...)
```
- **Pros:** Minimal launch overhead, fine-grained scheduling
- **Cons:** Complex implementation, debugging difficulty

### 1.4 Persistent Kernel
```python
@triton.jit
def persistent_kernel(...):
    while work_available():
        task = get_next_task()
        process(task)
        signal_completion()
```
- **Pros:** Avoids kernel launch, dynamic scheduling
- **Cons:** Occupancy management, synchronization complexity

---

## 2. Attention Patterns

### 2.1 Standard Attention (Memory Bound)
```python
def standard_attention(Q, K, V):
    # Materializes full attention matrix
    scores = Q @ K.T / sqrt(d)        # [B, H, S, S]
    weights = softmax(scores, dim=-1)  # [B, H, S, S]
    output = weights @ V               # [B, H, S, D]
    return output
```
- Memory: O(S²) for attention matrix
- Use case: Short sequences, debugging

### 2.2 Flash Attention (Tiled, Online Softmax)
```python
def flash_attention(Q, K, V, BLOCK_M, BLOCK_N):
    """
    Key insight: Never materialize full attention matrix
    Use online softmax to compute incrementally
    """
    for q_block in tiles(Q, BLOCK_M):
        m_i, l_i = -inf, 0  # softmax accumulators
        acc = 0

        for k_block, v_block in tiles(K, V, BLOCK_N):
            scores = q_block @ k_block.T
            # Online softmax update
            m_new = max(m_i, max(scores))
            l_i = l_i * exp(m_i - m_new) + sum(exp(scores - m_new))
            acc = acc * exp(m_i - m_new) + softmax_local @ v_block
            m_i = m_new

        output_block = acc / l_i
```
- Memory: O(S) - no attention matrix materialized
- I/O: Q loaded once, K/V streamed

### 2.3 Multi-Query Attention (MQA)
```python
# Multiple query heads share single K, V heads
# Q: [B, S, num_q_heads, head_dim]
# K, V: [B, S, 1, head_dim]  # Single head
scores = einsum('bqhd,bkd->bqhk', Q, K)
```
- Memory savings: KV cache reduced by num_q_heads×
- Common in: GPT-Neo, Falcon

### 2.4 Grouped Query Attention (GQA)
```python
# Query heads grouped, share K, V within group
# Q: [B, S, num_q_heads, head_dim]
# K, V: [B, S, num_kv_heads, head_dim]
# num_q_heads = num_kv_heads * group_size
group_size = num_q_heads // num_kv_heads
k_expanded = K.repeat_interleave(group_size, dim=2)
```
- Balance between MHA and MQA
- Common in: Llama 2/3, Qwen, Mistral

### 2.5 Sliding Window Attention
```python
def sliding_window_attention(Q, K, V, window_size):
    # Each query attends to window_size previous tokens
    for i, q in enumerate(Q):
        start = max(0, i - window_size)
        k_window = K[start:i+1]
        v_window = V[start:i+1]
        output[i] = attention(q, k_window, v_window)
```
- Memory: O(S × W) where W is window size
- Common in: Mistral, Longformer

---

## 3. MLP Patterns

### 3.1 Standard MLP
```python
def mlp(x, w_up, w_down):
    hidden = activation(x @ w_up.T)
    output = hidden @ w_down.T
    return output
```

### 3.2 Gated MLP (SwiGLU, GeGLU)
```python
def gated_mlp(x, w_gate, w_up, w_down):
    """
    Gate and up projections computed together
    SwiGLU: gate_activation = SiLU
    GeGLU: gate_activation = GELU
    """
    gate = gate_activation(x @ w_gate.T)
    up = x @ w_up.T
    hidden = gate * up  # element-wise
    output = hidden @ w_down.T
    return output
```
- Common in: Llama, Qwen, Mistral
- Often fused: gate_up_proj combined into single matmul

### 3.3 MoE (Mixture of Experts)
```python
def moe_mlp(x, router, experts):
    # Router selects top-k experts per token
    scores = router(x)
    top_k_experts = topk(scores, k)

    output = 0
    for expert_id, weight in top_k_experts:
        output += weight * experts[expert_id](x)
    return output
```
- Sparse activation: only k experts run per token
- Common in: Mixtral, Switch Transformer

---

## 4. Normalization Patterns

### 4.1 Layer Norm
```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x_norm = (x - mean) / sqrt(var + eps)
    return gamma * x_norm + beta
```

### 4.2 RMS Norm (Root Mean Square)
```python
def rms_norm(x, weight, eps=1e-6):
    """
    No mean centering, no bias
    Faster than LayerNorm
    """
    rms = sqrt(mean(x ** 2) + eps)
    return weight * (x / rms)
```
- Common in: Llama, Qwen, Mistral
- Pre-norm: norm applied before attention/MLP

### 4.3 QK Norm
```python
def qk_norm(q, k, q_weight, k_weight):
    """
    Normalize Q and K before attention
    Helps with training stability
    """
    q_norm = rms_norm(q, q_weight)
    k_norm = rms_norm(k, k_weight)
    return q_norm, k_norm
```
- Common in: Qwen3, some newer models

---

## 5. Position Encoding Patterns

### 5.1 Absolute Positional Embedding
```python
def add_position(x, pos_embed):
    # pos_embed: [max_seq_len, hidden_dim]
    seq_len = x.shape[1]
    return x + pos_embed[:seq_len]
```

### 5.2 Rotary Position Embedding (RoPE)
```python
def apply_rope(x, cos, sin):
    """
    Rotate pairs of dimensions based on position
    Encodes relative position in attention scores
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)
    return x_rotated
```
- Applied to Q and K
- Relative position: naturally captured in Q·K
- Common in: Llama, Qwen, Mistral

### 5.3 ALiBi (Attention with Linear Biases)
```python
def alibi_attention(Q, K, V, slopes):
    # Add position-dependent bias to attention scores
    positions = torch.arange(seq_len)
    bias = slopes * (positions[:, None] - positions[None, :])
    scores = Q @ K.T + bias
    return softmax(scores) @ V
```
- No learned parameters
- Common in: BLOOM, MPT

---

## 6. Parallelism Patterns

### 6.1 Tensor Parallelism (TP)
```python
# Column-parallel: split output dimension
# w_full: [out_dim, in_dim]
# w_shard: [out_dim // TP, in_dim] per rank
output_shard = x @ w_shard.T
output = all_gather(output_shard)  # or use directly

# Row-parallel: split input dimension
# w_shard: [out_dim, in_dim // TP]
input_shard = scatter(x)  # or x is already sharded
output_partial = input_shard @ w_shard.T
output = all_reduce(output_partial)
```
- Attention: QKV column-parallel, O row-parallel
- MLP: up/gate column-parallel, down row-parallel

### 6.2 Pipeline Parallelism (PP)
```python
# Layers distributed across ranks
# Micro-batching for efficiency
for micro_batch in micro_batches:
    if rank == 0:
        output = layers_0_to_n(input)
        send(output, rank=1)
    elif rank == last:
        input = recv(rank=rank-1)
        output = layers_m_to_end(input)
    else:
        input = recv(rank=rank-1)
        output = layers_i_to_j(input)
        send(output, rank=rank+1)
```

### 6.3 Sequence Parallelism (SP)
```python
# Sequence dimension split across ranks
# Used with TP for activation memory reduction
# Sequence-parallel regions: LayerNorm, Dropout
x_sp = scatter(x, dim=seq_dim)  # each rank has S/SP tokens
# Tensor-parallel regions: Attention, MLP
x_full = all_gather(x_sp, dim=seq_dim)
```

---

## 7. Memory Optimization Patterns

### 7.1 Activation Checkpointing
```python
def checkpointed_layer(x):
    # Don't save activations, recompute in backward
    return torch.utils.checkpoint.checkpoint(layer, x)
```

### 7.2 KV Cache (Inference)
```python
class KVCache:
    def __init__(self, max_len, num_layers):
        # Pre-allocate cache
        self.k_cache = torch.zeros(num_layers, max_len, ...)
        self.v_cache = torch.zeros(num_layers, max_len, ...)
        self.seq_len = 0

    def update(self, layer_id, k_new, v_new):
        start = self.seq_len
        end = start + k_new.shape[1]
        self.k_cache[layer_id, start:end] = k_new
        self.v_cache[layer_id, start:end] = v_new
        self.seq_len = end

    def get(self, layer_id):
        return self.k_cache[layer_id, :self.seq_len], \
               self.v_cache[layer_id, :self.seq_len]
```

### 7.3 Paged Attention (vLLM)
```python
# KV cache split into fixed-size blocks
# Virtual pages mapped to physical blocks
class PagedKVCache:
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.k_blocks = torch.zeros(num_blocks, block_size, ...)
        self.v_blocks = torch.zeros(num_blocks, block_size, ...)

    def access(self, block_table, positions):
        # block_table: [batch, max_blocks] - maps logical to physical
        physical_blocks = block_table[positions // self.block_size]
        offsets = positions % self.block_size
        return self.k_blocks[physical_blocks, offsets]
```

---

## 8. Tiling Patterns

### 8.1 2D Tiling (GEMM)
```
     K
   ┌─────────────┐
   │             │
 M │   A         │ × K×N Matrix B = M×N Matrix C
   │             │
   └─────────────┘

Tiled:
   BLOCK_K
   ┌───┐
   │   │ BLOCK_M
   └───┘

Each thread block computes BLOCK_M × BLOCK_N output tile
Loads BLOCK_M × BLOCK_K of A, BLOCK_K × BLOCK_N of B
```

### 8.2 Flash Attention Tiling
```
Q tiled along M dimension (query positions)
K, V tiled along N dimension (key positions)

For each Q block (BLOCK_M):
    For each K, V block (BLOCK_N):
        - Load K block
        - Compute partial attention scores
        - Update online softmax state
        - Load V block
        - Accumulate weighted values
```

### 8.3 Split-K Pattern
```python
# For memory-bound GEMMs, split K dimension
# Partial results combined at end
for k_split in range(num_splits):
    partial[k_split] = A[:, k_split*K_SPLIT:(k_split+1)*K_SPLIT] @ \
                       B[k_split*K_SPLIT:(k_split+1)*K_SPLIT, :]
output = sum(partial)  # reduction
```

---

## 9. Synchronization Patterns

### 9.1 Scoreboard-Based Dependencies
```python
# Producer-consumer synchronization via shared scoreboard
# Producer:
compute_output(tile_id)
scoreboard[layer_id, task_id, tile_id] = 1  # signal completion

# Consumer:
while scoreboard[dep_layer, dep_task, dep_tiles] != 1:
    wait()  # spin or yield
consume_input()
```

### 9.2 Barrier Synchronization
```python
# All threads reach barrier before continuing
__syncthreads()  # CUDA block-level
torch.distributed.barrier()  # Process-level
```

### 9.3 Double Buffering
```python
# Overlap compute with next load
buffer = [alloc(), alloc()]
load(buffer[0], data[0])

for i in range(1, N):
    load_async(buffer[i % 2], data[i])  # load next
    compute(buffer[(i-1) % 2])           # compute current
    sync()                                # wait for load

compute(buffer[(N-1) % 2])  # last iteration
```

---

## 10. Fusion Patterns

### 10.1 Attention + RoPE Fusion
```python
# Instead of:
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
attn_out = attention(q, k, v)

# Fused:
attn_out = fused_rope_attention(q, k, v, cos, sin)
```

### 10.2 Linear + Bias + Activation Fusion
```python
# Instead of:
x = linear(x)
x = x + bias
x = activation(x)

# Fused into single kernel:
x = fused_linear_bias_activation(x, weight, bias)
```

### 10.3 QKV Projection Fusion
```python
# Instead of 3 separate projections:
q = x @ w_q
k = x @ w_k
v = x @ w_v

# Single matmul with concatenated weights:
qkv = x @ w_qkv  # w_qkv = [w_q; w_k; w_v]
q, k, v = split(qkv, dim=-1)
```

---

*Reference for LLM Code Analysis Skill*
