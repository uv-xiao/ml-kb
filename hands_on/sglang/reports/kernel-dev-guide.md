# SGLang Kernel Development Guide

**Generated:** 2026-01-09
**Purpose:** Guide for understanding and profiling SGLang kernels

---

## Overview

This guide provides:
1. Execution flow diagrams with code positions
2. Kernel-level analysis methodology
3. Profiling commands for each component
4. Hardware behavior interpretation

---

## Part 1: Request Execution Flow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REQUEST LIFECYCLE DIAGRAM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLIENT                                                                      │
│    │                                                                         │
│    │  POST /v1/chat/completions                                             │
│    ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TOKENIZER MANAGER                                 │    │
│  │                                                                      │    │
│  │  File: python/sglang/srt/managers/tokenizer_manager.py              │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ def handle_request(req):                                       │ │    │
│  │  │     tokens = self.tokenizer.encode(req.text)  # CPU            │ │    │
│  │  │     send_to_scheduler(TokenizedGenerateReqInput(...))          │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│                                      │ ZMQ IPC                               │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SCHEDULER                                     │    │
│  │                                                                      │    │
│  │  File: python/sglang/srt/managers/scheduler.py                      │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ def event_loop_overlap():              # Main loop             │ │    │
│  │  │     recv_reqs = self.recv_requests()   # Get from queue        │ │    │
│  │  │     self.process_input_requests()      # Add to waiting_queue  │ │    │
│  │  │     batch = self.get_next_batch_to_run()                       │ │    │
│  │  │     result = self.run_batch(batch)     # ─► GPU Forward        │ │    │
│  │  │     self.process_batch_result(batch, result)                   │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                      │    │
│  │  Key Methods:                                                        │    │
│  │  • get_next_batch_to_run()  → Line ~1600                            │    │
│  │  • run_batch()              → Line ~1750                            │    │
│  │  • handle_generate_request() → Line ~1423                           │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       TP MODEL WORKER                                │    │
│  │                                                                      │    │
│  │  File: python/sglang/srt/managers/tp_worker.py                      │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ def forward_batch_generation(batch):                           │ │    │
│  │  │     forward_batch = ForwardBatch.init_new(batch, self.runner)  │ │    │
│  │  │     return self.model_runner.forward(forward_batch)            │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       MODEL RUNNER                                   │    │
│  │                                                                      │    │
│  │  File: python/sglang/srt/model_executor/model_runner.py             │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │ def forward(forward_batch):                                    │ │    │
│  │  │     # Initialize attention metadata                            │ │    │
│  │  │     self.attn_backend.init_forward_metadata(forward_batch)     │ │    │
│  │  │                                                                │ │    │
│  │  │     # Run model forward                                        │ │    │
│  │  │     logits = self.model.forward(                               │ │    │
│  │  │         input_ids,                                             │ │    │
│  │  │         positions,                                             │ │    │
│  │  │         forward_batch,                                         │ │    │
│  │  │         input_embeds                                           │ │    │
│  │  │     )                                                          │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MODEL FORWARD (Llama Example)                     │    │
│  │                                                                      │    │
│  │  File: python/sglang/srt/models/llama.py                            │    │
│  │                                                                      │    │
│  │  for layer in self.layers:                                          │    │
│  │      ┌──────────────────────────────────────────────────────────┐   │    │
│  │      │  1. RMSNorm              (memory-bound kernel)           │   │    │
│  │      │  2. QKV Projection       (GEMM, TC-heavy)                │   │    │
│  │      │  3. RoPE                 (memory-bound kernel)           │   │    │
│  │      │  4. Attention            (FlashInfer/Triton)  ◄──HOTSPOT │   │    │
│  │      │  5. O Projection         (GEMM + AllReduce)              │   │    │
│  │      │  6. RMSNorm              (memory-bound kernel)           │   │    │
│  │      │  7. Gate+Up Projection   (GEMM, TC-heavy)                │   │    │
│  │      │  8. SiLU Activation      (memory-bound kernel)           │   │    │
│  │      │  9. Down Projection      (GEMM + AllReduce)              │   │    │
│  │      └──────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  LM Head: logits = self.lm_head(hidden_states)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Attention Backend Architecture

### FlashInfer Backend Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FLASHINFER ATTENTION BACKEND                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  File: python/sglang/srt/layers/attention/flashinfer_backend.py             │
│                                                                              │
│  INITIALIZATION (Line ~112):                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  class FlashInferAttnBackend(AttentionBackend):                        │ │
│  │      def __init__(self, model_runner, ...):                            │ │
│  │          # Workspace buffer (shared across wrappers)                   │ │
│  │          global_workspace_buffer = torch.empty(128MB, dtype=uint8)     │ │
│  │                                                                         │ │
│  │          # KV index buffers                                            │ │
│  │          self.kv_indptr = torch.zeros((max_bs + 1,), dtype=int32)      │ │
│  │          self.kv_last_page_len = torch.ones((max_bs,), dtype=int32)    │ │
│  │                                                                         │ │
│  │          # Create wrappers                                              │ │
│  │          self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper()  │ │
│  │          self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper()    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  PREFILL FLOW (Line ~400):                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def init_forward_metadata(forward_batch):                             │ │
│  │      if forward_batch.forward_mode.is_extend():                        │ │
│  │          # Build KV indices for paged cache                            │ │
│  │          create_flashinfer_kv_indices_triton(                          │ │
│  │              req_to_token,                                              │ │
│  │              req_pool_indices,                                          │ │
│  │              seq_lens,                                                  │ │
│  │              kv_indptr,                                                 │ │
│  │              kv_indices  # Output: flattened KV indices                │ │
│  │          )                                                              │ │
│  │                                                                         │ │
│  │          # Plan prefill operation                                       │ │
│  │          self.prefill_wrapper.plan(                                     │ │
│  │              qo_indptr=qo_indptr,                                       │ │
│  │              kv_indptr=kv_indptr,                                       │ │
│  │              kv_indices=kv_indices,                                     │ │
│  │              ...                                                        │ │
│  │          )                                                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  DECODE FLOW (Line ~500):                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def init_forward_metadata(forward_batch):                             │ │
│  │      if forward_batch.forward_mode.is_decode():                        │ │
│  │          # Update KV indices (one new token per request)               │ │
│  │          kv_indptr[1:bs+1] = cumsum(seq_lens)                          │ │
│  │          create_flashinfer_kv_indices_triton(...)                      │ │
│  │                                                                         │ │
│  │          # Plan decode with tensor cores                               │ │
│  │          self.decode_wrapper.plan(                                      │ │
│  │              kv_indptr=kv_indptr,                                       │ │
│  │              kv_indices=kv_indices,                                     │ │
│  │              use_tensor_cores=self.decode_use_tensor_cores             │ │
│  │          )                                                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  FORWARD EXECUTION (Line ~700):                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def forward(q, k, v, layer, forward_batch):                           │ │
│  │      if forward_batch.forward_mode.is_extend():                        │ │
│  │          # Ragged attention for single-request batches                 │ │
│  │          if use_ragged:                                                 │ │
│  │              out = self.prefill_wrapper_ragged.forward(q, k, v)        │ │
│  │          else:                                                          │ │
│  │              out = self.prefill_wrapper.forward(q, kv_cache)           │ │
│  │      else:  # Decode                                                    │ │
│  │          out = self.decode_wrapper.forward(q, kv_cache)                │ │
│  │      return out                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Triton Backend Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRITON ATTENTION BACKEND                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  File: python/sglang/srt/layers/attention/triton_backend.py                 │
│                                                                              │
│  DECODE KERNEL (triton_ops/decode_attention.py):                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  @triton.jit                                                           │ │
│  │  def decode_attention_fwd_kernel(                                      │ │
│  │      Q, K, V, Out,                                                      │ │
│  │      kv_indptr, kv_indices,                                            │ │
│  │      ...                                                                │ │
│  │  ):                                                                     │ │
│  │      # Each thread block handles one (batch, head) pair                │ │
│  │      pid_batch = tl.program_id(0)                                      │ │
│  │      pid_head = tl.program_id(1)                                       │ │
│  │      pid_split = tl.program_id(2)  # Split-K                          │ │
│  │                                                                         │ │
│  │      # Load query (single token)                                       │ │
│  │      q = tl.load(Q + pid_batch * stride_qb + pid_head * stride_qh)     │ │
│  │                                                                         │ │
│  │      # Iterate over KV cache chunks                                    │ │
│  │      for kv_start in range(split_start, split_end, BLOCK_SIZE):        │ │
│  │          k = tl.load(K + kv_indices[kv_start:kv_end] * stride_kb)      │ │
│  │          v = tl.load(V + kv_indices[kv_start:kv_end] * stride_vb)      │ │
│  │                                                                         │ │
│  │          # Compute attention scores                                    │ │
│  │          scores = tl.dot(q, tl.trans(k)) * scale                       │ │
│  │          p = tl.softmax(scores)                                        │ │
│  │          out_partial += tl.dot(p, v)                                   │ │
│  │                                                                         │ │
│  │      # Store partial results for split-K reduction                     │ │
│  │      tl.store(Out_partial + ...)                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  EXTEND KERNEL (triton_ops/extend_attention.py):                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  @triton.jit                                                           │ │
│  │  def extend_attention_fwd_kernel(                                      │ │
│  │      Q, K, V, Out,                                                      │ │
│  │      qo_indptr, kv_indptr, kv_indices,                                 │ │
│  │      custom_mask, mask_indptr,  # For complex patterns                 │ │
│  │      ...                                                                │ │
│  │  ):                                                                     │ │
│  │      # Each thread block handles one (batch, head, q_tile) triple      │ │
│  │      pid = tl.program_id(0)                                            │ │
│  │      # ... batch/head/tile index computation                           │ │
│  │                                                                         │ │
│  │      # FlashAttention-style tiling                                     │ │
│  │      for kv_tile in range(num_kv_tiles):                               │ │
│  │          # Load KV tile from paged cache                               │ │
│  │          k_tile = load_kv_tile(K, kv_indices, kv_tile)                 │ │
│  │          v_tile = load_kv_tile(V, kv_indices, kv_tile)                 │ │
│  │                                                                         │ │
│  │          # Compute attention with online softmax                       │ │
│  │          scores = tl.dot(q_tile, tl.trans(k_tile)) * scale             │ │
│  │          # Apply mask if provided                                      │ │
│  │          if custom_mask:                                                │ │
│  │              scores += load_mask(...)                                   │ │
│  │                                                                         │ │
│  │          # Online softmax update                                       │ │
│  │          m_new = tl.maximum(m_prev, tl.max(scores))                    │ │
│  │          p = tl.exp(scores - m_new)                                    │ │
│  │          out = rescale(out_prev) + tl.dot(p, v_tile)                   │ │
│  │                                                                         │ │
│  │      tl.store(Out + ...)                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: RadixCache Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RADIX CACHE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  File: python/sglang/srt/mem_cache/radix_cache.py                           │
│                                                                              │
│  DATA STRUCTURE:                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          ROOT NODE                                     │ │
│  │                     (key=[], value=[])                                 │ │
│  │                            │                                           │ │
│  │         ┌──────────────────┼──────────────────┐                        │ │
│  │         │                  │                  │                        │ │
│  │         ▼                  ▼                  ▼                        │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │ │
│  │  │  TreeNode   │    │  TreeNode   │    │  TreeNode   │                │ │
│  │  │ key=[1,2,3] │    │ key=[4,5,6] │    │ key=[7,8,9] │                │ │
│  │  │ value=[KV]  │    │ value=[KV]  │    │ value=[KV]  │                │ │
│  │  │ lock_ref=0  │    │ lock_ref=2  │    │ lock_ref=0  │                │ │
│  │  └──────┬──────┘    └─────────────┘    └──────┬──────┘                │ │
│  │         │                                     │                        │ │
│  │         ▼                                     ▼                        │ │
│  │  ┌─────────────┐                       ┌─────────────┐                │ │
│  │  │ key=[10,11] │                       │ key=[12,13] │                │ │
│  │  │ value=[KV]  │                       │ value=[KV]  │                │ │
│  │  └─────────────┘                       └─────────────┘                │ │
│  │                                                                        │ │
│  │  key = RadixKey(token_ids, extra_key)                                 │ │
│  │  value = torch.Tensor of KV cache indices                             │ │
│  │  lock_ref > 0 = protected from eviction (in-use)                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  MATCH PREFIX (Line ~340):                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def match_prefix(self, key: RadixKey) -> MatchResult:                 │ │
│  │      """Find longest cached prefix in O(key_length) time."""          │ │
│  │                                                                         │ │
│  │      # Start at root                                                   │ │
│  │      node = self.root_node                                             │ │
│  │      matched_values = []                                                │ │
│  │                                                                         │ │
│  │      # Traverse tree following matching segments                       │ │
│  │      while len(key) > 0:                                               │ │
│  │          child_key = get_child_key(key)                                │ │
│  │          if child_key not in node.children:                            │ │
│  │              break  # No more matches                                  │ │
│  │                                                                         │ │
│  │          child = node.children[child_key]                              │ │
│  │          prefix_len = key_match(child.key, key)                        │ │
│  │                                                                         │ │
│  │          if prefix_len < len(child.key):                               │ │
│  │              # Partial match: SPLIT node                               │ │
│  │              new_node = split_node(child, prefix_len)                  │ │
│  │              matched_values.append(new_node.value)                     │ │
│  │              break                                                      │ │
│  │          else:                                                          │ │
│  │              matched_values.append(child.value)                        │ │
│  │              key = key[prefix_len:]                                    │ │
│  │              node = child                                               │ │
│  │                                                                         │ │
│  │      # Return concatenated KV indices                                  │ │
│  │      return MatchResult(                                                │ │
│  │          device_indices=torch.cat(matched_values),                     │ │
│  │          last_device_node=node                                         │ │
│  │      )                                                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  EVICTION (Line ~544):                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def evict(self, num_tokens: int):                                     │ │
│  │      """Evict tokens using configured policy (LRU, LFU, etc.)"""       │ │
│  │                                                                         │ │
│  │      # Collect evictable leaves (lock_ref == 0)                        │ │
│  │      leaves = self._collect_leaves()                                   │ │
│  │                                                                         │ │
│  │      # Build min-heap by eviction priority                             │ │
│  │      eviction_heap = [                                                  │ │
│  │          (self.eviction_strategy.get_priority(node), node)             │ │
│  │          for node in leaves                                             │ │
│  │      ]                                                                  │ │
│  │      heapq.heapify(eviction_heap)                                      │ │
│  │                                                                         │ │
│  │      # Pop and delete until enough freed                               │ │
│  │      num_evicted = 0                                                    │ │
│  │      while num_evicted < num_tokens and eviction_heap:                 │ │
│  │          _, node = heapq.heappop(eviction_heap)                        │ │
│  │          self.token_to_kv_pool_allocator.free(node.value)              │ │
│  │          num_evicted += len(node.value)                                │ │
│  │          self._delete_leaf(node)                                       │ │
│  │                                                                         │ │
│  │          # Check if parent became evictable                            │ │
│  │          if len(node.parent.children) == 0 and node.parent.lock_ref==0:│ │
│  │              heapq.heappush(eviction_heap, (..., node.parent))         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: NCCL Communication Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM COMMUNICATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  File: python/sglang/srt/distributed/parallel_state.py                      │
│  File: python/sglang/srt/distributed/communication_op.py                    │
│                                                                              │
│  ALLREDUCE PATTERN (Per Layer):                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  Attention Output:                                                      │ │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                            │ │
│  │  │ GPU 0 │  │ GPU 1 │  │ GPU 2 │  │ GPU 3 │                            │ │
│  │  │ O_0   │  │ O_1   │  │ O_2   │  │ O_3   │  (Partial outputs)         │ │
│  │  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘                            │ │
│  │      │          │          │          │                                 │ │
│  │      └──────────┴────┬─────┴──────────┘                                 │ │
│  │                      │ AllReduce(sum)                                   │ │
│  │      ┌──────────┬────┴─────┬──────────┐                                 │ │
│  │      ▼          ▼          ▼          ▼                                 │ │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                            │ │
│  │  │ O_sum │  │ O_sum │  │ O_sum │  │ O_sum │  (All have same result)    │ │
│  │  └───────┘  └───────┘  └───────┘  └───────┘                            │ │
│  │                                                                         │ │
│  │  Same pattern for FFN down projection                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  IMPLEMENTATION (Line ~169):                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  class GroupCoordinator:                                               │ │
│  │      def all_reduce(self, input_: torch.Tensor):                       │ │
│  │          """Reduce across all ranks."""                                │ │
│  │                                                                         │ │
│  │          # Backend selection based on tensor size                      │ │
│  │          if self.use_pynccl and input_.numel() > threshold:            │ │
│  │              # PyNCCL for large tensors                                │ │
│  │              self.pynccl_comm.all_reduce(input_, stream=stream)        │ │
│  │                                                                         │ │
│  │          elif self.use_custom_allreduce:                               │ │
│  │              # Custom AllReduce for small tensors                      │ │
│  │              # Uses shared memory for intra-node                       │ │
│  │              out = self.ca_comm.custom_all_reduce(input_)              │ │
│  │                                                                         │ │
│  │          else:                                                          │ │
│  │              # Fallback to PyTorch distributed                         │ │
│  │              torch.distributed.all_reduce(                              │ │
│  │                  input_,                                                │ │
│  │                  group=self.device_group                                │ │
│  │              )                                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  RING ALLREDUCE VISUALIZATION:                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                         │ │
│  │  Step 1: Scatter-Reduce (N-1 steps)                                    │ │
│  │                                                                         │ │
│  │    GPU0 ──chunk0──▶ GPU1 ──chunk1──▶ GPU2 ──chunk2──▶ GPU3             │ │
│  │      ▲                                                    │             │ │
│  │      └──────────────────chunk3────────────────────────────┘             │ │
│  │                                                                         │ │
│  │  Step 2: Allgather (N-1 steps)                                         │ │
│  │                                                                         │ │
│  │    GPU0 ◀─result─ GPU1 ◀─result─ GPU2 ◀─result─ GPU3                   │ │
│  │      │                                            ▲                     │ │
│  │      └────────────────result──────────────────────┘                     │ │
│  │                                                                         │ │
│  │  Data movement per GPU: 2 × (N-1)/N × data_size                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Profiling Commands by Component

### Attention Kernels

```bash
# Profile all attention kernels with NCU
ncu --set full \
    --kernel-regex ".*flash.*attention.*|.*decode.*attention.*" \
    --launch-count 20 \
    -o attention_profile \
    python benchmark.py

# Key metrics to examine:
# - sm__throughput.avg.pct_of_peak_sustained_elapsed  (SM utilization)
# - gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (HBM BW)
# - sm__sass_thread_inst_executed_op_ffma_pred_on.avg (Tensor core ops)
# - smsp__warp_stall_reasons (Bottleneck identification)
```

### GEMM Kernels

```bash
# Profile GEMM/cuBLAS kernels
ncu --set full \
    --kernel-regex ".*gemm.*|.*cublas.*" \
    --launch-count 20 \
    -o gemm_profile \
    python benchmark.py

# Key metrics:
# - Tensor core utilization
# - Memory efficiency
# - Occupancy
```

### Normalization Kernels

```bash
# Profile RMSNorm, LayerNorm
ncu --set full \
    --kernel-regex ".*norm.*|.*rms.*" \
    --launch-count 20 \
    -o norm_profile \
    python benchmark.py

# Expected: Memory-bound (>80% HBM, <30% SM)
```

### Communication (NCCL)

```bash
# Profile with NCCL traces
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=COLL \
    python -m sglang.launch_server --tp 2 ...

# Profile with nsys
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    -o nccl_trace \
    python tp_benchmark.py
```

### Full System Timeline

```bash
# Comprehensive nsys profile
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas,cusparse \
    --cuda-memory-usage=true \
    --gpuctxsw=true \
    --capture-range=cudaProfilerApi \
    -o full_trace \
    python benchmark.py
```

---

## Part 6: Hardware Behavior Reference

### A100 Kernel Performance Expectations

| Kernel Type | SM Util | TC Util | HBM BW | Bottleneck |
|-------------|---------|---------|--------|------------|
| Prefill Attention | 60-75% | 30-45% | 60-75% | Mixed |
| Decode Attention | 40-55% | 10-20% | 75-85% | Memory |
| QKV Projection | 75-85% | 40-50% | 40-55% | Compute |
| FFN (large) | 80-90% | 45-55% | 35-50% | Compute |
| RMSNorm | 20-35% | 0% | 80-90% | Memory |
| Embedding | 25-40% | 0% | 70-85% | Memory |

### Warp Stall Interpretation

| Stall Reason | Cause | Action |
|--------------|-------|--------|
| `long_scoreboard` | Waiting for HBM loads | Increase prefetching, improve tiling |
| `barrier` | Synchronization overhead | Reduce sync points, warp specialization |
| `short_scoreboard` | SMEM bank conflicts | Pad SMEM arrays, change access pattern |
| `not_selected` | Low occupancy | Increase parallelism, reduce register usage |
| `wait` | Memory dependency | Pipeline memory accesses |
| `math_pipe_throttle` | Compute backpressure | Expected for compute-bound kernels |

---

## Part 7: Quick Reference

### Key File Locations

| Component | Path |
|-----------|------|
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| TpWorker | `python/sglang/srt/managers/tp_worker.py` |
| ModelRunner | `python/sglang/srt/model_executor/model_runner.py` |
| FlashInfer Backend | `python/sglang/srt/layers/attention/flashinfer_backend.py` |
| Triton Backend | `python/sglang/srt/layers/attention/triton_backend.py` |
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` |
| ParallelState | `python/sglang/srt/distributed/parallel_state.py` |
| ForwardBatch | `python/sglang/srt/model_executor/forward_batch_info.py` |
| Llama Model | `python/sglang/srt/models/llama.py` |

### Environment Variables

```bash
# Attention backend selection
export SGLANG_ATTENTION_BACKEND=flashinfer  # or triton

# Profiling
export CUDA_LAUNCH_BLOCKING=0  # Keep 0 for accurate timing
export NVTX_ENABLED=1

# Debug
export SGLANG_LOG_LEVEL=DEBUG
```

---

## Next Steps

1. Run profiling scripts in `scripts/` directory
2. Collect results in `results/` directory
3. Compare with expected behavior in this guide
4. Document findings in `reports/experiments.md`
