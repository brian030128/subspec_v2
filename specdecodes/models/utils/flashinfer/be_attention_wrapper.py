"""
FlashInfer attention wrapper using the new .plan() / .run() API.

The shared FlashinferAttentionWrapper (attention_wrapper.py) uses the old
.begin_forward() / .forward() API which is incompatible with newer FlashInfer
versions.  This module provides the same interface but calls .plan() / .run()
internally so that be_classic_sd_fi works on machines with the newer library.
"""

import math
from typing import Optional

import torch
import flashinfer

from .cache_manager import KvCacheBatchPosition
from .attention_wrapper import (
    AttentionRotaryParams,
    find_padded_head_dim,
    FLASH_INFER_SUPPORTED_DIMS,
    POS_ENCODING_MODE,
)


class BeFlashinferWrapper:
    """Same public interface as FlashinferAttentionWrapper, new FlashInfer API."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        page_len: int,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self._head_padded_dim = find_padded_head_dim(self.head_dim)
        self.page_len = page_len

        self.group_size = self.num_attention_heads // self.num_key_value_heads
        _workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.int8, device=torch.cuda.current_device()
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer, kv_layout="NHD",
        )
        _use_tensor_cores = self.group_size in [7, 16]
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=_use_tensor_cores,
        )

        self.tree_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer, kv_layout="NHD"
        )

        # for cuda graph
        batch_size = 1

        self.qo_indptr_buf = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )
        self.paged_kv_indptr_buf = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )
        self.paged_kv_indices_buf = torch.zeros(
            1024,
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )
        self.paged_kv_last_page_len_buf = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )
        self.custom_mask_buf = torch.zeros(
            100000 * 1000,
            dtype=torch.uint8,
            device=torch.cuda.current_device()
        )
        self.mask_indptr_buf = torch.zeros(
            (batch_size + 1,),
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )

        self.tree_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self.qo_indptr_buf,
            paged_kv_indptr_buf=self.paged_kv_indptr_buf,
            paged_kv_indices_buf=self.paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf,
            custom_mask_buf=self.custom_mask_buf,
            mask_indptr_buf=self.mask_indptr_buf,
        )

        # Pre-allocated output buffer for decode (avoids per-call allocation)
        self._decode_output_buf = None

    # ------------------------------------------------------------------
    # prepareAttention  — uses .plan() instead of .begin_forward()
    # ------------------------------------------------------------------
    def prepareAttention(
        self,
        mode: str,
        batch_position: KvCacheBatchPosition,
        page_len: int,
        pos_encoding_mode: POS_ENCODING_MODE,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if mode == "tree" and attention_mask is not None:
            self.tree_wrapper.plan(
                qo_indptr=batch_position.seq_indptr,
                paged_kv_indptr=batch_position.kv_page_indptr,
                paged_kv_indices=batch_position.kv_page_indices,
                paged_kv_last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim_qk=self._head_padded_dim,
                page_size=page_len,
                custom_mask=attention_mask,
                causal=False,
            )
        elif mode == "tree" and attention_mask is None:
            self.tree_wrapper.plan(
                qo_indptr=batch_position.seq_indptr,
                paged_kv_indptr=batch_position.kv_page_indptr,
                paged_kv_indices=batch_position.kv_page_indices,
                paged_kv_last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim_qk=self._head_padded_dim,
                page_size=page_len,
                causal=True,
            )
        elif mode == "prefill":
            self.prefill_wrapper.plan(
                qo_indptr=batch_position.seq_indptr,
                paged_kv_indptr=batch_position.kv_page_indptr,
                paged_kv_indices=batch_position.kv_page_indices,
                paged_kv_last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim_qk=self._head_padded_dim,
                page_size=page_len,
                causal=True,
            )
        elif mode == "decode":
            self.decode_wrapper.plan(
                indptr=batch_position.kv_page_indptr,
                indices=batch_position.kv_page_indices,
                last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self._head_padded_dim,
                page_size=page_len,
                data_type=dtype,
            )
        else:
            raise ValueError("the mode for attention must be prefill, decode or tree")

    # ------------------------------------------------------------------
    # reshape / pad / unpad  — identical to FlashinferAttentionWrapper
    # ------------------------------------------------------------------
    def reshape_qkv_for_attention(self, q, k, v, batchPosition: KvCacheBatchPosition):
        return (
            q.view(-1, self.num_attention_heads, self.head_dim),
            k.view(-1, self.num_key_value_heads, self.head_dim),
            v.view(-1, self.num_key_value_heads, self.head_dim),
        )

    def _unpad_attention(self, attn_output):
        if self._head_padded_dim > self.head_dim:
            return attn_output[:, :, : self.head_dim].reshape(-1, self.hidden_size)
        else:
            return attn_output.view(-1, self.hidden_size)

    def _pad_qkv(self, q, k, v):
        if self._head_padded_dim > self.head_dim:
            q = torch.nn.functional.pad(q, (0, self._head_padded_dim - self.head_dim))
            k = torch.nn.functional.pad(k, (0, self._head_padded_dim - self.head_dim))
            v = torch.nn.functional.pad(v, (0, self._head_padded_dim - self.head_dim))
        return q, k, v

    # ------------------------------------------------------------------
    # append_kv_cache  — identical to FlashinferAttentionWrapper
    # ------------------------------------------------------------------
    def append_kv_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_position,
        paged_kv_cache: torch.Tensor,
        page_len: int,
    ):
        flashinfer.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_position.batch_indices,
            paged_kv_cache=paged_kv_cache,
            kv_indices=batch_position.kv_page_indices,
            positions=batch_position.positions,
            kv_indptr=batch_position.kv_page_indptr,
            kv_last_page_len=batch_position.kv_last_page_len,
        )

    # ------------------------------------------------------------------
    # computeAttention  — identical dispatch, different internals
    # ------------------------------------------------------------------
    def computeAttention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        mode: str,
        batchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
        layer_idx: int,
    ):
        q, k, v = self._pad_qkv(q, k, v)
        if mode == 'prefill':
            attn_output = self._batchPrefill(q, k, v, cacheData, batchPosition, rotaryParams)
        elif mode == 'decode':
            attn_output = self._batchDecode(q, k, v, cacheData, batchPosition, rotaryParams)
        elif mode == 'tree':
            attn_output = self._treeDecode(q, k, v, cacheData, batchPosition, rotaryParams)

        return self._unpad_attention(attn_output)

    # ------------------------------------------------------------------
    # _batchPrefill  — uses .run() instead of .forward()
    # ------------------------------------------------------------------
    def _batchPrefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        prefillBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        self.append_kv_cache(q, k, v, prefillBatchPosition, cacheData, self.page_len)
        attn_output_prefill = self.prefill_wrapper.run(q, cacheData)
        return attn_output_prefill

    # ------------------------------------------------------------------
    # _treeDecode  — uses .run() instead of .forward()
    # ------------------------------------------------------------------
    def _treeDecode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        treeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        self.append_kv_cache(q, k, v, treeBatchPosition, cacheData, self.page_len)
        attn_output = self.tree_wrapper.run(q, cacheData)
        return attn_output

    # ------------------------------------------------------------------
    # _batchDecode  — uses .run() with pre-allocated output buffer
    # ------------------------------------------------------------------
    def _batchDecode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        decodeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        self.append_kv_cache(q, k, v, decodeBatchPosition, cacheData, self.page_len)

        # Reuse pre-allocated output buffer to avoid per-call allocation
        buf = self._decode_output_buf
        if buf is None or buf.shape[0] < q.shape[0]:
            buf = torch.empty_like(q)
            self._decode_output_buf = buf
        out = buf[: q.shape[0]]

        self.decode_wrapper.run(q, cacheData, out=out)
        return out
