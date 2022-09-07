from functools import partial
import math
import torch, gc
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from benchmarks.utils import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

import numpy as np

def attention_ref(qkv, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    print("ref is running...")
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)

torch.manual_seed(0)
repeats = 1
batch_size = 48
nheads = 16
seqlen = 256
d = 64             #head_dim 64
n = d * nheads     #embed size
dropout_p = 0.0
causal = False
dtype = torch.float16
device = 'cuda'

x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

lengths = torch.randint(seqlen, seqlen + 1, (batch_size, 1), device='cuda')
attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) < lengths
attention_mask = torch.zeros(batch_size, seqlen, device='cuda', dtype=dtype)
attention_mask[~attention_mask_bool] = -10000.0
attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask_bool)
qkv_unpad = rearrange(Wqkv(x_unpad), 'nnz (t h d) -> nnz t h d', t=3,
                      h=nheads).detach().requires_grad_()
qkv = rearrange(Wqkv(x), 'b s (t h d) -> b s t h d', t=3, h=nheads).detach().requires_grad_()

qkv_unpad_np = qkv_unpad.to('cpu').detach().numpy()
qkv_np = qkv.to('cpu').detach().numpy()
np.savetxt('qkv_unpad_np.txt', qkv_unpad_np.reshape(-1,1))
np.savetxt('qkv_np.txt', qkv_np.reshape(-1,1))

###q, k, v = qkv.unbind(dim=2)

q_np = rearrange(qkv_np[:,:,0,:,:], 'b s h d -> (b h) s d')
k_np = rearrange(qkv_np[:,:,1,:,:], 'b s h d -> (b h) s d')
v_np = rearrange(qkv_np[:,:,2,:,:], 'b s h d -> (b h) s d')

#q_transpose_np = q_np.transpose(0,1,3,2)
#k_transpose_np = k_np.transpose(0,1,3,2)
#v_transpose_np = v_np.transpose(0,1,3,2)

np.savetxt('q_in_row_v4.txt', q_np.reshape(-1,1))
np.savetxt('k_in_col_v4.txt', k_np.reshape(-1,1))
np.savetxt('v_in_row_v4.txt', v_np.reshape(-1,1))

fn = lambda qkv_unpad: flash_attn_unpadded_qkvpacked_func(
    qkv_unpad, cu_seqlens, max_seqlen_in_batch, dropout_p, causal=causal
)
out = flash_attn_unpadded_qkvpacked_func(
    qkv_unpad, cu_seqlens, max_seqlen_in_batch, dropout_p, causal=causal
)

out_np = out.to('cpu').data.reshape(batch_size, seqlen, nheads, d).numpy()
print(out_np.shape)
#np.savetxt('out_result_cuda_v3.txt', out_np.reshape(-1,1))

out_np2 = rearrange(out_np, 'b s h d -> (b h) s d')
np.savetxt('out_result_cuda_v4.txt', out_np2.reshape(-1,1))

#benchmark_forward(fn, qkv_unpad, repeats=repeats, desc='FlashAttention')
fn = lambda qkv: attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)

out_ref = attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)


out_ref_np = out_ref.to('cpu').data.numpy()
#np.savetxt('out_result_ref_cuda_v3.txt', out_ref_np.reshape(-1,1))

out_ref_np2 = rearrange(out_ref_np, 'b s h d -> (b h) s d')
np.savetxt('out_result_ref_cuda_v4.txt', out_ref_np2.reshape(-1,1))

#benchmark_forward(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')

