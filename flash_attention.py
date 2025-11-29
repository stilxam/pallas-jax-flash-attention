import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from functools import partial
import equinox as eqx

def flash_attention_kernel(
        q_ref, k_ref, v_ref, o_ref,
        *,
        sm_scale, block_q, block_k, head_dim, seq_len
):
    pid_q = pl.program_id(0)
    start_q = pid_q * block_q
    
    row_idx = (jnp.arange(block_q) + start_q)[:, None]

    q = pl.load(q_ref, (slice(None), slice(None)))

    acc_o = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
    m_i = jnp.full((block_q,), -float("inf"), dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)

    num_k_blocks = seq_len // block_k

    def body_fn(i, val):
        acc_o, m_i, l_i = val
        
        start_k = i * block_k
        
        k_chunk = pl.load(k_ref, (pl.dslice(start_k, block_k), slice(None)))
        v_chunk = pl.load(v_ref, (pl.dslice(start_k, block_k), slice(None)))

        att_scores = jnp.dot(q, k_chunk.T) * sm_scale

        col_idx = (jnp.arange(block_k) + start_k)[None, :]
        mask = col_idx > row_idx
        att_scores = jnp.where(mask, -float("inf"), att_scores)

        # Online Softmax Update
        m_curr = jnp.max(att_scores, axis=1)
        m_new = jnp.maximum(m_i, m_curr) 
        
        alpha = jnp.exp(m_i - m_new)
        
        p_f32 = jnp.exp(att_scores - m_new[:, None])
        
        p_f16 = p_f32.astype(q.dtype)

        acc_o = acc_o * alpha[:, None] + jnp.dot(p_f16, v_chunk)
        
        row_sum_p = jnp.sum(p_f32, axis=1)
        l_i = l_i * alpha + row_sum_p
        
        return acc_o, m_new, l_i

    acc_o, m_i, l_i = jax.lax.fori_loop(0, num_k_blocks, body_fn, (acc_o, m_i, l_i))

    acc_o = acc_o / l_i[:, None]

    pl.store(o_ref, (slice(None), slice(None)), acc_o.astype(o_ref.dtype))


@eqx.filter_jit
def flash_attn_head(q, k, v, sm_scale=1.0, block_q=128, block_k=128):
    seq_len, head_dim = q.shape
    
    grid = (seq_len // block_q,)

    q_spec = pl.BlockSpec((block_q, head_dim), lambda i: (i, 0))
    kv_spec = pl.BlockSpec((seq_len, head_dim), lambda _: (0, 0))
    o_spec = pl.BlockSpec((block_q, head_dim), lambda i: (i, 0))

    return pl.pallas_call(
            partial(flash_attention_kernel, 
                    sm_scale=sm_scale, 
                    block_q=block_q, 
                    block_k=block_k,
                    head_dim=head_dim, 
                    seq_len=seq_len),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            grid=grid,
            in_specs=[q_spec, kv_spec, kv_spec], 
            out_specs=o_spec,
            interpret=True,
            )(q, k, v)


def flash_attention(q, k, v):
    heads, seq_len, dim = q.shape
    scale = 1.0 / (dim ** 0.5)
    
    BLOCK_Q = 128
    BLOCK_K = 128
    
    flash_fn = jax.vmap(
        partial(flash_attn_head, sm_scale=scale, block_q=BLOCK_Q, block_k=BLOCK_K)
    )
    
    return flash_fn(q, k, v)


if __name__ == "__main__":
    key = jax.random.key(0)
    B, H, S, D = 10, 4, 1024, 64

    q = jax.random.normal(key, (B, H,S,D), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (B, H,S,D), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (B, H,S,D), dtype=jnp.bfloat16)

    print(f"Running FlashAttention with [Heads={H}, Seq={S}, Dim={D}]...")
    out = jax.vmap(flash_attention)(q, k, v).block_until_ready()
    print("Execution successful. Shape:", out.shape)

    # Basic Numerical Check
    def manual_attn_ref(q, k, v):
        q, k, v = map(lambda x: x.astype(jnp.float32), (q, k, v))
        scale = 1.0 / (D ** 0.5)
        scores = jnp.einsum('hsd,hld->hsl', q, k) * scale
        mask = jnp.tril(jnp.ones((S, S)))
        scores = jnp.where(mask, scores, -float('inf'))
        probs = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum('hsl,hld->hsd', probs, v).astype(jnp.float16)

    out_ref = jax.vmap(manual_attn_ref)(q, k, v)
    err = jnp.max(jnp.abs(out - out_ref))
    print(f"Max Error vs Reference: {err}")
