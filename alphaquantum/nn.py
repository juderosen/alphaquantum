import chex
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
from typing import Optional


# Complex activation functions
@jit
def crelu(x: chex.Array) -> chex.Array:
    """Complex ReLU activation function
    Arguments:
        x: JAX array input of complex dtype

    Returns:
        Complex ReLU applied to Re and Im parts separately
    """
    # chex.assert_type(x, jnp.complex64)
    re = jnp.real(x)
    im = jnp.imag(x)
    return jnp.maximum(re, 0) + jnp.maximum(im, 0) * 1j


@jit
def cardiod(x: chex.Array) -> chex.Array:
    """Cardiod activation function. Similar to crelu.
    Arguments:
        x: JAX array input of complex dtype

    Returns:
        Cardiod activation
    """
    arg = jnp.angle(x)
    return 0.5 * (1 + jnp.cos(arg)) * x


class ResBlockV2(hk.Module):
    """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027."""

    def __init__(self, embedding_dim: int, name: str = "res_block_v2"):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        # TODO: figure out for this (and all other layer norms) if I should go with create_* T/F

    def __call__(self, x):
        out = x
        block = hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Conv1D(
                    output_channels=1, kernel_shape=3, with_bias=False
                ),  # TODO: params??
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Conv1D(
                    output_channels=1, kernel_shape=3, with_bias=False
                ),  # TODO: Conv1D vs Conv2d??
            ]
        )
        out = block(out)
        return x + out


class MultiQueryAttention(hk.Module):
    """MQA variant of built-in haiku MHA module https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py#L37#L165"""

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init: hk.initializers.Initializer,
        with_bias: bool = True,
        b_init: Optional[hk.initializers.Initializer] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        self.w_init = w_init
        self.with_bias = with_bias
        self.b_init = b_init

    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        *leading_dims, sequence_length, _ = query.shape  # pyright: ignore
        projection = self._linear_projection

        query_heads = projection(query, self.key_size, "query")

        attn_logits = jnp.einsum("...thd,...Td->...htT", query_heads, key)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        atnn_weights = jax.nn.softmax(attn_logits)

        attn = jnp.einsum("...htT,...Td->...thd", atnn_weights, value)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))

        final_projection = hk.Linear(
            self.model_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init,
        )
        return final_projection(attn)

    @hk.transparent
    def _linear_projection(
        self,
        x: chex.Array,
        head_size: int,
        name: Optional[str] = None,
    ) -> chex.Array:
        y = hk.Linear(
            self.num_heads * head_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init,
            name=name,
        )(x)  # pyright: ignore
        *leading_dims, _ = x.shape  # pyright: ignore
        return y.reshape((*leading_dims, self.num_heads, head_size))


class MultiQueryAttentionBlock(hk.Module):
    """Attention with multiple query heads and a single shared key and value head."""

    def __init__(
        self,
        head_depth: int = 128,
        num_heads: int = 4,
        attention_dropout: bool = False,
        position_encoding: str = "absolute",
        causal_mask: bool = False,
        name: str = "multiquery_attention_block",
    ):
        super().__init__(name=name)
        self.head_depth = head_depth
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.position_encoding = position_encoding
        self.causal_mask = causal_mask

        self.multi_query_attention = MultiQueryAttention(
            num_heads=self.num_heads,
            key_size=self.head_depth,
            w_init=hk.initializers.VarianceScaling(1.0),
            model_size=self.head_depth,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.multi_query_attention(x, x, x)

    @staticmethod
    def sinusoid_position_encoding(seq_size, feat_size, dtype=np.float32):
        """Implementation shamelessly stolen from
        https://github.com/google-research/perceiver-ar/blob/main/perceiver_ar/perceiver_ar_model.py
        """
        # TODO: How does complex dtype affect this position encoding?
        # we don't need it to be complex for action embedding, right?
        # but then how does it work when we concatenate everything later?
        # Perhaps the solution is to go back to the drawing board on the state
        # representation network and use 2N real numbers instead of N complex numbers?
        min_scale = 1.0
        max_scale = 10_000

        pe = np.zeros((seq_size, feat_size), dtype=dtype)
        position = np.arange(0, seq_size)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (feat_size // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, feat_size // 2) * scale_factor)
        pe[:, : feat_size // 2] = np.sin(position * div_term)
        pe[:, feat_size // 2 : 2 * (feat_size // 2)] = np.cos(position * div_term)
        return jnp.array(pe)
        # PERF: would this be faster if we rewrote it using jnp and partial JITed it? requires perf testing

    def _multi_query_attention(self):
        pass


if __name__ == "__main__":
    one_hot = jax.nn.one_hot(jnp.array([0, 2, 3]), 5)
    rng_key = jax.random.PRNGKey(0)
    forward = hk.transform(
        lambda q, k, v: MultiQueryAttention(
            2, 5, w_init=hk.initializers.VarianceScaling(1), model_size=5
        )(q, k, v)
    )
    params = forward.init(rng_key, one_hot, one_hot, one_hot)
    pred = forward.apply(params, rng_key, one_hot, one_hot, one_hot)

    forward_mha = hk.transform(
        lambda q, k, v: hk.MultiHeadAttention(
            2, 5, w_init=hk.initializers.VarianceScaling(1), model_size=5
        )(q, k, v)
    )
    params_mha = forward_mha.init(rng_key, one_hot, one_hot, one_hot)
    pred_mha = forward_mha.apply(params_mha, rng_key, one_hot, one_hot, one_hot)

    print("MQA Result:")
    print(pred, pred.shape)
    print("MHA Result:")
    print(pred_mha, pred_mha.shape)
