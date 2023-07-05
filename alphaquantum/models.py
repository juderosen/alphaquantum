import chex
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Sequence, Tuple, Union, Optional, List, Callable
import haiku as hk
import optax

from alphaquantum.qtools import states
from alphaquantum.envs.fidelity_game import TaskSpec, Observation

from rich import print


ObservationBatch = Sequence[Observation]  # WARN: is the batch_size always 1?


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


@chex.dataclass(frozen=True)
class RepresentationHyperparams:
    head_depth: int
    num_heads: int
    attention_dropout: bool
    position_encoding: str
    activation_function: Callable  # relu replacement for complex-valued NNs


@chex.dataclass(frozen=True)
class Hyperparams:
    representation: RepresentationHyperparams
    attention_num_layers: int


class MultiQueryAttentionBlock(hk.Module):
    """Attention with multiple query heads and a single shared key and value head."""

    def __init__(
        self,
        head_depth: int = 128,
        num_heads: int = 4,
        attention_dropout: bool = False,
        position_encoding: str = "absolute",
        causal_mask: bool = False,
        name="multiquery_attention_block",
    ):
        super().__init__(name=name)
        self.head_depth = head_depth
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.position_encoding = position_encoding
        self.causal_mask = causal_mask

        # self.multihead =

    def __call__(self, x):
        return x

    def sinusoid_position_encoding(self, seq_size, feat_size):
        big_number = 10_000
        pass

    def _multi_query_attention(self):
        pass


class RepresentationNet(hk.Module):
    """Representation network"""

    def __init__(
        self,
        task_spec: TaskSpec,
        embedding_dim: int,
        hparams: Hyperparams,
        name: str = "representation",
    ):
        super().__init__(name=name)
        self._task_spec = task_spec
        self._embedding_dim = embedding_dim
        self._hparams = hparams

    def __call__(self, inputs: Observation):
        batch_size = 1  # inputs.program.shape[1]

        return self._encode_program(inputs, batch_size)

    def _encode_program(self, inputs: Observation, batch_size: int):
        program = inputs.program
        max_program_size = inputs.program.shape[0] # pyright: ignore
        program_length = inputs.program_length
        program_onehot = self.make_program_onehot(program, batch_size, max_program_size)
        program_encoding = self.apply_program_mlp_embedder(program_onehot)
        # program_encoding = self.apply_program_attention_embedder(program_onehot)
        # TODO: padding?
        return program_encoding

    def make_program_onehot(self, program, batch_size, max_program_size):
        return jax.nn.one_hot(
            program, self._task_spec.num_actions
        )  # negative values are mapped to 0-rows

    def apply_program_mlp_embedder(self, program_encoding):
        program_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=False, create_offset=False),
                self._hparams.representation.activation_function,
                hk.Linear(self._embedding_dim),
            ],
            name="per_instruction_program_embedder",
        )
        program_encoding = program_embedder(program_encoding)
        return program_encoding

    def apply_program_attention_embedder(self, program_encoding):
        attention_params = self._hparams.representation
        ake_attention_block = partial(
            MultiQueryAttentionBlock, attention_params, causal_mask=False
        )
        pass


class PredictionNet(hk.Module):
    """MuZero prediction network."""

    def __init__(
        self,
        task_spec: TaskSpec,
        value_max: float,
        value_num_bins: int,
        embedding_dim: int,
        name: str = "prediction",
    ):
        super().__init__(name=name)
        pass

    def __call__(self, x):
        pass


# testing stuff
if __name__ == "__main__":
    ghz3 = states.ket2dm(states.ghz_state(2))
    # ghz3_k = states.ghz_state(2)

    rng_key = jax.random.PRNGKey(42)
    task_spec_ghz3 = TaskSpec(
        max_program_size=10,
        num_wires=3,
        num_actions=jnp.int8(6),
        fidelity_reward_weight=1.0,
        time_reward_weight=0.0,
        goal_state=states.ket2dm(states.ghz_state(3)),
        possible_actions=jnp.array(
            [
                jnp.kron(states.hadamard(), jnp.kron(jnp.eye(2), jnp.eye(2))),  # H_0
                jnp.kron(jnp.kron(jnp.eye(2), states.hadamard()), jnp.eye(2)),  # H_1
                jnp.kron(jnp.kron(jnp.eye(2), jnp.eye(2)), states.hadamard()),  # H_2
                states.CNOT(3, 0, 1),  # CNOT_01
                states.CNOT(3, 0, 2),  # CNOT_02
                states.CNOT(3, 1, 2),  # CNOT_12
            ]
        ),
        fidelity_threshold=0.95,
    )  # pyright: ignore reportUnknownArgumentType

    # batched_states = jnp.array([ghz3])
    # print(batched_states)
    # test_forward = hk.transform(lambda x: StateEncoder()(x))
    # params = test_forward.init(rng_key, batched_states)
    # print(params)

    # pred = test_forward.apply(params, rng_key, batched_states)
    # print(pred)
    # print(batched_states.shape, pred.shape)

    test_program = jnp.array([0, 3, 5, -1, -1, -1, -1, -1, -1, -1])
    test_obs = Observation(  # pyright: ignore reportUnknownArgumentType
        state=ghz3,
        program=test_program,
        program_length=3,
    )
    # batch_obs = [test_obs, test_obs]
    # print(test_obs.program.shape)  # pyright: ignore
    # onehot = jax.nn.one_hot(test_program, 6)
    # print(onehot)
    # print(onehot.shape)
    hparams = RepresentationHyperparams(  # pyright: ignore reportUnknownArgumentType
        head_depth=20,
        num_heads=4,
        attention_dropout=False,
        position_encoding="?",
        activation_function=cardiod,
    )

    def pre(x):
        nn = RepresentationNet(task_spec_ghz3, 6, hparams)
        return nn(x)

    forward = hk.transform(pre)
    params = forward.init(rng_key, test_obs)
    pred = forward.apply(params, rng_key, test_obs)
    print(pred)
