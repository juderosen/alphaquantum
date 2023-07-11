import chex
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Sequence, Tuple, Union, Optional, List, Callable, Dict, Any
import haiku as hk
#import optax

from alphaquantum.qtools import states
from alphaquantum.envs.fidelity_game import TaskSpec, Observation, Action, Program
from alphaquantum.nn import ResBlockV2, MultiQueryAttentionBlock

from rich import print


ObservationBatch = Sequence[Observation]  # WARN: is the batch_size always 1?
# TODO: Understand batching/batch_size in alphadev algorithm


# # Complex activation functions
# @jit
# def crelu(x: chex.Array) -> chex.Array:
#     """Complex ReLU activation function
#     Arguments:
#         x: JAX array input of complex dtype
#
#     Returns:
#         Complex ReLU applied to Re and Im parts separately
#     """
#     # chex.assert_type(x, jnp.complex64)
#     re = jnp.real(x)
#     im = jnp.imag(x)
#     return jnp.maximum(re, 0) + jnp.maximum(im, 0) * 1j
#
#
# @jit
# def cardiod(x: chex.Array) -> chex.Array:
#     """Cardiod activation function. Similar to crelu.
#     Arguments:
#         x: JAX array input of complex dtype
#
#     Returns:
#         Cardiod activation
#     """
#     arg = jnp.angle(x)
#     return 0.5 * (1 + jnp.cos(arg)) * x


@chex.dataclass(frozen=True)
class AttentionHyperparams:
    head_depth: int
    num_heads: int
    attention_dropout: bool
    position_encoding: str
    num_layers: int


@chex.dataclass(frozen=True)
class RepresentationHyperparams:
    attention: AttentionHyperparams
    repr_net_res_blocks: int
    # activation_function: Callable  # relu replacement for complex-valued NNs


@chex.dataclass(frozen=True)
class ValueHyperparams:
    num_bins: int
    max: float


@chex.dataclass(frozen=True)
class Hyperparams:
    representation: RepresentationHyperparams
    attention_num_layers: int
    embedding_dim: int
    value: ValueHyperparams


@chex.dataclass(frozen=True)
class NetworkOutput:
    value: float
    correctness_value_logits: chex.Array
    latency_value_logits: chex.Array
    policy_logits: Dict[Action, float]


class Network(object):
    """Wrapper around Representation and Prediction networks."""

    def __init__(self, hparams: Hyperparams, task_spec: TaskSpec):
        self.representation = hk.transform(
            RepresentationNet(hparams, task_spec, hparams.embedding_dim)
        )
        self.prediction = hk.transform(
            PredictionNet(
                task_spec=task_spec,
                value_max=hparams.value.max,
                value_num_bins=hparams.value.num_bins,
                embedding_dim=hparams.embedding_dim,
            )
        )
        rep_key, pred_key = jax.random.split(jax.random.PRNGKey(42))
        self.params = {
            "representation": self.representation.init(rep_key),
            "prediction": self.prediction.init(pred_key),
        }

    def inference(self, params: Any, observation: Observation) -> NetworkOutput:
        # representation + prediction function
        embedding = self.representation.apply(params["representation"], observation)
        return self.prediction.apply(params["prediction"], embedding)

    def get_params(self):
        # Returns the weights of this network.
        return self.params

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0



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

        program_encoding = self._encode_program(inputs, batch_size)
        state_encoding = self._encode_state(inputs, batch_size)

        return self.aggregate_state_program(
            state_encoding, program_encoding, batch_size
        )

    def _encode_program(self, inputs: Observation, batch_size: int) -> chex.Array:
        program = inputs.program
        max_program_size = inputs.program.shape[0]  # pyright: ignore
        program_length = inputs.program_length
        program_onehot = self.make_program_onehot(program, batch_size, max_program_size)
        program_encoding = self.apply_program_mlp_embedder(program_onehot)
        # program_encoding = self.apply_program_attention_embedder(program_onehot)
        # TODO: padding?
        return program_encoding

    def make_program_onehot(
        self, program: Program, batch_size: int, max_program_size: int
    ) -> chex.Array:
        return jax.nn.one_hot(
            program, self._task_spec.num_actions
        )  # negative values are mapped to 0-rows

    def apply_program_mlp_embedder(self, program_encoding: chex.Array) -> chex.Array:
        program_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name="per_instruction_program_embedder",
        )
        program_encoding = program_embedder(program_encoding)
        return program_encoding

    def apply_program_attention_embedder(self, program_encoding):
        attention_params = self._hparams.representation.attention # TODO: make this passable via partial?
        make_attention_block = partial(
            MultiQueryAttentionBlock, attention_params, causal_mask=False
        )
        pass

    def _encode_state(self, inputs: Observation, batch_size: int) -> chex.Array:
        state = inputs.state
        state_re = jnp.real(state)
        state_im = jnp.imag(state)
        return jnp.concatenate([state_re, state_im], axis=-1)

    def aggregate_state_program(
        self, state_encoding: chex.Array, program_encoding: chex.Array, batch_size: int
    ) -> chex.Array:
        state_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name="per_state_embedder",
        )

        # TODO: implement repeat_program_encoding()? why is this there?

        state_embedding = hk.vmap(
            state_embedder, in_axes=1, out_axes=1, split_rng=False
        )(state_encoding)

        grouped_representation = jnp.concatenate(
            [state_embedding, program_encoding], axis=-1
        )

        return self.apply_joint_embedder(grouped_representation, batch_size)

    def apply_joint_embedder(
        self, grouped_representation: chex.Array, batch_size: int
    ) -> chex.Array:
        all_state_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name="per_element_embedder",
        )
        joint_state_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name="joint_embedder",
        )

        joint_resnet = [
            ResBlockV2(self._embedding_dim, name=f"joint_resblock_{i}")
            for i in range(self._hparams.representation.repr_net_res_blocks)
        ]
        permutations_encoded = all_state_net(grouped_representation)
        joint_encoding = joint_state_net(jnp.mean(permutations_encoded, axis=1))
        for net in joint_resnet:
            joint_encoding = net(joint_encoding)
        return joint_encoding


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

    # print(MultiQueryAttentionBlock.sinusoid_position_encoding(10, 5))
