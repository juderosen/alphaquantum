import jax
import jax.numpy as jnp
from jax import jit, vmap
import chex
from functools import partial
from typing import Sequence, Tuple, Union, Optional

from alphaquantum.qtools import states
from alphaquantum.envs.wrappers import StatefulWrapper

from rich import print

State = chex.Array
Operator = chex.Array
Action = Union[chex.Array, int]  # TODO: do we need these unions?
Program = Union[chex.Array, Sequence[Action]]
Fidelity = Union[chex.Array, float]
Time = Union[chex.Array, float]
Reward = Union[chex.Array, float]
Done = Union[chex.Array, bool]
Info = chex.Array
Truncated = Union[chex.Array, bool]
Seed = Union[chex.Array, float]


@chex.dataclass(frozen=True)
class Observation:
    state: State
    program: Program
    program_length: int


@chex.dataclass(frozen=True)
class TaskSpec:
    max_program_size: int
    num_wires: int
    num_actions: int
    fidelity_reward_weight: float
    time_reward_weight: float
    goal_state: State
    possible_actions: Sequence[Operator]
    fidelity_threshold: int


@chex.dataclass(frozen=True)
class Env:
    # task_spec: TaskSpec
    program: Program
    program_length: int
    result_state: State
    done: Done
    truncated: Truncated
    fidelity: Fidelity
    time: Time
    reward: Reward


PAD = -1  # action padding value for no action


def get_program_sequence_unpadded(program: Program) -> Program:  # not JITable
    """Take a fixed-length program with padding and remove padding"""
    program_zeroed = jnp.add(program, -PAD)  # pyright: ignore
    last_action_index = jnp.nonzero(program_zeroed)[0][-1]
    return jnp.split(program, [last_action_index + 1])[0]  # pyright: ignore


class FidelityGameEnv:
    """Stateless environment for fidelity optimization game"""

    def __init__(self, task_spec: TaskSpec):
        """Just stores task_spec"""
        self.task_spec = task_spec

    @partial(jit, static_argnums=(0,))
    def step(
        self, env: Env, action: Action
    ) -> Tuple[Env, Observation, Reward, Done, Truncated, Info]:
        """Step function to perform action on environment
        Args:
            env: Current environment
            action: Action to take
        Returns:
            (new env, observation, reward, done, truncated, info)
        """
        action_index = jnp.array(action, int)
        action_operator = self.task_spec.possible_actions[action_index]
        new_program = env.program
        new_program = new_program.at[env.program_length].set(action)  # pyright: ignore
        new_program_length = env.program_length + 1
        new_state = states.apply_operator(action_operator, env.result_state)
        new_fidelity = states.fidelity_dm(new_state, self.task_spec.goal_state)
        new_time = 0.0  # todo: implement this

        reward = (
            new_fidelity - env.fidelity
        ) * self.task_spec.fidelity_reward_weight + (
            new_time - env.time
        ) * self.task_spec.time_reward_weight
        done = new_fidelity >= self.task_spec.fidelity_threshold
        truncated = False
        info = jnp.array([])

        env = Env(  # pyright: ignore reportUnknownArgumentType
            # task_spec=self.task_spec,
            program=new_program,
            program_length=new_program_length,
            result_state=new_state,
            done=done,
            truncated=truncated,
            fidelity=new_fidelity,
            time=new_time,
            reward=reward,
        )
        observation = Observation(  # pyright: ignore reportUnknownArgumentType
            state=new_state,
            program=new_program,
            program_length=new_program_length,
        )
        return env, observation, reward, done, truncated, info

    @partial(jit, static_argnums=(0,))
    def reset(self) -> Tuple[Env, Observation, Info]:
        """Reset environment to |0>
        Returns:
            (env, observation, info)
        """
        state = states.ket2dm(states.basis(2**self.task_spec.num_wires, 0))
        program = jnp.add(jnp.zeros(self.task_spec.max_program_size), PAD)
        # -1 is padding value for no action (yet). will need to deal with this in code
        env = Env(  # pyright: ignore reportUnknownArgumentType
            # task_spec=self.task_spec,
            program=program,
            program_length=0,
            result_state=state,
            done=False,
            truncated=False,
            fidelity=states.fidelity_dm(state, self.task_spec.goal_state),
            time=0.0,
            reward=0.0,
        )
        observation = Observation(  # pyright: ignore reportUnknownArgumentType
            state=state,
            program=program,
            program_length=0,
        )
        info = jnp.array([])
        return env, observation, info

    def close(self):
        pass


def fidelityWrapper(task_spec: TaskSpec):
    """Create stateful environment wrapper class for FidelityGameEnv
    Args:
        task_spec: TaskSpec dataclass
    Returns:
        StatefulWrapper class for interacting with FidelityGameEnv
    """
    return StatefulWrapper(FidelityGameEnv, task_spec)


# testing stuff
if __name__ == "__main__":
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

    # test_env = FidelityGameEnv(task_spec_ghz3)
    # test_env_reset = test_env.reset()[0]
    # test_env_step = test_env.step(test_env_reset, 0)
    # print(test_env_step)

    stateful_env = StatefulWrapper(FidelityGameEnv, task_spec_ghz3)
    print(stateful_env.state)
    print(stateful_env.step(0))
    print(stateful_env.step(3))
    print(stateful_env.step(5))
    print(get_program_sequence_unpadded(stateful_env.state.program))

    # print(fidelityWrapper(task_spec_ghz3).step(0))
