"""Using QuTiP measurement module as guide https://github.com/qutip/qutip/blob/master/qutip/measurement.py"""
import jax
from jax import jit, lax
import jax.numpy as jnp
import chex
from typing import List, Tuple, Union

from alphaquantum.qtools.states import (
    dag,
    expect_dm,
    overlap_kets,
    proj,
    Ket,
    Operator,
    Qobj,
)


@jit
def measurement_statistics_povm_ket(
    state: Ket, ops: List[Operator]
) -> Tuple[List[Ket], List[float]]:
    """
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified ket.

    Args:
        state: Ket state to measure
        ops: List of measurement operators

    Returns:
        list of measurement outcomes (kets), and corresponding probabilities
    """
    chex.assert_rank([state, *ops], [1, *[2 for _ in range(len(ops))]])

    jax_ops = jnp.array(ops)

    def helper(op, state):
        prob = jnp.abs(dag(state) @ dag(op) @ op @ state)
        collapsed = (op @ state) / jnp.sqrt(prob)
        return prob, collapsed

    probabilities, collapsed_states = lax.map(lambda op: helper(op, state), jax_ops)
    return collapsed_states, probabilities


@jit
def measurement_statistics_povm_dm(
    density_mat: Operator, ops: List[Operator]
) -> Tuple[List[Ket], List[float]]:
    """
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified density matrix.

    Args:
        state: Density matrix state to measure
        ops: List of measurement operators

    Returns:
        list of measurement outcomes (dms), and corresponding probabilities
    """
    chex.assert_rank([density_mat, *ops], [2, *[2 for _ in range(len(ops))]])

    jax_ops = jnp.array(ops)

    def helper(op, density_mat):
        collapsed = op @ density_mat @ dag(op)
        prob = jnp.abs(jnp.trace(collapsed))
        return prob, collapsed

    probabilities, collapsed_states = lax.map(
        lambda op: helper(op, density_mat), jax_ops
    )
    return collapsed_states, probabilities


# WARN: No good way to get around JAX control-flow restrictions to make rank-agnostic
# measurement functions. So, for now, I'm going to expose _measurement_statistics_povm_ket
# and _measurement_statistics_povm_dm (no underscores) and just assume the user always knows
# whether they're measuring a ket or dm (seems reasonable). As a result, this function WILL fail.
def measurement_statistics_povm(
    state: Qobj, ops: List[Operator]
) -> Tuple[List[Ket], List[float]]:
    chex.assert_rank(ops, [2 for _ in range(len(ops))])

    jax_ops = jnp.array(ops)

    E = lax.map(lambda op: op @ dag(op), jax_ops)
    is_ID = jnp.sum(E, axis=0)
    chex.assert_trees_all_close(is_ID, jnp.eye(is_ID.shape[0], dtype=is_ID.dtype))

    state_rank = jnp.ndim(state)
    print(state_rank == 1)
    print(state, ops)
    # BUG: lax.cond traces both true_fun and false_fun with same input,
    # and it expects same output shapes. So we're triggering errors by
    # passing a dm to _measurement_statistics_povm_ket. How to fix?
    return lax.cond(
        state_rank == 1,
        measurement_statistics_povm_ket,
        measurement_statistics_povm_dm,
        state,
        ops,
    )


@jit
def measurement_statistics_observable_ket(
    state: Ket, op: Operator
) -> Tuple[chex.Array, chex.Array, chex.Array]:  # TODO:should this output to lists?
    """
    Return the measurement eigenvalues, eigenstates (or projectors) and
    measurement probabilities for the given state and measurement operator.

    Args:
        state: Ket
        op: Measurement operator

    Returns:
        eigenvalues, eigenstates, probabilities (all as JAX arrays)
    """

    eigenvalues, eigenstates = jnp.linalg.eig(op)
    probabilities = lax.map(lambda e: jnp.abs(overlap_kets(e, state)) ** 2, eigenstates)
    return eigenvalues, eigenstates, probabilities


@jit
def measurement_statistics_observable_dm(
    state: Operator, op: Operator
) -> Tuple[chex.Array, chex.Array, chex.Array]:  # TODO:should this output to lists?
    """
    Return the measurement eigenvalues, eigenstates (or projectors) and
    measurement probabilities for the given state and measurement operator.

    Args:
        state: Ket
        op: Measurement operator

    Returns:
        eigenvalues, eigenstates, probabilities (all as JAX arrays)
    """

    eigenvalues, eigenstates = jnp.linalg.eig(op)
    projectors = lax.map(lambda e: proj(e), eigenstates)
    probabilities = lax.map(lambda v: expect_dm(v, state), projectors)
    return eigenvalues, projectors, probabilities


@jit
def measure_observable_ket(
    state: Ket, op: Operator, rng_key: jax.random.PRNGKeyArray
) -> Tuple[Union[float, complex], Ket]:
    """
    Perform a measurement specified by an operator on the given state (ket).
    This function simulates the classic quantum measurement described in many
    introductory texts on quantum mechanics. The measurement collapses the
    state to one of the eigenstates of the given operator and the result of the
    measurement is the corresponding eigenvalue.

    Args:
        state: Ket
        op: Measurement operator

    Returns:
        eigenvalue, collapsed state (ket)
    """
    eigenvalues, eigenstates, probabilities = measurement_statistics_observable_ket(
        state, op
    )
    probabilities /= jnp.sum(
        probabilities
    )  # HACK: normalize so probs sum to 1 to placate jnp.random.choice
    i = jax.random.choice(
        rng_key, jnp.array(range(eigenvalues.shape[0])), p=probabilities
    )
    state = eigenstates[i]
    return eigenvalues[i], state


@jit
def measure_observable_dm(
    state: Operator, op: Operator, rng_key: jax.random.PRNGKeyArray
) -> Tuple[Union[float, complex], Operator]:
    """
    Perform a measurement specified by an operator on the given state (dm).
    This function simulates the classic quantum measurement described in many
    introductory texts on quantum mechanics. The measurement collapses the
    state to one of the eigenstates of the given operator and the result of the
    measurement is the corresponding eigenvalue.

    Args:
        state: Operator
        op: Measurement operator

    Returns:
        eigenvalue, collapsed state (dm)
    """
    eigenvalues, projectors, probabilities = measurement_statistics_observable_dm(
        state, op
    )
    probabilities /= jnp.sum(
        probabilities
    )  # HACK: normalize so probs sum to 1 to placate jnp.random.choice
    i = jax.random.choice(
        rng_key, jnp.array(range(eigenvalues.shape[0])), p=probabilities
    )
    state = (projectors[i] @ state @ projectors[i]) / probabilities[i]
    return eigenvalues[i], state


@jit
def measure_povm_ket(
    state: Ket, ops: List[Operator], rng_key: jax.random.PRNGKeyArray
) -> Tuple[chex.Array, Ket]:
    """
    This function simulates a POVM measurement. The measurement collapses the
    state (ket) to one of the resultant states of the measurement and returns the
    index of the operator corresponding to the collapsed state as well as the
    collapsed state.

    Args:
        state: Ket state to measure
        ops: List of measurement operators

    Returns:
        index of measurement op, state
    """
    collapsed_states, probabilities = measurement_statistics_povm_ket(state, ops)
    probabilities /= jnp.sum(probabilities)
    index = jax.random.choice(
        rng_key, jnp.array(range(len(collapsed_states))), p=probabilities
    )
    state = collapsed_states[index]
    return index, state


@jit
def measure_povm_dm(
    state: Operator, ops: List[Operator], rng_key: jax.random.PRNGKeyArray
) -> Tuple[chex.Array, Operator]:
    """
    This function simulates a POVM measurement. The measurement collapses the
    state (dm) to one of the resultant states of the measurement and returns the
    index of the operator corresponding to the collapsed state as well as the
    collapsed state.

    Args:
        state: Operator state to measure
        ops: List of measurement operators

    Returns:
        index of measurement op, state
    """
    collapsed_states, probabilities = measurement_statistics_povm_dm(state, ops)
    probabilities /= jnp.sum(probabilities)
    index = jax.random.choice(
        rng_key, jnp.array(range(len(collapsed_states))), p=probabilities
    )
    state = collapsed_states[index]
    return index, state
