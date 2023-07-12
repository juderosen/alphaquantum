import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import chex
from functools import partial
from typing import Union


Ket = chex.Array  # rank 1 row vec
Bra = chex.Array  # rank 1 col vec
# TODO: should bra actually be represented as a column vec?
# chex.assert_rank(bra, 1) will fail
Operator = chex.Array  # including density matrix (rank 2)
Qobj = Union[Ket, Bra, Operator]


# Various transformations/operations
@jax.jit
def ket2dm(psi: Ket) -> Operator:
    """
    Returns a JAX array representing the density matrix of the ket `psi`.
    """
    chex.assert_rank(psi, 1)
    psi = psi.reshape((-1, 1))
    return jnp.matmul(psi, psi.T.conj())


@jax.jit
def dag(A: Qobj) -> Qobj:
    return A.conj().T


@jax.jit
def apply_operator(op: Operator, state: Operator) -> Operator:
    chex.assert_rank([op, state], [2, 2])
    return jnp.matmul(op, jnp.matmul(state, dag(op)))


@jax.jit
def expm(A: Qobj) -> Qobj:
    """
    Returns the matrix exponential of the JAX array `A`.
    """
    expm_A = linalg.expm(A)
    return expm_A


@jax.jit
def expect_ket(oper: Operator, state: Ket) -> Union[float, complex]:
    """
    Calculate the expectation value for state.  The
    expectation of state `k` on operator `A` is defined as `dag(k) @ A @ k`.

    Args:
        oper: Operator
        state: Ket

    Returns:
        Expectation value
    """
    return dag(state) @ oper @ state


@jax.jit
def expect_dm(oper: Operator, state: Operator) -> Union[float, complex]:
    """
    Calculate the expectation value for state.  The
    expectation of density matrix `R` on operator `A` it is `trace(A @ R)`.

    Args:
        oper: Operator
        state: Operator

    Returns:
        Expectation value
    """
    return jnp.abs(jnp.trace(oper @ state))  # pyright: ignore


@jax.jit
def variance_ket(oper: Operator, state: Ket):
    """Variance of operator for given ket

    Args:
        oper: Operator
        state: Ket

    Returns:
        Variance
    """
    return expect_ket(oper**2, state) - expect_ket(oper, state) ** 2


@jax.jit
def variance_dm(oper: Operator, state: Operator):
    """Variance of operator for given dm

    Args:
        oper: Operator
        state: Operator

    Returns:
        Variance
    """
    return expect_dm(oper**2, state) - expect_dm(oper, state) ** 2


@jax.jit
def overlap_kets(A: Ket, B: Ket) -> Union[float, complex]:
    """Overlap between two gets (inner product)

    Args:
        A: Ket
        B: Ket

    Returns:
        Overlap (inner product) of `A` and `B`.
    """
    return jnp.dot(dag(A), B)


@jax.jit
def row_to_col(A: Ket) -> Bra:
    """Just reshapes row to column vec. (doesn't conjugate)

    Args:
        A: Ket (row)

    Returns:
        Bra (col)
    """
    return jnp.reshape(A, (-1, 1))


@jax.jit
def proj(A: Ket) -> Operator:
    """Calculates projection of ket

    Args:
        A: Ket

    Returns:
        Projection (dm)
    """
    return A * row_to_col(dag(A))


@partial(jax.jit, static_argnums=(0, 1))
def basis(n: int, i: int, dtype=jnp.complex64) -> Ket:
    state = jnp.zeros(n, dtype=dtype)
    state = state.at[i].set(1.0 + 0.0j)
    return state


@partial(jax.jit, static_argnums=(0, 1))
def ghz_state(n: int, dtype=jnp.complex64) -> Ket:
    N = 2**n
    state = jnp.zeros(N, dtype=dtype)
    state = state.at[0].set(1.0 / jnp.sqrt(2))
    state = state.at[N - 1].set(1.0 / jnp.sqrt(2))
    return state


@jax.jit
def sigmax(dtype=jnp.complex64) -> Operator:
    """
    Returns Pauli-x.
    """
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)


@jax.jit
def sigmay(dtype=jnp.complex64) -> Operator:
    """
    Returns Pauli-y.
    """
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)


@jax.jit
def sigmaz(dtype=jnp.complex64) -> Operator:
    """
    Returns Pauli-z.
    """
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)


@jax.jit
def hadamard(dtype=jnp.complex64) -> Operator:
    return (1 / jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]], dtype=dtype)


def CNOT(
    n: int, control: int, target: int, dtype=jnp.complex64
) -> Operator:  # TODO: JITable?
    # assert control < target
    basis0 = ket2dm(basis(2, 0))
    basis1 = ket2dm(basis(2, 1))
    identity1 = jnp.eye(2 ** (control - 0))
    identity2 = jnp.eye(2 ** (target - control - 1))
    identity3 = jnp.eye(2 ** (n - target - 1))
    return jnp.kron(
        identity1,
        jnp.kron(basis0, jnp.kron(identity2, jnp.kron(jnp.eye(2), identity3))),
    ) + jnp.kron(
        identity1, jnp.kron(basis1, jnp.kron(identity2, jnp.kron(sigmax(), identity3)))
    )


@jax.jit
def fidelity_ket(state1: chex.Array, state2: chex.Array) -> float:
    chex.assert_rank([state1, state2], [1, 1])
    inner_product = jnp.dot(state1, jnp.conj(state2))
    fidelity = (jnp.abs(inner_product) ** 2).astype(float)
    return fidelity


@jax.jit
def fidelity_dm(state1: chex.Array, state2: chex.Array) -> float:
    chex.assert_rank([state1, state2], [2, 2])
    state1_sqrt = linalg.sqrtm(state1)
    fidelity = (
        jnp.abs(
            jnp.trace(
                linalg.sqrtm(jnp.matmul(state1_sqrt, jnp.matmul(state2, state1_sqrt)))
            )
            ** 2
        )
    ).astype(float)
    return fidelity
