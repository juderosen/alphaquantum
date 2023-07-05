import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import chex


# Various transformations/operations
def ket2dm(psi: chex.Array) -> chex.Array:
    """
    Returns a PyTorch tensor representing the density matrix of the ket `psi`.
    """
    chex.assert_rank(psi, 1)
    psi = psi.reshape((-1, 1))
    return jnp.matmul(psi, psi.T.conj())

@jax.jit
def dag(A: chex.Array) -> chex.Array:
    return A.conj().T

@jax.jit
def apply_operator(op: chex.Array, state: chex.Array) -> chex.Array:
    chex.assert_rank([op, state], [2, 2])
    return jnp.matmul(op, jnp.matmul(state, dag(op)))

def expm(A: chex.Array) -> chex.Array:
    """
    Returns the matrix exponential of the PyTorch tensor `A`.
    """
    expm_A = linalg.expm(A)
    return expm_A

def basis(n: int, i: int, dtype=jnp.complex64) -> chex.Array:
    state = jnp.zeros(n, dtype=dtype)
    state = state.at[i].set(1.0 + 0.0j)
    return state

def ghz_state(n: int, dtype=jnp.complex64) -> chex.Array:
    N = 2**n
    state = jnp.zeros(N, dtype=dtype)
    state = state.at[0].set(1.0 / jnp.sqrt(2))
    state = state.at[N - 1].set(1.0 / jnp.sqrt(2))
    return state

@jax.jit
def sigmax(dtype=jnp.complex64) -> chex.Array:
    """
    Returns Pauli-x.
    """
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)

@jax.jit
def sigmay(dtype=jnp.complex64) -> chex.Array:
    """
    Returns Pauli-y.
    """
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)

@jax.jit
def sigmaz(dtype=jnp.complex64) -> chex.Array:
    """
    Returns Pauli-z.
    """
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)

def hadamard(dtype=jnp.complex64) -> chex.Array:
    return (1/jnp.sqrt(2))*jnp.array([[1, 1], [1, -1]], dtype=dtype)

def CNOT(n: int, control: int, target: int, dtype=jnp.complex64) -> chex.Array:
    #assert control < target
    basis0 = ket2dm(basis(2, 0))
    basis1 = ket2dm(basis(2, 1))
    identity1 = jnp.eye(2 ** (control - 0))
    identity2 = jnp.eye(2 ** (target - control - 1))
    identity3 = jnp.eye(2 ** (n - target - 1))
    return jnp.kron(identity1, jnp.kron(basis0, jnp.kron(identity2, jnp.kron(jnp.eye(2), identity3)))) + jnp.kron(identity1, jnp.kron(basis1, jnp.kron(identity2, jnp.kron(sigmax(), identity3))))

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
    fidelity = (jnp.abs(jnp.trace(linalg.sqrtm(jnp.matmul(state1_sqrt, jnp.matmul(state2, state1_sqrt)))) ** 2)).astype(float)
    return fidelity

#state1, state2 = ghz_state(2), ghz_state(2)
#print(fidelity_ket(state1, state2))

#state1_dm, state2_dm = ket2dm(state1), ket2dm(state2)
#print(fidelity_dm(state1_dm, state2_dm))

#cnot_test = CNOT(3, 0, 2)
#print(cnot_test.shape)
#print(cnot_test)

