from alphaquantum.qtools import states
import jax.numpy as jnp
import pytest
import chex


def test_CNOT():
    cnot_02_module = states.CNOT(3, 0, 2)
    cnot_02_manual = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=jnp.complex64,
    )
    assert (cnot_02_module == cnot_02_manual).all()

    cnot_01_module = states.CNOT(3, 0, 1)
    cnot_01_manual = jnp.kron(
        states.ket2dm(states.basis(2, 1)), jnp.kron(states.sigmax(), jnp.eye(2))
    ) + jnp.kron(states.ket2dm(states.basis(2, 0)), jnp.kron(jnp.eye(2), jnp.eye(2)))
    assert (cnot_01_module == cnot_01_manual).all()


def test_ghz_state():
    ghz_ket_2_module = states.ghz_state(2)
    ghz_ket_2_manual = (1 / jnp.sqrt(2)) * jnp.array([1, 0, 0, 1], dtype=jnp.complex64)
    assert (
        ghz_ket_2_module == ghz_ket_2_manual
    ).all()  # there's a jnp.arrayalmosteq thing right?


def test_ket2dm():
    ghz_ket = states.ghz_state(2)
    ghz_dm = states.ket2dm(ghz_ket)
    chex.assert_rank(ghz_dm, 2)


def test_hadamard():
    hadamard_module = states.hadamard()
    hadamard_manual = jnp.array(
        [
            [0.70710677 + 0.0j, 0.70710677 + 0.0j],
            [0.70710677 + 0.0j, -0.70710677 + 0.0j],
        ]
    )
    assert jnp.allclose(hadamard_module, hadamard_manual)
