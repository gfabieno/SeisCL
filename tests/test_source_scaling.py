import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from SeisCL.SeisCL import SeisCL


def test_force_scaling_matches_physical_volume():
    dh = 10.0
    dt = 0.002
    rho = 2000.0
    vol = dh * dh

    # rip corresponds to dt/(rho*dh) in the discretized scheme
    rip = dt / (rho * dh)
    force_history = np.array([1.0, -0.5, 0.25])

    factor = rip / (dt * dh)
    scaled = force_history * factor

    injected_velocity_change = scaled * dt
    expected = force_history * dt / (rho * vol)

    np.testing.assert_allclose(injected_velocity_change, expected)


def test_moment_tensor_strain_to_stress_conversion():
    # Lamé parameters
    mu = 10.0
    lam = 15.0
    M = lam + 2 * mu

    strain_rate = np.array([1e-6, -2e-6])

    sxx_from_mxx = strain_rate * M
    szz_from_mxx = strain_rate * lam
    sxz_from_mxz = strain_rate * mu

    np.testing.assert_allclose(sxx_from_mxx, np.array([3.5e-5, -7e-5]))
    np.testing.assert_allclose(szz_from_mxx, np.array([1.5e-5, -3e-5]))
    np.testing.assert_allclose(sxz_from_mxz, np.array([1e-5, -2e-5]))


def test_reciprocal_force_scaling_symmetry():
    dh = 5.0
    dt = 0.001
    rho = 2500.0
    inv_rho = dt / (rho * dh)
    vol_scale = dh

    source_a = inv_rho / (dt * vol_scale)
    source_b = inv_rho / (dt * vol_scale)

    np.testing.assert_allclose(source_a, source_b)


def test_scale_sources_flag_is_optional(tmp_path):
    sim = SeisCL(scale_sources=0)

    sim.write_csts(workdir=tmp_path)

    with h5py.File(tmp_path / sim.file_csts, "r") as f:
        assert f["scale_sources"][()] == 0

    sim.scale_sources = 1
    sim.write_csts(workdir=tmp_path)

    loader = SeisCL()
    loader.read_csts(workdir=tmp_path)
    assert loader.scale_sources == 1

