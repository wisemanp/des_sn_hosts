import os
import pytest
import numpy as np
from des_sn_hosts.simulations import aura

CFG_ENV = "DES_SN_AURA_CFG"

pytestmark = pytest.mark.skipif(
    CFG_ENV not in os.environ or not os.path.exists(os.environ.get(CFG_ENV, "")),
    reason=f"Set {CFG_ENV} to an aura YAML config to run these tests."
)

@pytest.fixture(scope="module")
def cfg_path():
    return os.environ[CFG_ENV]

@pytest.fixture
def zarr(cfg_path):
    sim = aura.Sim(cfg_path)
    return np.sort(sim.flux_df['z'].unique().astype(float))

def test_single_z_baked(cfg_path, zarr):
    sim = aura.Sim(cfg_path)
    z0 = float(zarr[len(zarr)//2])
    df = sim._sample_SNe_z(z0, 300)
    assert len(df) == 300
    assert np.isclose(df['z'].iloc[0], z0)

def test_single_z_recompute_dtd(cfg_path, zarr):
    sim = aura.Sim(cfg_path)
    sim.config['force_recompute_dtd'] = True
    sim.config['DTD'] = {'model': 'power_law', 'params': {'beta': 1.18, 'norm': 2.08e-13}}
    z0 = float(zarr[len(zarr)//2])
    df = sim._sample_SNe_z(z0, 300)
    assert len(df) == 300

def test_multi_z_baked(cfg_path, zarr):
    sim = aura.Sim(cfg_path)
    w = (zarr.astype(float) ** 2.5)
    w = w / w.sum()
    n_per = sim._get_z_dist(w, n=400, zbins=zarr)
    sim.sample_SNe(zarr, n_per, save_df=False)
    assert len(sim.sim_df) == 400
    assert set(np.unique(sim.sim_df['z'].values)).issubset(set(zarr))

def test_multi_z_recompute_dtd(cfg_path, zarr):
    sim = aura.Sim(cfg_path)
    sim.config['force_recompute_dtd'] = True
    sim.config['DTD'] = {'model': 'power_law', 'params': {'beta': 1.05, 'norm': 2.08e-13}}
    w = (zarr.astype(float) ** 2.5)
    w = w / w.sum()
    n_per = sim._get_z_dist(w, n=400, zbins=zarr)
    sim.sample_SNe(zarr, n_per, save_df=False)
    assert len(sim.sim_df) == 400
