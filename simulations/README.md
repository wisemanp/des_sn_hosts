# Simulations Pipeline (Refactor)

This refactor streamlines the workflow into three clear stages:

1. Build SFH catalog (rarely changes)
2. Build hosts (photometry/colours) from SFH using spectral templates, dust, and nebular
3. Simulate supernovae using delay-time distributions (DTD), stretch/colour models, noise and selection

The heavy spectral work (templates, Av/Rv, nebular) is separated from DTD application so you can re-run DTDs quickly without recomputing spectra. Host libs now embed SFH arrays directly (HDF5), and DTD-based SN age distributions can be baked at build time or recomputed on demand during simulation.

## CLI scripts

- scripts/build_sfh_catalog.py
  - Reads an input SFH HDF5 and emits a compact SFH catalog with per-row arrays: ages_gyr, m_formed, plus metadata (z, t_f, m_tot)
  - Example:
    - python simulations/scripts/build_sfh_catalog.py -c configs/pipeline.yaml --time-res 5

- scripts/build_hosts_from_sfh.py
  - Consumes the SFH catalog and produces a hostlib with photometry/colours using SynSpec
  - Supports an Av grid and nebular options via config; persists SFH arrays in-row: `SFH_ages`, `SFH_m_formed`
  - Applies and bakes a DTD by default, storing `SN_ages`, `SN_age_dist`, and `pred_rate_total`
  - Choose DTD with `--dtd-model` (power_law|two_component|broken_power_law) and `--dtd-params` (e.g., beta=1.14,norm=2.08e-13), or disable with `--no-dtd`
  - Example:
    - python simulations/scripts/build_hosts_from_sfh.py -c configs/pipeline.yaml --sfh sfh_catalog.h5 --dtd-model power_law --dtd-params beta=1.14,norm=2.08e-13

- scripts/apply_dtd.py (optional)
  - If you built a raw hostlib without DTD (`--no-dtd`), you can later apply a DTD and bake distributions using embedded SFH arrays
  - Example:
    - python simulations/scripts/apply_dtd.py --hostlib hostlib_from_sfh.h5 --sfh_dir NOT_USED --model power_law --params beta=1.14,norm=2.08e-13 --out hostlib_powerlaw.h5

- scripts/simulate_sne.py
  - Uses aura.Sim from an aura config to simulate SNe (single process)
  - aura.Sim uses baked `SN_ages`/`SN_age_dist` if present. If not present (or if `--recompute-dtd` is set), it recomputes distributions from `SFH_ages`/`SFH_m_formed` using the requested DTD
  - Example:
    - python simulations/scripts/simulate_sne.py -c configs/aura.yaml -n 200000
    - python simulations/scripts/simulate_sne.py -c configs/aura.yaml -n 200000 --recompute-dtd --dtd-model power_law --dtd-params beta=1.14,norm=2.08e-13

- scripts/run_pipeline.py
  - Simple orchestrator placeholder; prefer running individual scripts for now

- scripts/sweep_sne_params.py
  - Sweep over a YAML-defined parameter grid and run SNe simulations for each combo
  - Supports dotted keys to override nested config entries (e.g. `SN_rv_model.params.mu`)
  - Example grid YAML:
```
grid:
  SN_rv_model.params.mu: [2.6, 3.1, 3.4]
  Host_Av_model.params.sigma: [0.2, 0.4]
  x1_model.params.mu: [-0.2, 0.0, 0.2]
  SN_E_model.params.beta: [1.14, 1.18]
```
  - Run:
    - python simulations/scripts/sweep_sne_params.py -c configs/aura.yaml -g configs/grid.yaml -n 200000 --out-dir /path/to/out

## aura.py integration

The Sim._sample_SNe_z now:
- Prefers baked `SN_ages` and `SN_age_dist` columns on the hostlib rows
- If missing (or if recompute is requested), rebuilds SN age distributions from embedded `SFH_ages`/`SFH_m_formed` using the configured DTD, and maps onto the simulation age grid
- There is no SNANA file fallback in this path

## Configuration

Use a YAML config (e.g., configs/pipeline.yaml) including:

```
save:
  dir: /path/to/outputs
input_sfh_path: /path/to/sfh_store.h5
templates: BC03  # or PEGASE
neb: true
logU: -2
av_lo: 0.0
av_hi: 1.0
n_av: 20
av_step_type: lin
```

## Notes
- Keep host building rare; DTDs can be baked at build time or recomputed at simulate time.
- Pre-existing ages: if your hostlib already has `SN_ages`/`SN_age_dist`, those are used. If it only has `SFH_ages`/`SFH_m_formed`, the simulator can compute DTD-based distributions on the fly.
- Legacy scripts are retained under scripts/ but considered deprecated; see scripts/DEPRECATED.md.
