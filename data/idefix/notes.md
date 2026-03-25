# Simulation Setup Notes

## Comparison: `simulations/idefix/` vs `idefix/test/MHD/OrszagTang/`

Both setups implement the same **2D Orszag-Tang vortex** with identical:
- **Preprocessor defines**: `COMPONENTS=2`, `DIMENSIONS=2`, `GEOMETRY=CARTESIAN`
- **Initial conditions**: same density (25/36π), pressure (5/12π), velocity (-sin(2πy), sin(2πx)), and magnetic field with B0 = 1/√(4π)
- **Resolution**: 128×128 uniform grid
- **Boundary conditions**: periodic on all sides
- **Time integration**: 2nd-order RK, first_dt = 1e-4

## Key Differences

| Aspect | This setup (`simulations/idefix/`) | Reference (`test/MHD/OrszagTang/`) |
|--------|-----------------------------------|------------------------------------|
| **Domain** | [0, 2π] × [0, 2π] | [0, 1] × [0, 1] |
| **CFL** | 0.3 | 0.6 |
| **CFL_max_var** | 1.1 | not set |
| **tstop** | 50.0 | 0.5 |
| **Riemann solver** | `hlld` | `roe` (default), plus 7 variants testing hlld/hll/tvdlf |
| **EMF scheme** | `uct_contact` (explicit) | default (not specified) |
| **Viscosity** | explicit, constant, ν = 1e-3 | **none** (ideal MHD) |
| **Resistivity** | explicit, constant, η = 1e-3 | **none** (ideal MHD) |
| **VTK output** | every 0.05 (→ 1000 snapshots) | only at t = 0.5 (single snapshot) |
| **B-field init** | always via vector potential (Az finite differences) | two paths: vector potential or direct staggered B (via `EVOLVE_VECTOR_POTENTIAL` flag) |

## Main Takeaways

1. **Domain scaling**: This setup uses the physical [0, 2π]² domain while the reference uses [0, 1]². The 2π factor is absorbed into the initial condition formulas in the reference, so both are physically equivalent.

2. **Dissipation**: The biggest physics difference — this setup includes **explicit viscosity and resistivity** (varied across [1e-5, 5e-2] for training data generation), while the reference is purely **ideal MHD**. This is by design since the FNO is conditioned on ν and η.

3. **Long runs vs. quick test**: This setup runs 100× longer (tstop=50 vs 0.5) with 1000 output snapshots — it's a production data-generation setup. The reference is a quick validation test that outputs a single snapshot.

4. **Solver conservatism**: This setup uses a more conservative CFL (0.3 vs 0.6), likely because the long integration time and explicit dissipation terms warrant extra stability. The reference tests many solver/EMF combinations; this setup settled on `hlld` + `uct_contact`.

5. **setup.cpp**: Functionally equivalent initial conditions. This version always uses the vector potential path for B-field initialization; the reference supports both paths.
