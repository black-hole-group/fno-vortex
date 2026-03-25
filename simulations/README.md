# Simulations

Data generation for the FNO model using the Idefix MHD code.

See [usage.md](usage.md) for step-by-step instructions.

## Idefix-specific gotchas

- Idefix uses **Kokkos** for GPU portability, so all setup code runs on the device. `setup.cpp` uses `DataBlockHost` (CPU mirror) in `InitFlow` — the correct pattern. Do not access `data.Vc` directly.
- The `Vs` (staggered) arrays for the magnetic field are face-centred, not cell-centred. The CT scheme in Idefix is strict about this — the vector potential initialisation in `setup.cpp` handles it correctly.
- Idefix's `idefix.ini` is whitespace-sensitive in some versions. The `inifix` library handles patching safely; avoid hand-editing the patched files.

## Before running all 25 simulations

- Do a short test run (`tstop = 1.0`, 20 VTK files) to verify the setup compiles and produces physically reasonable output (density should stay near 25/36π ≈ 0.22, vortices should be visible).
- Check the CFL is satisfied — with explicit viscosity/resistivity at small ν/μ the timestep is MHD-limited (fine), but at large values (ν = μ ~ 1e-2) the diffusive CFL can become restrictive and slow things down significantly.

## Output volume

25 simulations × 1001 VTK files × ~5 fields × 128×128 floats ≈ **~33 GB** of raw VTK data. Make sure the server has enough scratch space before launching the full batch.

## Sanity check after conversion

```python
import numpy as np
x = np.load("data/density/train/x_0.npy")
print(x.shape)            # (20, 128, 128, 10, 7)
print(x[0,:,:,0,0].min(), x[0,:,:,0,0].max())  # physical units (normalisation happens in train.py)
```

The raw `.npy` values are in physical units — normalisation to [-1, 1] is applied at load time in `train.py`, so this is expected.
