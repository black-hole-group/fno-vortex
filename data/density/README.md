# Input Data: Density Field

## File Structure

The data is split into `train/` (91 x/y file pairs) and `test/` (21 x/y file pairs). Each pair consists of:

- `x_<idx>.npy` — input block, shape `(16, 128, 128, 10, 7)`
- `y_<idx>.npy` — output block, shape `(16, 128, 128, 10)`

### Dimensions of `x`

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 16 | Samples (sliding temporal windows from the same simulation) |
| 1,2 | 128x128 | Spatial grid |
| 3 | 10 | Temporal frames |
| 4 | 7 | Channels: 5 density snapshots + nu + mu |

- **Channels 0-4**: input density frames, spaced dt = 1.0 code units apart
- **Channel 5**: kinematic viscosity nu (constant per sample, spatially uniform)
- **Channel 6**: Ohmic diffusivity mu (constant per sample, spatially uniform)

### Dimensions of `y`

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 16 | Samples (same as x) |
| 1,2 | 128x128 | Spatial grid |
| 3 | 10 | Predicted future frames, spaced dt = 4.0 code units apart |

## Origin

These files were produced by a preprocessing script (not included in this repo) that:

1. Ran 50 FARGO3D simulations of the Orszag-Tang vortex, each with 1000 timesteps on a 128x128 grid
2. Sampled nu = mu values from the range [1e-5, 5e-2]
3. Extracted overlapping sliding windows of input/output frames from each simulation
4. Appended nu and mu as the last 2 channels of the input array
5. Grouped 16 windows per `.npy` file

The model trains on the first 160 frames (~0.73 Alfven times) and predicts frames 160-1000 (~0.73-4.39 Alfven times).

## Training/Test Split

- **Training**: 48 simulations, 13 unique (nu, mu) values found in the files
- **Test**: 2 held-out simulations with nu = mu = 5e-5 and nu = mu = 3e-4

## Open Questions

- The preprocessing script that converted raw FARGO3D `.dat` outputs into these `.npy` blocks is missing from the repo, so the exact sliding-window construction cannot be verified.
- The paper mentions 48 training simulations, but only 13 unique (nu, mu) values appear across the 91 train files. It is unclear whether multiple simulations share the same parameters or if the mapping is more complex.
- The temporal axis of `x` has size 10, but only 5 channels are density frames. The relationship between the temporal axis and the 5 input channels is not fully clear from the code or paper alone.
