// Orszag-Tang vortex initial conditions for Idefix
//
// Physical setup:
//   rho = 25 / (36*pi)
//   P   = 5  / (12*pi)
//   vx  = -sin(2*pi*y)
//   vy  =  sin(2*pi*x)
//   Bx  = -sin(2*pi*y) / sqrt(4*pi)
//   By  =  sin(4*pi*x) / sqrt(4*pi)
//
// The magnetic field is initialised via the vector potential Az to guarantee
// divB = 0 from the first timestep (required by constrained transport):
//   Az = cos(4*pi*x) / (4*pi) * B0 + cos(2*pi*y) / (2*pi) * B0

#include "idefix.hpp"
#include "setup.hpp"

static const real pi  = M_PI;
static const real twopi = 2.0 * pi;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
}

void Setup::InitFlow(DataBlock &data) {
  DataBlockHost dh(data);

  const real rho0 = 25.0 / (36.0 * pi);
  const real P0   =  5.0 / (12.0 * pi);
  const real B0   = 1.0 / std::sqrt(4.0 * pi);
  const real n    = 1.0;

  for (int k = 0; k < dh.np_tot[KDIR]; k++) {
    for (int j = 0; j < dh.np_tot[JDIR]; j++) {
      for (int i = 0; i < dh.np_tot[IDIR]; i++) {

        real x = dh.x[IDIR][i];
        real y = dh.x[JDIR][j];

        dh.Vc(RHO, k, j, i) = rho0;
        dh.Vc(PRS, k, j, i) = P0;
        dh.Vc(VX1, k, j, i) = -std::sin(n * twopi * y);
        dh.Vc(VX2, k, j, i) =  std::sin(n * twopi * x);

        // Cell-centred B (used as initial guess; CT will correct from Az)
        dh.Vc(BX1, k, j, i) = -std::sin(n * twopi * y) * B0;
        dh.Vc(BX2, k, j, i) =  std::sin(2.0 * n * twopi * x) * B0;
      }
    }
  }

  // Initialise the face-centred magnetic field via vector potential Az.
  // For wavenumber multiplier n the vector potential is:
  //   Az = cos(2*n*2*pi*x) / (2*n*2*pi) * B0 + cos(n*2*pi*y) / (n*2*pi) * B0
  // which gives Bx = dAz/dy = -sin(n*2*pi*y)*B0
  //             By = -dAz/dx =  sin(2*n*2*pi*x)*B0
  const real kx = 2.0 * n * twopi;   // wavenumber for By component
  const real ky =       n * twopi;   // wavenumber for Bx component

  for (int k = 0; k < dh.np_tot[KDIR]; k++) {
    for (int j = 0; j < dh.np_tot[JDIR] + JOFFSET; j++) {
      for (int i = 0; i < dh.np_tot[IDIR] + IOFFSET; i++) {

        real xf = dh.xl[IDIR][i];   // left face x
        real yf = dh.xl[JDIR][j];   // left face y

        real Az_ij   = std::cos(kx * xf)                        / kx * B0 +
                       std::cos(ky * yf)                        / ky * B0;
        real Az_ip1j = std::cos(kx * (xf + dh.dx[IDIR][i]))    / kx * B0 +
                       std::cos(ky * yf)                        / ky * B0;
        real Az_ijp1 = std::cos(kx * xf)                        / kx * B0 +
                       std::cos(ky * (yf + dh.dx[JDIR][j]))    / ky * B0;

        // Bx face = -(Az(i,j+1) - Az(i,j)) / dy  (sign from curl: Bx = dAz/dy)
        if (i < dh.np_tot[IDIR]) {
          dh.Vs(BX1s, k, j, i) = -(Az_ijp1 - Az_ij) / dh.dx[JDIR][j];
        }
        // By face = (Az(i+1,j) - Az(i,j)) / dx  (sign from curl: By = -dAz/dx)
        if (j < dh.np_tot[JDIR]) {
          dh.Vs(BX2s, k, j, i) = (Az_ip1j - Az_ij) / dh.dx[IDIR][i];
        }
      }
    }
  }

  dh.SyncToDevice();
}
