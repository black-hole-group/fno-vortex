// Orszag-Tang vortex initial conditions for Idefix
//
// Physical setup (from paper):
//   rho = 25 / (36*pi)
//   P   = 5  / (12*pi)
//   vx  = -sin(2*pi*y)
//   vy  =  sin(2*pi*x)
//   Bx  = -sin(2*pi*y) / sqrt(4*pi)
//   By  =  sin(4*pi*x) / sqrt(4*pi)
//
// The magnetic field is initialised via the vector potential Az to guarantee
// divB = 0 from the first timestep (required by constrained transport):
//   Az = cos(4*pi*x) / (4*pi*sqrt(4*pi)) + cos(2*pi*y) / (2*pi*sqrt(4*pi))

#include "idefix.hpp"
#include "setup.hpp"

static const real pi  = M_PI;
static const real twopi = 2.0 * pi;

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Nothing to do at setup construction
}

void Setup::InitFlow(DataBlock &data) {
  DataBlockHost dh(data);

  const real rho0 = 25.0 / (36.0 * pi);
  const real P0   =  5.0 / (12.0 * pi);
  const real B0   = 1.0 / std::sqrt(4.0 * pi);

  for (int k = 0; k < dh.np_tot[KDIR]; k++) {
    for (int j = 0; j < dh.np_tot[JDIR]; j++) {
      for (int i = 0; i < dh.np_tot[IDIR]; i++) {

        real x = dh.x[IDIR][i];
        real y = dh.x[JDIR][j];

        dh.Vc(RHO, k, j, i) = rho0;
        dh.Vc(PRS, k, j, i) = P0;
        dh.Vc(VX1, k, j, i) = -std::sin(twopi * y);
        dh.Vc(VX2, k, j, i) =  std::sin(twopi * x);

        // Cell-centred B (used as initial guess; CT will correct from Az)
        dh.Vc(BX1, k, j, i) = -std::sin(twopi * y) * B0;
        dh.Vc(BX2, k, j, i) =  std::sin(4.0 * pi * x) * B0;
      }
    }
  }

  // Initialise the face-centred magnetic field via vector potential Az
  // Az = cos(4*pi*x)/(4*pi*B0_norm) + cos(2*pi*y)/(2*pi*B0_norm)
  // where the normalisation reproduces Bx = -sin(2*pi*y)*B0, By = sin(4*pi*x)*B0
  for (int k = 0; k < dh.np_tot[KDIR]; k++) {
    for (int j = 0; j < dh.np_tot[JDIR] + JOFFSET; j++) {
      for (int i = 0; i < dh.np_tot[IDIR] + IOFFSET; i++) {

        real xf = dh.xl[IDIR][i];   // left face x
        real yf = dh.xl[JDIR][j];   // left face y

        real Az_ij   = std::cos(4.0*pi*xf)  / (4.0*pi) * B0 +
                       std::cos(twopi*yf)   / (twopi)  * B0;
        real Az_ip1j = std::cos(4.0*pi*(xf + dh.dx[IDIR][i])) / (4.0*pi) * B0 +
                       std::cos(twopi*yf)                       / (twopi)  * B0;
        real Az_ijp1 = std::cos(4.0*pi*xf)  / (4.0*pi) * B0 +
                       std::cos(twopi*(yf + dh.dx[JDIR][j]))   / (twopi)  * B0;

        // Bx face = (Az(i,j+1) - Az(i,j)) / dy  (sign from curl)
        if (i < dh.np_tot[IDIR]) {
          dh.Vs(BX1s, k, j, i) = -(Az_ijp1 - Az_ij) / dh.dx[JDIR][j];
        }
        // By face = -(Az(i+1,j) - Az(i,j)) / dx
        if (j < dh.np_tot[JDIR]) {
          dh.Vs(BX2s, k, j, i) = (Az_ip1j - Az_ij) / dh.dx[IDIR][i];
        }
      }
    }
  }

  dh.SyncToDevice();
}
