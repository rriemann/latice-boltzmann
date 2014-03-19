#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <array>
#include <cstddef>
#include <boost/format.hpp>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::array;

typedef double real; //!< define precision (choose from double, float, ...)

int main()
{
    //-------GLOBAL CONSTANTS-----------------
    const real pi = 4*atan(1);
    const real eps = std::numeric_limits<real>::epsilon();
    #ifdef _OPENMP
    const u_char num_procs = omp_get_num_procs(); //!< number of available processors
    #else
    const u_char num_procs = 1;
    #endif
    cerr << "processors in use: " << short(num_procs) << endl;

    //-------SIMULATION PARAMETERS------------
    constexpr u_char Nx = 21; //!< Number of lattices in the x-direction.
    constexpr u_char Ny = Nx; //!< Number of lattices in the y-direction.
    typedef real realm[Nx*Ny]; //!<  real matrix, column/x-wise
    u_char Length = Nx-1; //!< Length of the square computational domain in lattice units.
    real Re = 10.0; //!< Reynolds number.
    real tau = 0.65; //!< Relaxation time.
    real omega = 1.0/tau; //!< Relaxation frequency.
    real vlat = (tau-0.5)/3; //!< Lattice kinematic viscosity.
    real CsSquare = 1.0/3; //!< Square of the speed of sound in lattice units.
    real Ulat = Re*vlat/Length; //!< Lattice characteristic velocity.
    real Kx = 2*pi/Length; //!< Wavenumber in the x- and y-direction.
    real Ro = 1.0; //!< Initial fluid density in lattice and physical units.
    size_t Tsim = 20; //!< Simulation time.

    //-------LATTICE ARRANGEMENT PARAMETERS (D2Q9)-------
    constexpr u_char Nl = 9; //!< Number of lattice linkages.

    const real weight[9] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}; //!< Weighting factors.
    const char ex[9] = {0, 1, -1, 0, 0, 1, -1, -1, 1}; //!< X-component of the particle velocity.
    const char ey[9] = {0, 0, 0, 1, -1, 1, 1, -1, -1}; //!< Y-component of the particle velocity.


    //-------INITIALIZATION OF THE VARIABLES-------
    realm Ux; //!< X-component of the fluid velocity in the computational domain.
    realm Uy; //!< Y-component of the fluid velocity in the computational domain.
    realm rho; //!< Fluid density in the computational domain.
    realm Uexact; //!< X-component of the fluid velocity in the physical space.
    realm Vexact; //!< Y-component of the fluid velocity in the physical space.
    array<realm, Nl> feq; //!< Particle equilibrium distribution function.
    array<realm, Nl> f; //!< Particle distribution function.

    u_char orow[Nl] = {}; //!<  offset for streaming, row (y)
    u_char ocol[Nl] = {}; //!<  offset for streaming, col (x)

    //-------INITIALIZATION OF THE SIMULATION (t=0)-------
    #pragma omp parallel for
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i) {
            Ux[i+j*Nx] = -Ulat*cos(Kx*i)*sin(Kx*j);
            Uy[i+j*Nx] = +Ulat*sin(Kx*i)*cos(Kx*j);
            real P = Ro*CsSquare-0.25*Ulat*Ulat*(cos(2*Kx*i)+cos(2*Kx*j)); //!< Pressure defined in the computational space.
            rho[i+j*Nx] = P/CsSquare;
        }
    }

    #pragma omp parallel for
    for (u_char k = 0;  k <Nl; ++k) {
        for (u_char j = 0; j < Ny; ++j) {
            for (u_char i = 0; i < Nx; ++i) {
                real term1 = ex[k]*Ux[i+Nx*j]+ey[k]*Uy[i+j*Nx];
                real term2 = pow(Ux[i+j*Nx],2)+pow(Uy[i+j*Nx],2);
                feq[k][i+j*Nx] = weight[k]*rho[i+j*Nx]*(1+3*term1+4.5*term1*term1-1.5*term2);
                f  [k][i+j*Nx] = feq[k][i+j*Nx];
            }
        }
    }

    //-------MAIN LOOP-------
    for (u_char t = 0; t < Tsim; ++t) {
        //-------COLLISION-------
        #pragma omp parallel for
        for (u_char k = 0; k < Nl; ++k) {
            for (u_char j = 0; j < Ny; ++j) {
                for (u_char i = 0; i < Nx; ++i) {
                    f[k][i+j*Nx] = f[k][i+j*Nx]*(1-omega)+omega*feq[k][i+j*Nx];
                }
            }
        }

        //-------STREAMING-------
        for (u_char k = 0; k < Nl; ++k) {
            orow[k] = (Ny + orow[k] - ey[k]) % Ny;
            ocol[k] = (Nx + ocol[k] - ex[k]) % Nx;
        }

        //-------CALCULATION OF THE MACROSCOPIC VARIABLES-------
        #pragma omp parallel for
        for (u_char j = 0; j < Ny; ++j) {
            for (u_char i = 0; i < Nx; ++i) {
                real rsum = 0; //!< Fluid density counter.
                real usum = 0; //!< Counter of the x-component of the fluid velocity.
                real vsum = 0; //!< Counter of the y-component of the fluid velocity.
                for (u_char k = 0; k < Nl; ++k) {
                    // compute index
                    u_char nj = (Ny+j+orow[k]) % Ny;
                    u_char ni = (Nx+i+ocol[k]) % Nx;
                    rsum += f[k][ni+nj*Nx];
                    usum += f[k][ni+nj*Nx]*ex[k];
                    vsum += f[k][ni+nj*Nx]*ey[k];
                }
                rho[i+j*Nx] = rsum;
                Ux[i+j*Nx] = usum/rsum;
                Uy[i+j*Nx] = vsum/rsum;
            }
        }

        //-------CALCULATION OF THE EQUILIBRIUM DISTRIBUTION FUNCTION-------
        #pragma omp parallel for
        for (u_char j = 0; j < Ny; ++j) {
            for (u_char i = 0; i < Nx; ++i) {
                for (u_char k = 0; k < Nl; ++k) {
                    // compute index
                    u_char nj = (j+orow[k]+Ny) % Ny;
                    u_char ni = (i+ocol[k]+Nx) % Nx;

                    real term1 = ex[k]*Ux[i+Nx*j]+ey[k]*Uy[i+j*Nx];
                    real term2 = Ux[i+j*Nx]*Ux[i+j*Nx]+Uy[i+j*Nx]*Uy[i+j*Nx];
                    if(k == 0) {
                        assert(term1 == 0);
                    }
                    feq[k][ni+nj*Nx] = weight[k]*rho[i+j*Nx]*(1+3*term1+4.5*term1*term1-1.5*term2);
                }
            }
        }
    }

    //-------EXACT ANALYTICAL SOLUTION-------
    #pragma omp parallel for
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i) {
            Uexact[i+j*Nx] = -Ulat*cos(Kx*i)*sin(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
            Vexact[i+j*Nx] = +Ulat*sin(Kx*i)*cos(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
        }
    }

    //------- ABSOLUTE NUMERICAL ERROR (L2)-------
    real esum = 0; //!<  Counter for the error calculation.
    #pragma omp parallel for reduction(+:esum)
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i) {
            esum += pow(Ux[i+j*Nx]-Uexact[i+j*Nx],2)+pow(Uy[i+j*Nx]-Vexact[i+j*Nx],2);
        }
    }
    real AbsL2error = sqrt(esum/(Nx*Ny));
    cout << "error: " << (boost::format(" %1.20e") % AbsL2error) << endl;
    assert(fabs(0.00087126772875501965962-AbsL2error) <= eps);

    return 0;
}
