#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <array>
#include <cstddef>
#include <chrono>
#include <boost/format.hpp>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::array;

typedef double real; //!< define precision (choose from double, float, ...)
constexpr u_short Nx = 21; //!< Number of lattices in the x-direction.
constexpr u_short Ny = Nx; //!< Number of lattices in the y-direction.
constexpr u_char Nl = 9; //!< Number of lattice linkages.

struct point_t {
    real f[Nl]; //!< Particle distribution function.
    real feq[Nl]; //!< Particle equilibrium distribution function.
    real fbak;

    real Ux; //!< X-component of the fluid velocity in the computational domain.
    real Uy; //!< Y-component of the fluid velocity in the computational domain.
    real rho; //!< Fluid density in the computational domain.
};

//-------GLOBAL CONSTANTS-----------------
const real pi = 4*atan(1);
const real eps = std::numeric_limits<real>::epsilon();
//-------SIMULATION PARAMETERS------------
const u_short Length = Nx-1; //!< Length of the square computational domain in lattice units.
const real Re = 10.0; //!< Reynolds number.
const real tau = 0.65; //!< Relaxation time.
const real omega = 1.0/tau; //!< Relaxation frequency.
const real vlat = (tau-0.5)/3; //!< Lattice kinematic viscosity.
const real CsSquare = 1.0/3; //!< Square of the speed of sound in lattice units.
const real Ulat = Re*vlat/Length; //!< Lattice characteristic velocity.
const real Kx = 2*pi/Length; //!< Wavenumber in the x- and y-direction.
const real Ro = 1.0; //!< Initial fluid density in lattice and physical units.
const size_t Tsim = 20; //!< Simulation time.
//-------LATTICE ARRANGEMENT PARAMETERS (D2Q9)-------
const real weight[9] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}; //!< Weighting factors.
const char ex[9] = {0, 1, -1, 0, 0, 1, -1, -1, 1}; //!< X-component of the particle velocity.
const char ey[9] = {0, 0, 0, 1, -1, 1, 1, -1, -1}; //!< Y-component of the particle velocity.

inline void getTheory(const u_short i, const u_short j, real &Uexact, real &Vexact, const size_t &Tsim) {
    Uexact = -Ulat*cos(Kx*i)*sin(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
    Vexact = +Ulat*sin(Kx*i)*cos(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
}

inline void equilibriumHelper(point_t &p) {
    real U2 = pow(p.Ux,2)+pow(p.Uy,2);
    for (u_char k = 0; k < Nl; ++k) {
        real term1 = ex[k]*p.Ux+ey[k]*p.Uy;
        p.feq[k] = weight[k]*p.rho*(1+3*term1+4.5*term1*term1-1.5*U2);
    }
}

int main()
{
    auto start = std::chrono::steady_clock::now(); //!< start time of the app

    #ifdef _OPENMP
    const u_char num_procs = omp_get_num_procs(); //!< number of available processors
    #else
    const u_char num_procs = 1;
    #endif
    cerr << "processors in use: " << short(num_procs) << endl;

    //-------INITIALIZATION OF THE VARIABLES-------
    array<point_t, Nx*Ny> points;

    //-------INITIALIZATION OF THE SIMULATION (t=0)-------
    #pragma omp parallel for
    for (u_short j = 0; j < Ny; ++j) {
        for (u_short i = 0; i < Nx; ++i) {
            point_t &p = points[i+Nx*j];

            // p.Ux = -Ulat*cos(Kx*i)*sin(Kx*j);
            // p.Uy = +Ulat*sin(Kx*i)*cos(Kx*j);
            getTheory(i, j, p.Ux, p.Uy, 0);

            real P = Ro*CsSquare-0.25*Ulat*Ulat*(cos(2*Kx*i)+cos(2*Kx*j)); //!< Pressure defined in the computational space.
            p.rho = P/CsSquare;

            equilibriumHelper(p);
            std::copy(std::begin(p.feq), std::end(p.feq), std::begin(p.f));
            // memcpy(p.f, p.feq, sizeof(p.feq)); // pure C
        }
    }

    //-------MAIN LOOP-------
    for (size_t t = 0; t < Tsim; ++t) {
        for (u_char k = 0; k < Nl; ++k) {
            //-------COLLISION-------
            #pragma omp parallel for collapse(2)
            for (u_short j = 0; j < Ny; ++j) {
                for (u_short i = 0; i < Nx; ++i) {
                    point_t &p = points[i+Nx*j];
                    p.fbak = p.f[k]*(1-omega)+omega*p.feq[k];
                }
            }
            //-------STREAMING-------
            #pragma omp parallel for collapse(2)
            for (u_short j = 0; j < Ny; ++j) {
                for (u_short i = 0; i < Nx; ++i) {
                    point_t &p = points[i+Nx*j];

                    // compute index
                    u_short nj = (Ny+j-ey[k]) % Ny;
                    u_short ni = (Nx+i-ex[k]) % Nx;
                    point_t &np = points.at(ni+Nx*nj);
                    p.f[k] = np.fbak;
                }
            }
        }

        //-------CALCULATION OF THE MACROSCOPIC VARIABLES-------
        #pragma omp parallel for collapse(2)
        for (u_short j = 0; j < Ny; ++j) {
            for (u_short i = 0; i < Nx; ++i) {
                point_t &p = points[i+Nx*j];
                real rsum = 0; //!< Fluid density counter.
                real usum = 0; //!< Counter of the x-component of the fluid velocity.
                real vsum = 0; //!< Counter of the y-component of the fluid velocity.
                for (u_char k = 0; k < Nl; ++k) {
                    rsum += p.f[k];
                    usum += p.f[k]*ex[k];
                    vsum += p.f[k]*ey[k];
                }
                p.rho = rsum;
                p.Ux  = usum/rsum;
                p.Uy  = vsum/rsum;
            }
        }

        //-------CALCULATION OF THE EQUILIBRIUM DISTRIBUTION FUNCTION-------
        #pragma omp parallel for collapse(2)
        for (u_short j = 0; j < Ny; ++j) {
            for (u_short i = 0; i < Nx; ++i) {
                point_t &p = points[i+Nx*j];
                equilibriumHelper(p);
            }
        }
    }

    //------- ABSOLUTE NUMERICAL ERROR (L2)-------
    real esum = 0; //!<  Counter for the error calculation.
    real Uexact;
    real Vexact;
    #pragma omp parallel for collapse(2) reduction(+:esum) private(Uexact,Vexact)
    for (u_short j = 0; j < Ny; ++j) {
        for (u_short i = 0; i < Nx; ++i) {
            point_t &p = points[i+Nx*j];
            getTheory(i, j, Uexact, Vexact, Tsim);
            esum += pow(p.Ux-Uexact,2)+pow(p.Uy-Vexact,2);
        }
    }

    //-------OUTPUT----------
    // calculate absolute error in L^2 norm and output
    real AbsL2error = sqrt(esum/(Nx*Ny));
    cerr << "error: " << (boost::format(" %1.20e") % AbsL2error) << endl;
    assert(fabs(0.00087126772875501965962-AbsL2error) <= eps); // Tsim = 20, Nx = 21
    // assert(fabs(0.000557275730831370335813-AbsL2error) <= eps); // Tsim = 1, Nx = 21

    // calculate runtime and output
    auto done = std::chrono::steady_clock::now(); //!< finishing time of the app
    double elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(done - start).count();
    cerr << "calculation time: " << elapsed_time << std::endl;

    return 0;
}
