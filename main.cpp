#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <array>
#include <cstddef>
#include <boost/format.hpp>

using namespace std;

typedef double real;

int main()
{

    constexpr real pi = 4*atan(1);
    //-------SIMULATION PARAMETERS-------
    constexpr u_char Nx = 21; //Number of lattices in the x-direction.
    constexpr u_char Ny = Nx; //Number of lattices in the y-direction.
    u_char Length = Nx-1; //Length of the square computational domain in lattice units.
    real Re = 10.0; //Reynolds number.
    real tau = 0.65; //Relaxation time.
    real omega = 1.0/tau; //Relaxation frequency.
    real vlat = (tau-0.5)/3; //Lattice kinematic viscosity.
    real CsSquare = 1.0/3; //Square of the speed of sound in lattice units.
    real Uo = 1.0; //Initial maximum fluid velocity in physical units.
    real Ulat = Re*vlat/Length; //Lattice characteristic velocity.
    real Kx = 2*pi/Length; //Wavenumber in the x- and y-direction.
    real Cu = Uo/Ulat; //Conversion factor for velocity.
    real Ct = 2*Ulat/Length; //Conversion factor for time.
    real Ro = 1.0; //Initial fluid density in lattice and physical units.
    size_t Tsim = 20; //Simulation time.
    //-------LATTICE ARRANGEMENT PARAMETERS (D2Q9)-------
    constexpr u_char Nl = 9; //Number of lattice linkages.

    const real weight[9] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}; //Weighting factors.
    const char ex[9] = {0, 1, -1, 0, 0, 1, -1, -1, 1}; //X-component of the particle velocity.
    const char ey[9] = {0, 0, 0, 1, -1, 1, 1, -1, -1}; //Y-component of the particle velocity.

    typedef real realm[Nx*Ny]; // real matrix, column/x-wise


    //-------INITIALIZATION OF THE VARIABLES-------
    realm Ux; //X-component of the fluid velocity in the computational domain.
    realm Uy; //Y-component of the fluid velocity in the computational domain.
    realm rho; //Fluid density in the computational domain.
    realm Uexact; //X-component of the fluid velocity in the physical space.
    realm Vexact; //Y-component of the fluid velocity in the physical space.
    //   real Rexact[Ny*Nx]; //Fluid density in the physical space.
    std::array<realm, Nl> feq; //Particle equilibrium distribution function.
    std::array<realm, Nl> f; //Particle distribution function.

    u_char orow[Nl] = {}; // offset for streaming, row (y)
    u_char ocol[Nl] = {}; // offset for streaming, col (x)
    //-------INITIALIZATION OF THE SIMULATION (t=0)-------
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i){
            Ux[i+j*Nx] = -Ulat*cos(Kx*i)*sin(Kx*j);
            Uy[i+j*Nx] = Ulat*sin(Kx*i)*cos(Kx*j);
            real P = Ro*CsSquare-0.25*Ulat*Ulat*(cos(2*Kx*i)+cos(2*Kx*j)); //Pressure defined in the computational space.
            rho[i+j*Nx] = P/CsSquare;

        }
    }
    /* {
        real sum = 0;
        for (size_t i = 0; i < Nx*Ny; ++i) {
            sum += rho[i];
        }
        std::cerr << "rho: " << sum << std::endl;
    } */

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
    /* {
        // debug
        for (u_char k = 0; k < Nl; ++ k) {
            for (u_char j = 0; j < Ny; ++j) {
                for (u_char i = 0; i < Nx; ++i) {
                    std::cerr << (boost::format(" %1.8e") % feq[k][i+j*Nx]);
                }
                std::cerr << std::endl;
            }
            std::cerr << std::endl << std::endl;
        }
    } */

    //-------MAIN LOOP-------
    for (u_char t = 0; t < Tsim; ++t) {
        //-------COLLISION-------
        for (u_char k = 0; k < Nl; ++k) {
            for (u_char j = 0; j < Ny; ++j) {
                for (u_char i = 0; i < Nx; ++i){
                    f[k][i+j*Nx] = f[k][i+j*Nx]*(1-omega)+omega*feq[k][i+j*Nx];
                }
            }
        }
        /* if (t == 1) {
            // debug
            for (u_char k = 0; k < Nl; ++ k) {
                for (u_char j = 0; j < Ny; ++j) {
                    for (u_char i = 0; i < Nx; ++i) {
                        std::cerr << (boost::format(" %1.8e") % feq[k][i+j*Nx]);
                    }
                    std::cerr << std::endl;
                }
                std::cerr << std::endl;
            }
        } */
        //-------STREAMING-------
        for (u_char k = 0; k < Nl; ++k) {
            orow[k] = (Ny + orow[k] - ey[k]) % Ny;
            ocol[k] = (Nx + ocol[k] - ex[k]) % Nx;
        }

        //-------CALCULATION OF THE MACROSCOPIC VARIABLES-------
        for (u_char j = 0; j < Ny; ++j) {
            for (u_char i = 0; i < Nx; ++i){
                real rsum = 0; //Fluid density counter.
                real usum = 0; //Counter of the x-component of the fluid velocity.
                real vsum = 0; //Counter of the y-component of the fluid velocity.
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
        /* if (t == 0) {
            // debug
            for (u_char j = 0; j < Ny; ++j) {
                for (u_char i = 0; i < Nx; ++i) {
                    std::cerr << (boost::format(" %1.8e") % rho[i+j*Nx]);
                }
                std::cerr << std::endl;
            }
        } */
        //-------CALCULATION OF THE EQUILIBRIUM DISTRIBUTION FUNCTION-------
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
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i) {
            Uexact[i+j*Nx] = -Ulat*cos(Kx*i)*sin(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
            Vexact[i+j*Nx] = +Ulat*sin(Kx*i)*cos(Kx*j)*exp(-2*Kx*Kx*vlat*Tsim);
        }
    }
    //------- ABSOLUTE NUMERICAL ERROR (L2)-------
    real esum = 0; // Counter for the error calculation.
    for (u_char j = 0; j < Ny; ++j) {
        for (u_char i = 0; i < Nx; ++i){
            esum += pow(Ux[i+j*Nx]-Uexact[i+j*Nx],2)+pow(Uy[i+j*Nx]-Vexact[i+j*Nx],2);
        }
    }
    real AbsL2error = sqrt(esum/(Nx*Ny));
    cout << "error: " << (boost::format(" %1.8e") % AbsL2error) << endl;

    return 0;
}
