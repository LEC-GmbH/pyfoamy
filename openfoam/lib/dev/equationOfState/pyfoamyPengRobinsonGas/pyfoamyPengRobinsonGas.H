/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2014-2026 OpenFOAM Foundation
    Copyright (C) 2026 LEC - Large Engines Competence Center
-------------------------------------------------------------------------------
License
    This file is part of pyfoamy's OpenFOAM extensions.
    Derived from OpenFOAM.

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::pyfoamyPengRobinsonGas

Description
    Peng-Robinson cubic equation of state for gases and gas mixtures, with
    van der Waals one-fluid mixing as Cantera's PengRobinsonPhase.

    Formulation, per kmol, with the universal gas constant R:
    \verbatim
        p = R T/(v - b) - a alpha(T)/(v^2 + 2 b v - b^2)

        a     = Omega_a R^2 Tc^2/Pc
        b     = Omega_b R Tc/Pc
        alpha = (1 + kappa (1 - sqrt(T/Tc)))^2
    \endverbatim
    with A = a alpha p/(R T)^2 and B = b p/(R T) the compression factor
    Z = p v/(R T) solves
    \verbatim
        Z^3 + (B - 1) Z^2 + (A - 2 B - 3 B^2) Z - (A B - B^2 - B^3) = 0
    \endverbatim
    The class returns the departures from the ideal gas: the enthalpy,
    internal energy, heat capacities and entropy contributions, added to the
    ideal-gas thermo by the thermo model, and rho, psi, Z, CpMCv and alphav
    (the latter needed by the Jacobian of the standard chemistry model).

    Mixing:
        sqrt(a alpha) is linear in sqrt(T):
        \verbatim
            sqrt(a alpha) = A0 - A1 sqrt(T)
            A0 = sqrt(a) (1 + kappa),  A1 = sqrt(a) kappa/sqrt(Tc)
        \endverbatim
        The class stores A0, A1 and b. The van der Waals one-fluid rule with
        zero binary interaction parameters,
        \verbatim
            a alpha_mix = (sum_i x_i sqrt(a_i alpha_i))^2,  b_mix = sum_i x_i b_i
        \endverbatim
        is then exact as mole-fraction-weighted sums of A0, A1 and b, so a
        coefficient mixture (operator+=) is the one-fluid mixture at every
        temperature. valueMulticomponentMixture instead mixes the species'
        properties, each species at the mixture's p and T.

    Constants, as Cantera's PengRobinsonPhase:
        \verbatim
            Omega_a = 0.45723552892138218
            Omega_b = 0.0777960739038885
            kappa   = 0.37464 + 1.54226 omega - 0.26992 omega^2,
                      omega <= 0.491
            kappa   = 0.374642 + 1.487503 omega - 0.164423 omega^2
                    + 0.016666 omega^3,  omega > 0.491
        \endverbatim

    Root choice:
        Where the cubic has three real roots the largest (vapour) root is
        taken, as Cantera from a gas state.

    Validity and limits:
        - A gas below its saturation pressure; where three roots exist and
          the liquid is the stable phase the result is the metastable
          (supersaturated) vapour. Where the vapour root vanishes, Z jumps to
          the liquid branch: density is discontinuous in T there.
        - Near a (pseudo-)critical point Cp grows large, which slows or
          stalls the inversion of T from the enthalpy.
        - Above Tc ((1 + kappa)/kappa)^2 (about 1400 K for N2), sqrt(a alpha)
          of a species changes sign. The one-fluid sum takes it signed,
          Cantera its magnitude: mixtures there differ from Cantera, by
          about 1e-3 in Z for burnt gas at 150 bar, 2000 K, 3e-3 for fuel
          vapour in N2. Pure species are unaffected.
        - The kappa fits do not meet at omega = 0.491; a pseudo-species
          lumped from a mixture whose kappa falls between them is given
          omega = 0.491.

    Reference:
        Cantera, PengRobinsonPhase, https://cantera.org - the source of the
        mixing rule and the constants and the reference of the unit tests.

Usage
    \table
        Property     | Description                    | Required
        Tc           | Critical temperature [K]       | yes
        Pc           | Critical pressure [Pa]         | yes
        omega        | Acentric factor [-]            | yes
    \endtable

    Example specification of the equation of state for methane:
    \verbatim
    equationOfState
    {
        Tc        190.56;
        Pc        4.599e6;
        omega     0.011;
    }
    \endverbatim
    with thermoType's equationOfState pyfoamyPengRobinsonGas.

SourceFiles
    pyfoamyPengRobinsonGasI.H
    pyfoamyPengRobinsonGas.C

\*---------------------------------------------------------------------------*/

#ifndef pyfoamyPengRobinsonGas_H
#define pyfoamyPengRobinsonGas_H

#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

template<class Specie> class pyfoamyPengRobinsonGas;

template<class Specie>
inline pyfoamyPengRobinsonGas<Specie> operator+
(
    const pyfoamyPengRobinsonGas<Specie>&,
    const pyfoamyPengRobinsonGas<Specie>&
);

template<class Specie>
inline pyfoamyPengRobinsonGas<Specie> operator*
(
    const scalar,
    const pyfoamyPengRobinsonGas<Specie>&
);

template<class Specie>
inline pyfoamyPengRobinsonGas<Specie> operator==
(
    const pyfoamyPengRobinsonGas<Specie>&,
    const pyfoamyPengRobinsonGas<Specie>&
);

template<class Specie>
Ostream& operator<<
(
    Ostream&,
    const pyfoamyPengRobinsonGas<Specie>&
);


/*---------------------------------------------------------------------------*\
                   Class pyfoamyPengRobinsonGas Declaration
\*---------------------------------------------------------------------------*/

template<class Specie>
class pyfoamyPengRobinsonGas
:
    public Specie
{
    // Private Data

        //- sqrt(a alpha) at T = 0 [sqrt(J m^3)/kmol]
        scalar A0_;

        //- Slope of sqrt(a alpha) over sqrt(T) [sqrt(J m^3/K)/kmol]
        scalar A1_;

        //- Co-volume [m^3/kmol]
        scalar b_;


    // Private Member Functions

        //- sqrt(a alpha) and its first and second T derivatives
        inline void sqrtAAlpha
        (
            const scalar T,
            scalar& s,
            scalar& dsdT,
            scalar& d2sdT2
        ) const;

        //- A and B of the cubic
        inline void AB
        (
            const scalar p,
            const scalar T,
            scalar& A,
            scalar& B
        ) const;

        //- ln((Z + (1 + sqrt2) B)/(Z + (1 - sqrt2) B))/(2 sqrt2 b)
        inline scalar logTerm(const scalar Z, const scalar B) const;


    // Private Constructors

        //- Construct from name and parent dictionary and dictionary
        pyfoamyPengRobinsonGas
        (
            const word& name,
            const dictionary& dict,
            const dictionary& subDict
        );


public:

    // Constructors

        //- Construct from components
        inline pyfoamyPengRobinsonGas
        (
            const Specie& sp,
            const scalar A0,
            const scalar A1,
            const scalar b
        );

        //- Construct from name and dictionary
        pyfoamyPengRobinsonGas(const word& name, const dictionary& dict);

        //- Construct as named copy
        inline pyfoamyPengRobinsonGas
        (
            const word& name,
            const pyfoamyPengRobinsonGas&
        );

        //- Construct and return a clone
        inline autoPtr<pyfoamyPengRobinsonGas> clone() const;


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return word("pyfoamyPengRobinsonGas<", word(Specie::typeName_()), '>');
        }

        //- kappa of an acentric factor
        static scalar kappa(const scalar omega);


        // Fundamental properties

            //- Is the equation of state is incompressible i.e. rho != f(p)
            static const bool incompressible = false;

            //- Is the equation of state is isochoric i.e. rho = const
            static const bool isochoric = false;

            //- Return density [kg/m^3]
            inline scalar rho(scalar p, scalar T) const;

            //- Return enthalpy contribution [J/kg]
            inline scalar h(const scalar p, const scalar T) const;

            //- Return Cp contribution [J/kg/K]
            inline scalar Cp(scalar p, scalar T) const;

            //- Return internal energy contribution [J/kg]
            inline scalar e(const scalar p, const scalar T) const;

            //- Return Cv contribution [J/kg/K]
            inline scalar Cv(scalar p, scalar T) const;

            //- Return entropy contribution to the integral of Cp/T [J/kg/K]
            inline scalar sp(const scalar p, const scalar T) const;

            //- Return entropy contribution to the integral of Cv/T [J/kg/K]
            inline scalar sv(const scalar p, const scalar T) const;

            //- Return compressibility [s^2/m^2]
            inline scalar psi(scalar p, scalar T) const;

            //- Return compression factor [-]
            inline scalar Z(scalar p, scalar T) const;

            //- Return (Cp - Cv) [J/kg/K]
            inline scalar CpMCv(scalar p, scalar T) const;

            //- Return volumetric coefficient of thermal expansion [1/K]
            inline scalar alphav(const scalar p, const scalar T) const;


        // IO

            //- Write to Ostream
            void write(Ostream& os) const;


    // Member Operators

        inline void operator+=(const pyfoamyPengRobinsonGas&);
        inline void operator*=(const scalar);


    // Friend operators

        friend pyfoamyPengRobinsonGas operator+ <Specie>
        (
            const pyfoamyPengRobinsonGas&,
            const pyfoamyPengRobinsonGas&
        );

        friend pyfoamyPengRobinsonGas operator* <Specie>
        (
            const scalar s,
            const pyfoamyPengRobinsonGas&
        );

        //- Finite coefficients (pg1's) for the reaction thermo
        friend pyfoamyPengRobinsonGas operator== <Specie>
        (
            const pyfoamyPengRobinsonGas&,
            const pyfoamyPengRobinsonGas&
        );


    // Ostream Operator

        friend Ostream& operator<< <Specie>
        (
            Ostream&,
            const pyfoamyPengRobinsonGas&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "pyfoamyPengRobinsonGasI.H"

#ifdef NoRepository
    #include "pyfoamyPengRobinsonGas.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
