/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2026 OpenFOAM Foundation
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
    Foam::pyfoamyIdealSpecieThermo

Description
    Multicomponent thermo whose per-specie properties (rhoi, Cpi, hei, hsi,
    hai) are those of the perfect gas, while the mixture keeps its own (e.g.
    real-gas) equation of state.

    These per-specie values are the enthalpy a specie carries into the
    mixture (spray evaporation, species diffusion). Under coefficient mixing
    the pure specie at the cell pressure is far off the partial value (heavy
    fuel vapour near its critical pressure is liquid-like); the perfect gas
    is within a few percent of it.

SourceFiles
    pyfoamyIdealSpecieThermo.C

\*---------------------------------------------------------------------------*/

#ifndef pyfoamyIdealSpecieThermo_H
#define pyfoamyIdealSpecieThermo_H

#include "perfectGasThermo.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*                    Class pyfoamyIdealSpecieThermo Declaration
\*---------------------------------------------------------------------------*/

template<class BaseThermo>
class pyfoamyIdealSpecieThermo
:
    public BaseThermo
{
    // Private Typedefs

        typedef typename perfectGasThermo
        <
            typename BaseThermo::mixtureType::thermoType
        >::type idealThermoType;


    // Private Data

        //- Perfect-gas specie thermo
        PtrList<idealThermoType> idealThermos_;


    // Private Member Functions

        template<class Method>
        tmp<scalarField> property
        (
            Method method,
            const label speciei,
            const scalarField& p,
            const scalarField& T
        ) const;

        template<class Method>
        tmp<volScalarField::Internal> property
        (
            const word& name,
            const dimensionSet& dims,
            Method method,
            const label speciei,
            const volScalarField::Internal& p,
            const volScalarField::Internal& T
        ) const;

        template<class Method>
        tmp<volScalarField> property
        (
            const word& name,
            const dimensionSet& dims,
            Method method,
            const label speciei,
            const volScalarField& p,
            const volScalarField& T
        ) const;


public:

    // Constructors

        //- Construct from mesh and phase name
        pyfoamyIdealSpecieThermo(const fvMesh&, const word& phaseName);


    //- Destructor
    virtual ~pyfoamyIdealSpecieThermo();


    // Member Functions

        // Specie thermodynamic properties

            virtual scalar rhoi
            (
                const label speciei,
                const scalar p,
                const scalar T
            ) const;

            virtual tmp<volScalarField> rhoi
            (
                const label speciei,
                const volScalarField& p,
                const volScalarField& T
            ) const;

            virtual scalar Cpi
            (
                const label speciei,
                const scalar p,
                const scalar T
            ) const;

            virtual tmp<volScalarField> Cpi
            (
                const label speciei,
                const volScalarField& p,
                const volScalarField& T
            ) const;

            #define pyfoamyIdealSpecieThermoEnthalpy(Name)                         \
                virtual scalar Name##i                                         \
                (                                                              \
                    const label speciei,                                       \
                    const scalar p,                                            \
                    const scalar T                                             \
                ) const;                                                       \
                                                                               \
                virtual tmp<scalarField> Name##i                               \
                (                                                              \
                    const label speciei,                                       \
                    const scalarField& p,                                      \
                    const scalarField& T                                       \
                ) const;                                                       \
                                                                               \
                virtual tmp<volScalarField::Internal> Name##i                  \
                (                                                              \
                    const label speciei,                                       \
                    const volScalarField::Internal& p,                         \
                    const volScalarField::Internal& T                          \
                ) const;                                                       \
                                                                               \
                virtual tmp<volScalarField> Name##i                            \
                (                                                              \
                    const label speciei,                                       \
                    const volScalarField& p,                                   \
                    const volScalarField& T                                    \
                ) const;

            pyfoamyIdealSpecieThermoEnthalpy(he)
            pyfoamyIdealSpecieThermoEnthalpy(hs)
            pyfoamyIdealSpecieThermoEnthalpy(ha)

            #undef pyfoamyIdealSpecieThermoEnthalpy
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "pyfoamyIdealSpecieThermo.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
