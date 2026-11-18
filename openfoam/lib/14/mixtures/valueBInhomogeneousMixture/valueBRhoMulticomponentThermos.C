/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2025-2026 OpenFOAM Foundation
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

Description
    Registers valueBInhomogeneousMixture as a selectable "mixture" for
    bRhoMulticomponentThermo (i.e. physicalProperties.b's thermoType) -
    see valueURhoMulticomponentThermos.C for the full rationale, this
    is the identical registration for the burnt side.

\*---------------------------------------------------------------------------*/

#include "bRhoMulticomponentThermo.H"
#include "valueBInhomogeneousMixture.H"

#include "forGases.H"
#include "logPolynomialTransport.H"

#include "makeThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define makeValueBRhoMulticomponentThermos(Mixture, ThermoPhysics)            \
                                                                               \
    defineThermo(bRhoMulticomponentThermo, Mixture, ThermoPhysics);           \
                                                                               \
    addThermo(basicThermo, bRhoMulticomponentThermo, Mixture, ThermoPhysics); \
    addThermo(fluidThermo, bRhoMulticomponentThermo, Mixture, ThermoPhysics); \
    addThermo                                                                 \
    (                                                                         \
        rhoFluidThermo,                                                       \
        bRhoMulticomponentThermo,                                             \
        Mixture,                                                              \
        ThermoPhysics                                                        \
    );                                                                        \
    addThermo                                                                 \
    (                                                                         \
        bRhoMulticomponentThermo,                                             \
        bRhoMulticomponentThermo,                                            \
        Mixture,                                                              \
        ThermoPhysics                                                        \
    )

#define forPyfoamyValueGasTransports(Macro, Args...)                          \
    forThermo                                                                 \
    (                                                                         \
        sutherlandTransport, sensibleEnthalpy, janafThermo, perfectGas,       \
        specie, Macro, Args                                                   \
    );                                                                        \
    forThermo                                                                 \
    (                                                                         \
        polynomialTransport, sensibleEnthalpy, janafThermo, perfectGas,       \
        specie, Macro, Args                                                  \
    );                                                                        \
    forThermo                                                                 \
    (                                                                         \
        logPolynomialTransport, sensibleEnthalpy, janafThermo, perfectGas,     \
        specie, Macro, Args                                                  \
    )

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    forPyfoamyValueGasTransports
    (
        makeValueBRhoMulticomponentThermos,
        valueBInhomogeneousMixture
    );
}

// ************************************************************************* //
