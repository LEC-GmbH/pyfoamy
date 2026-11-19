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

Description
    The thermo-physics type of the default fit (logPolynomial transport,
    janaf thermo, sensible enthalpy) for an equation of state.

\*---------------------------------------------------------------------------*/

#ifndef typedefPyfoamyThermo_H
#define typedefPyfoamyThermo_H

#include "specie.H"
#include "perfectGas.H"
#include "pyfoamyPengRobinsonGas.H"
#include "janafThermo.H"
#include "sensibleEnthalpy.H"
#include "logPolynomialTransport.H"
#include "thermo.H"
#include "typedefThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define typedefPyfoamyThermo(EquationOfState)                                     \
    typedefThermo                                                              \
    (                                                                          \
        logPolynomialTransport,                                                \
        sensibleEnthalpy,                                                      \
        janafThermo,                                                           \
        EquationOfState,                                                       \
        specie                                                                 \
    )

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
