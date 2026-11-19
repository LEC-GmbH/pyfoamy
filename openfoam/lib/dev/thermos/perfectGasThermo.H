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
    Trait: a thermo-physics type with its equation of state replaced by
    perfectGas.

\*---------------------------------------------------------------------------*/

#ifndef perfectGasThermo_H
#define perfectGasThermo_H

#include "perfectGas.H"
#include "thermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- ThermoType with its equation of state replaced by perfectGas
template<class ThermoType>
struct perfectGasThermo;

template
<
    template<class> class Transport,
    template<class> class Thermo,
    template<class> class EquationOfState,
    class Specie,
    template<class> class Type
>
struct perfectGasThermo
<
    Transport<species::thermo<Thermo<EquationOfState<Specie>>, Type>>
>
{
    typedef Transport<species::thermo<Thermo<perfectGas<Specie>>, Type>> type;
};

template
<
    template<class, int> class Transport,
    int PolySize,
    template<class> class Thermo,
    template<class> class EquationOfState,
    class Specie,
    template<class> class Type
>
struct perfectGasThermo
<
    Transport<species::thermo<Thermo<EquationOfState<Specie>>, Type>, PolySize>
>
{
    typedef
        Transport
        <
            species::thermo<Thermo<perfectGas<Specie>>, Type>,
            PolySize
        > type;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
