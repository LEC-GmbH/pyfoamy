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

\*---------------------------------------------------------------------------*/

#include "pyfoamyPengRobinsonGas.H"
#include "IOstreams.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Specie>
Foam::pyfoamyPengRobinsonGas<Specie>::pyfoamyPengRobinsonGas
(
    const word& name,
    const dictionary& dict,
    const dictionary& subDict
)
:
    Specie(name, dict)
{
    static const scalar OmegaA = 0.45723552892138218;
    static const scalar OmegaB = 0.0777960739038885;

    const scalar Tc = subDict.lookup<scalar>("Tc", dimensions::temperature);
    const scalar Pc = subDict.lookup<scalar>("Pc", dimensions::pressure);
    const scalar k = kappa(subDict.lookup<scalar>("omega", dimless));

    const scalar sqrtA = sqrt(OmegaA/Pc)*RR*Tc;

    A0_ = sqrtA*(1 + k);
    A1_ = sqrtA*k/sqrt(Tc);
    b_ = OmegaB*RR*Tc/Pc;
}


template<class Specie>
Foam::pyfoamyPengRobinsonGas<Specie>::pyfoamyPengRobinsonGas
(
    const word& name,
    const dictionary& dict
)
:
    pyfoamyPengRobinsonGas(name, dict, dict.subDict("equationOfState"))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Specie>
Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::kappa(const scalar omega)
{
    return
        omega <= 0.491
      ? 0.37464 + omega*(1.54226 - 0.26992*omega)
      : 0.374642 + omega*(1.487503 + omega*(-0.164423 + 0.016666*omega));
}


template<class Specie>
void Foam::pyfoamyPengRobinsonGas<Specie>::write(Ostream& os) const
{
    Specie::write(os);
}


// * * * * * * * * * * * * * * * Ostream Operator  * * * * * * * * * * * * * //

template<class Specie>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const pyfoamyPengRobinsonGas<Specie>& pg
)
{
    pg.write(os);
    return os;
}


// ************************************************************************* //
