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

\*---------------------------------------------------------------------------*/

#include "valueBInhomogeneousMixture.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::valueBInhomogeneousMixture<ThermoType>::valueBInhomogeneousMixture
(
    const dictionary& dict
)
:
    bInhomogeneousMixture(dict),
    fuel_("fuel", dict.subDict("fuel")),
    oxidant_("oxidant", dict.subDict("oxidant")),
    products_("products", dict.subDict("products")),
    thermoMixture_({&fuel_, &oxidant_, &products_}),
    transportMixture_({&fuel_, &oxidant_, &products_})
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
const typename Foam::valueBInhomogeneousMixture<ThermoType>::thermoMixtureType&
Foam::valueBInhomogeneousMixture<ThermoType>::thermoMixture
(
    const scalarFieldListSlice& Y
) const
{
    const scalar ft = Y[FT];

    if (ft < 0.0001)
    {
        thermoMixture_.setWeights({scalar(0), scalar(1), scalar(0)});
    }
    else
    {
        const scalar fu = max(ft - (1 - ft)/stoicRatio_, scalar(0));
        const scalar ox = 1 - ft - (ft - fu)*stoicRatio_;
        const scalar pr = 1 - fu - ox;

        thermoMixture_.setWeights({fu, ox, pr});
    }

    return thermoMixture_;
}


template<class ThermoType>
const typename
Foam::valueBInhomogeneousMixture<ThermoType>::transportMixtureType&
Foam::valueBInhomogeneousMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice& Y
) const
{
    const thermoMixtureType& tm = thermoMixture(Y);

    transportMixture_.setMassWeights({tm.Y(0), tm.Y(1), tm.Y(2)});

    return transportMixture_;
}


template<class ThermoType>
const typename
Foam::valueBInhomogeneousMixture<ThermoType>::transportMixtureType&
Foam::valueBInhomogeneousMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice& Y,
    const thermoMixtureType&
) const
{
    return transportMixture(Y);
}


template<class ThermoType>
void Foam::valueBInhomogeneousMixture<ThermoType>::read(const dictionary& dict)
{
    bInhomogeneousMixture::read(dict);
    fuel_ = ThermoType("fuel", dict.subDict("fuel"));
    oxidant_ = ThermoType("oxidant", dict.subDict("oxidant"));
    products_ = ThermoType("products", dict.subDict("products"));
}


// ************************************************************************* //
