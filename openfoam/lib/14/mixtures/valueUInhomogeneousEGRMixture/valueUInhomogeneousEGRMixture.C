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

#include "valueUInhomogeneousEGRMixture.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::valueUInhomogeneousEGRMixture<ThermoType>::valueUInhomogeneousEGRMixture
(
    const dictionary& dict
)
:
    uInhomogeneousEGRMixture(dict),
    fuel_("fuel", dict.subDict("fuel")),
    oxidant_("oxidant", dict.subDict("oxidant")),
    products_("products", dict.subDict("products")),
    thermoMixture_({&fuel_, &oxidant_, &products_}),
    transportMixture_({&fuel_, &oxidant_, &products_})
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
const ThermoType&
Foam::valueUInhomogeneousEGRMixture<ThermoType>::specieThermo
(
    const label speciei
) const
{
    switch (speciei)
    {
        case FU:
            return fuel_;
            break;

        case EGR:
            return products_;
            break;

        default:
            FatalErrorInFunction
                << "Cannot return specieThermo for specie " << speciei
                << exit(FatalError);
            return products_;
            break;
    }
}


template<class ThermoType>
const typename
Foam::valueUInhomogeneousEGRMixture<ThermoType>::thermoMixtureType&
Foam::valueUInhomogeneousEGRMixture<ThermoType>::thermoMixture
(
    const scalarFieldListSlice& Y
) const
{
    const scalar fu = Y[FU];
    const scalar egr = Y[EGR];

    if (fu < 0.0001 && egr < 0.0001)
    {
        thermoMixture_.setWeights({scalar(0), scalar(1), scalar(0)});
    }
    else
    {
        thermoMixture_.setWeights({fu, 1 - fu - egr, egr});
    }

    return thermoMixture_;
}


template<class ThermoType>
const typename
Foam::valueUInhomogeneousEGRMixture<ThermoType>::transportMixtureType&
Foam::valueUInhomogeneousEGRMixture<ThermoType>::transportMixture
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
Foam::valueUInhomogeneousEGRMixture<ThermoType>::transportMixtureType&
Foam::valueUInhomogeneousEGRMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice& Y,
    const thermoMixtureType&
) const
{
    return transportMixture(Y);
}


template<class ThermoType>
void Foam::valueUInhomogeneousEGRMixture<ThermoType>::read
(
    const dictionary& dict
)
{
    uInhomogeneousEGRMixture::read(dict);
    fuel_ = ThermoType("fuel", dict.subDict("fuel"));
    oxidant_ = ThermoType("oxidant", dict.subDict("oxidant"));
    products_ = ThermoType("products", dict.subDict("products"));
}


// ************************************************************************* //
