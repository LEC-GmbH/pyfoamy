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

\*---------------------------------------------------------------------------*/

#include "piecewisePhi.H"
#include "addToRunTimeSelectionTable.H"
#include "ListOps.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarFlameSpeedModels
{
    defineTypeNameAndDebug(piecewisePhi, 0);

    addToRunTimeSelectionTable
    (
        laminarFlameSpeed,
        piecewisePhi,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::laminarFlameSpeedModels::piecewisePhi::piecewisePhi
(
    const dictionary& dict,
    const dictionary& coeffDict,
    const uRhoMulticomponentThermo& ct
)
:
    laminarFlameSpeed(dict, ct)
{
    forAllConstIter(dictionary, coeffDict, iter)
    {
        if (iter().isDict())
        {
            const dictionary& subdict = iter().dict();

            if (subdict.found("Phi"))
            {
                Phis_.append(subdict.lookup<scalar>("Phi"));

                // Sl(p, T.u) at given constant Phi and zero EGR
                pTuFunctions_.append
                (
                    Function2<scalar>::New
                    (
                        "pTu",
                        dimensions::pressure,
                        dimensions::temperature,
                        dimensions::velocity,
                        subdict
                    )
                );

                fFunctions_.append
                (
                    Function2<scalar>::New
                    (
                        "f",
                        dimensions::pressure,
                        dimensions::temperature,
                        dimless,
                        subdict
                    )
                );
            }
        }
    }

    if (Phis_.size() == 0)
    {
        FatalErrorInFunction<< "No Phis specified" << exit(FatalError);
    }

    order_.resize(Phis_.size());
    sortedOrder(Phis_, order_);
    sort(Phis_);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::laminarFlameSpeedModels::piecewisePhi::~piecewisePhi()
{}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

inline Foam::label
Foam::laminarFlameSpeedModels::piecewisePhi::phiIndex(const scalar Phi)
const
{
    if (Phi <= Phis_.first())
    {
        return -1;
    }

    if (Phi >= Phis_.last())
    {
        return -2;
    }

    for (label i = 0; i < Phis_.size()-1; i++)
    {
        if (Phi >= Phis_[i] && Phi <= Phis_[i+1])
        {
            return i;
        }
    }

    // Should never happen
    return Phis_.size()-2;
}


inline Foam::scalar Foam::laminarFlameSpeedModels::piecewisePhi::SuRef
(
    const scalar p,
    const scalar Tu,
    const scalar Yegr,
    const label idx
) const
{
    return
        pTuFunctions_[order_[idx]]->value(p, Tu)
      * max(1 - fFunctions_[order_[idx]]->value(p, Tu)*Yegr, 0);
}


inline Foam::scalar Foam::laminarFlameSpeedModels::piecewisePhi::Su0pTPhi
(
    const scalar p,
    const scalar Tu,
    const scalar Phi,
    const scalar Yegr
) const
{
    const label i = phiIndex(Phi);

    if (i == -1)
    {
        return SuRef(p, Tu, Yegr, 0);
    }
    else if (i == -2)
    {
        return SuRef(p, Tu, Yegr, Phis_.size()-1);
    }

    const scalar Phi1 = Phis_[i];
    const scalar Phi2 = Phis_[i+1];
    const scalar x = (Phi2 - Phi)/(Phi2 - Phi1);

    const scalar Su1 = SuRef(p, Tu, Yegr, i);
    const scalar Su2 = SuRef(p, Tu, Yegr, i+1);

    return x*Su1 + (1-x)*Su2;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField> Foam::laminarFlameSpeedModels::piecewisePhi::Su
(
    const volScalarField& p,
    const volScalarField& Tu,
    const volScalarField& Phi
) const
{
    tmp<volScalarField> tSu0
    (
        volScalarField::New
        (
            "Su0",
            p.mesh(),
            dimensionedScalar(dimVelocity, 0)
        )
    );

    volScalarField& Su0 = tSu0.ref();

    forAll(Su0, celli)
    {
        Su0[celli] = Su0pTPhi(p[celli], Tu[celli], Phi[celli], 0);
    }

    volScalarField::Boundary& Su0Bf = Su0.boundaryFieldRef();

    forAll(Su0Bf, patchi)
    {
        forAll(Su0Bf[patchi], facei)
        {
            Su0Bf[patchi][facei] =
                Su0pTPhi
                (
                    p.boundaryField()[patchi][facei],
                    Tu.boundaryField()[patchi][facei],
                    Phi.boundaryField()[patchi][facei],
                    0
                );
        }
    }

    return tSu0;
}


Foam::tmp<Foam::volScalarField> Foam::laminarFlameSpeedModels::piecewisePhi::Su
(
    const volScalarField& p,
    const volScalarField& Tu,
    const volScalarField& Phi,
    const volScalarField& egr
) const
{
    tmp<volScalarField> tSu0
    (
        volScalarField::New
        (
            "Su0",
            p.mesh(),
            dimensionedScalar(dimensions::velocity, 0)
        )
    );

    volScalarField& Su0 = tSu0.ref();

    forAll(Su0, celli)
    {
        Su0[celli] = Su0pTPhi(p[celli], Tu[celli], Phi[celli], egr[celli]);
    }

    volScalarField::Boundary& Su0Bf = Su0.boundaryFieldRef();

    forAll(Su0Bf, patchi)
    {
        forAll(Su0Bf[patchi], facei)
        {
            Su0Bf[patchi][facei] =
                Su0pTPhi
                (
                    p.boundaryField()[patchi][facei],
                    Tu.boundaryField()[patchi][facei],
                    Phi.boundaryField()[patchi][facei],
                    egr.boundaryField()[patchi][facei]
                );
        }
    }

    return tSu0;
}


// ************************************************************************* //
