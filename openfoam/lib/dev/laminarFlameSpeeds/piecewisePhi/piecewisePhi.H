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
    Foam::laminarFlameSpeedModels::piecewisePhi

Description
    Laminar flame speed obtained by tabulating pTu (p, Tu) -> Su0 and the EGR
    reduction factor f (p, Tu) at a set of specified equivalence ratios Phi,
    then interpolating linearly in Phi between the two bracketing table
    entries (clamped to the nearest entry outside the tabulated range).

    Each Phi entry's Su0/f dependence on p and Tu is supplied as an arbitrary
    Function2 (e.g. a uniformTable), so no particular functional form (such
    as Gulder's) is assumed.

    with optional support for EGR (exhaust gas recirculation) if the \c egr
    mass-fraction field is available.

    Note: the EGR \c f coefficient multiplies the \c egr mass-fraction, not a
    mole-fraction.

SourceFiles
    piecewisePhi.C

\*---------------------------------------------------------------------------*/

#ifndef piecewisePhi_H
#define piecewisePhi_H

#include "laminarFlameSpeed.H"
#include "Function2.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarFlameSpeedModels
{

/*---------------------------------------------------------------------------*\
                           Class piecewisePhi Declaration
\*---------------------------------------------------------------------------*/

class piecewisePhi
:
    public laminarFlameSpeed
{
    // Private Data

        // Order
        labelList order_;

        // Specified equivalence ratios
        List<scalar> Phis_;

        // Laminar flame speed correlations
        List<autoPtr<Function2<scalar>>> pTuFunctions_;

        // EGR reduction factors
        List<autoPtr<Function2<scalar>>> fFunctions_;


    // Private Member Functions

        // Find lower interpolation index
        inline label phiIndex(const scalar Phi) const;

        inline scalar SuRef
        (
            const scalar p,
            const scalar Tu,
            const scalar Yegr,
            const label idx
        ) const;

        inline scalar Su0pTPhi
        (
            const scalar p,
            const scalar Tu,
            const scalar Phi,
            const scalar Yegr
        ) const;


public:

    //- Runtime type information
    TypeName("piecewisePhi");


    // Constructors

        //- Construct from dictionary and uRhoMulticomponentThermo
        piecewisePhi
        (
            const dictionary& dict,
            const dictionary& coeffDict,
            const uRhoMulticomponentThermo&
        );

        //- Disallow default bitwise copy construction
        piecewisePhi(const piecewisePhi&) = delete;


    //- Destructor
    virtual ~piecewisePhi();


    // Member Functions

        //- Return the laminar flame speed [m/s]
        virtual tmp<volScalarField> Su
        (
            const volScalarField& p,
            const volScalarField& Tu,
            const volScalarField& Phi
        ) const;

        //- Return the laminar flame speed [m/s]
        //  including EGR effects
        virtual tmp<volScalarField> Su
        (
            const volScalarField& p,
            const volScalarField& Tu,
            const volScalarField& Phi,
            const volScalarField& egr
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const piecewisePhi&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End laminarFlameSpeedModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
