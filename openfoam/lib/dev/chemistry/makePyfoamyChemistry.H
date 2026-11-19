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
    Reactions for one ThermoPhysics of the default fit (logPolynomial
    transport, janaf thermo, sensible enthalpy): the reversible ones, which
    depend on the thermo; the chemistry models are made by the includer.

\*---------------------------------------------------------------------------*/

#include "typedefPyfoamyThermo.H"

#include "makeChemistryModel.H"
#include "Standard_chemistryModel.H"

#include "makeRateReaction.H"
#include "Arrhenius_reactionRate.H"
#include "LandauTeller_reactionRate.H"
#include "ThirdBody_reactionRate.H"
#include "pLog_reactionRate.H"
#include "Janev_reactionRate.H"
#include "powerSeries_reactionRate.H"
#include "FallOff_reactionRate.H"
#include "ChemicallyActivated_reactionRate.H"
#include "LangmuirHinshelwood_reactionRate.H"
#include "Lindemann_fallOffFunction.H"
#include "Troe_fallOffFunction.H"
#include "SRI_fallOffFunction.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace reactionRates
{
    typedef ThirdBody<Arrhenius> ThirdBodyArrhenius;

    typedef FallOff<Arrhenius, fallOffFunctions::Lindemann>
        FallOffArrheniusLindemann;
    typedef FallOff<Arrhenius, fallOffFunctions::Troe> FallOffArrheniusTroe;
    typedef FallOff<Arrhenius, fallOffFunctions::SRI> FallOffArrheniusSRI;

    typedef ChemicallyActivated<Arrhenius, fallOffFunctions::Lindemann>
        ChemicallyActivatedArrheniusLindemann;
    typedef ChemicallyActivated<Arrhenius, fallOffFunctions::Troe>
        ChemicallyActivatedArrheniusTroe;
    typedef ChemicallyActivated<Arrhenius, fallOffFunctions::SRI>
        ChemicallyActivatedArrheniusSRI;
}
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define makePyfoamyChemistry(TP)                                               \
                                                                               \
    makeReversibleReaction(dictionary, Arrhenius, TP);                         \
    makeReversibleReaction(dictionary, LandauTeller, TP);                      \
    makeReversibleReaction(dictionary, ThirdBodyArrhenius, TP);                \
    makeReversibleReaction(dictionary, pLog, TP);                              \
    makeReversibleReaction(dictionary, Janev, TP);                             \
    makeReversibleReaction(dictionary, powerSeries, TP);                       \
    makeReversibleReaction(dictionary, FallOffArrheniusLindemann, TP);         \
    makeReversibleReaction(dictionary, FallOffArrheniusTroe, TP);              \
    makeReversibleReaction(dictionary, FallOffArrheniusSRI, TP);               \
    makeReversibleReaction                                                     \
    (                                                                          \
        dictionary,                                                            \
        ChemicallyActivatedArrheniusLindemann,                                 \
        TP                                                                     \
    );                                                                         \
    makeReversibleReaction                                                     \
    (                                                                          \
        dictionary,                                                            \
        ChemicallyActivatedArrheniusTroe,                                      \
        TP                                                                     \
    );                                                                         \
    makeReversibleReaction                                                     \
    (                                                                          \
        dictionary,                                                            \
        ChemicallyActivatedArrheniusSRI,                                       \
        TP                                                                     \
    );                                                                         \
    makeReversibleReaction(dictionary, LangmuirHinshelwood, TP)


// ************************************************************************* //
