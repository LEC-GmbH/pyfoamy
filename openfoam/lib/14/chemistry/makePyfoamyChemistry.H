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
    Reduction methods and reactions for one ThermoPhysics of the default fit
    (logPolynomial transport, janaf thermo, sensible enthalpy); the chemistry
    models are made by the includer.

\*---------------------------------------------------------------------------*/

#include "typedefPyfoamyThermo.H"

#include "makeChemistryModel.H"
#include "Standard_chemistryModel.H"
#include "makeChemistryReductionMethod.H"
#include "noChemistryReduction.H"
#include "DAC.H"
#include "DRG.H"
#include "DRGEP.H"
#include "EFA.H"
#include "PFA.H"

#include "makeReaction.H"
#include "ArrheniusReactionRate.H"
#include "LandauTellerReactionRate.H"
#include "thirdBodyArrheniusReactionRate.H"
// An OpenFOAM-14 without the pLog backport: pyfoamy's copy
#if __has_include("pLogReactionRate.H")
    #include "pLogReactionRate.H"
#else
    #include "pyfoamyPLogReactionRate.H"
#endif
#include "JanevReactionRate.H"
#include "powerSeriesReactionRate.H"
#include "FallOffReactionRate.H"
#include "ChemicallyActivatedReactionRate.H"
#include "LindemannFallOffFunction.H"
#include "SRIFallOffFunction.H"
#include "TroeFallOffFunction.H"
#include "MichaelisMentenReactionRate.H"
#include "LangmuirHinshelwoodReactionRate.H"
#include "fluxLimitedLangmuirHinshelwoodReactionRate.H"
#include "surfaceArrheniusReactionRate.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define makePyfoamyChemistry(TP)                                                  \
                                                                               \
    defineChemistryReductionMethod(nullArg, TP);                               \
    makeChemistryReductionMethod(none, TP);                                    \
    makeChemistryReductionMethod(DAC, TP);                                     \
    makeChemistryReductionMethod(DRG, TP);                                     \
    makeChemistryReductionMethod(DRGEP, TP);                                   \
    makeChemistryReductionMethod(EFA, TP);                                     \
    makeChemistryReductionMethod(PFA, TP);                                     \
                                                                               \
    defineReaction(nullArg, TP);                                               \
                                                                               \
    makeIRNReactions(ArrheniusReactionRate, TP);                               \
    makeIRNReactions(LandauTellerReactionRate, TP);                            \
    makeIRNReactions(thirdBodyArrheniusReactionRate, TP);                      \
    makeIRNReactions(pLogReactionRate, TP);                                    \
                                                                               \
    makeIRReactions(JanevReactionRate, TP);                                    \
    makeIRReactions(powerSeriesReactionRate, TP);                              \
                                                                               \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        FallOffReactionRate,                                                   \
        ArrheniusReactionRate,                                                 \
        LindemannFallOffFunction,                                              \
        TP                                                                     \
    );                                                                         \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        FallOffReactionRate,                                                   \
        ArrheniusReactionRate,                                                 \
        TroeFallOffFunction,                                                   \
        TP                                                                     \
    );                                                                         \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        FallOffReactionRate,                                                   \
        ArrheniusReactionRate,                                                 \
        SRIFallOffFunction,                                                    \
        TP                                                                     \
    );                                                                         \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        ChemicallyActivatedReactionRate,                                       \
        ArrheniusReactionRate,                                                 \
        LindemannFallOffFunction,                                              \
        TP                                                                     \
    );                                                                         \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        ChemicallyActivatedReactionRate,                                       \
        ArrheniusReactionRate,                                                 \
        TroeFallOffFunction,                                                   \
        TP                                                                     \
    );                                                                         \
    makeIRTemplate2Reactions                                                   \
    (                                                                          \
        ChemicallyActivatedReactionRate,                                       \
        ArrheniusReactionRate,                                                 \
        SRIFallOffFunction,                                                    \
        TP                                                                     \
    );                                                                         \
                                                                               \
    makeIReactions(MichaelisMentenReactionRate, TP);                           \
    makeIRReactions(LangmuirHinshelwoodReactionRate, TP);                      \
    makeGeneralReaction                                                        \
    (                                                                          \
        IrreversibleReaction,                                                  \
        fluxLimitedLangmuirHinshelwoodReactionRate,                            \
        TP                                                                     \
    );                                                                         \
    makeGeneralReaction                                                        \
    (                                                                          \
        IrreversibleReaction,                                                  \
        surfaceArrheniusReactionRate,                                          \
        TP                                                                     \
    )


// ************************************************************************* //
