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
    Thermo packages for one thermo-physics type of the default
    fit (typedefPyfoamyThermo): pure (fluid), coefficient/value multicomponent
    (multicomponentFluid), unburnt/burnt (XiFluid). DefineCoefficient defines
    the coefficient multicomponent one: defineThermo, or
    definePyfoamyIdealSpecieThermo for a real-gas equation of state.

\*---------------------------------------------------------------------------*/

#include "typedefPyfoamyThermo.H"

#include "psiThermo.H"
#include "pureMixture.H"
#include "psiMulticomponentThermo.H"
#include "coefficientMulticomponentMixture.H"
#include "valueMulticomponentMixture.H"
#include "uRhoMulticomponentThermo.H"
#include "UInhomogeneousEGRMixture.H"
#include "bRhoMulticomponentThermo.H"
#include "BInhomogeneousMixture.H"

#include "makeThermo.H"
#include "pyfoamyIdealSpecieThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define definePyfoamyIdealSpecieThermo(BaseThermo, Mixture, ThermoPhysics)         \
                                                                               \
    typedef                                                                    \
        NamedThermo                                                            \
        <                                                                      \
            pyfoamyIdealSpecieThermo                                               \
            <                                                                  \
                BaseThermo::DerivedThermoType<Mixture<ThermoPhysics>>          \
            >                                                                  \
        >                                                                      \
        BaseThermo##Mixture##ThermoPhysics;                                    \
                                                                               \
    defineTemplateTypeNameAndDebugWithName                                     \
    (                                                                          \
        BaseThermo##Mixture##ThermoPhysics,                                    \
        (                                                                      \
            BaseThermo::derivedThermoName() + "<"                              \
          + Mixture<ThermoPhysics>::typeName() + ">"                           \
        ).c_str(),                                                             \
        0                                                                      \
    );

#define makePyfoamyThermo(                                                        \
    Define, SelectThermo, BaseThermo, Mixture, ThermoPhysics)                  \
    Define(BaseThermo, Mixture, ThermoPhysics);                                \
    addThermo(SelectThermo, BaseThermo, Mixture, ThermoPhysics);               \
    addThermo(BaseThermo, BaseThermo, Mixture, ThermoPhysics)

#define makePyfoamyThermos(ThermoPhysics, DefineCoefficient)                      \
    makePyfoamyThermo                                                             \
    (                                                                          \
        defineThermo,                                                          \
        fluidThermo,                                                           \
        psiThermo,                                                             \
        pureMixture,                                                           \
        ThermoPhysics                                                          \
    );                                                                         \
    makePyfoamyThermo                                                             \
    (                                                                          \
        DefineCoefficient,                                                     \
        fluidMulticomponentThermo,                                             \
        psiMulticomponentThermo,                                               \
        coefficientMulticomponentMixture,                                      \
        ThermoPhysics                                                          \
    );                                                                         \
    makePyfoamyThermo                                                             \
    (                                                                          \
        defineThermo,                                                          \
        fluidMulticomponentThermo,                                             \
        psiMulticomponentThermo,                                               \
        valueMulticomponentMixture,                                            \
        ThermoPhysics                                                          \
    );                                                                         \
    makePyfoamyThermo                                                             \
    (                                                                          \
        defineThermo,                                                          \
        rhoFluidThermo,                                                        \
        uRhoMulticomponentThermo,                                              \
        UInhomogeneousEGRMixture,                                              \
        ThermoPhysics                                                          \
    );                                                                         \
    makePyfoamyThermo                                                             \
    (                                                                          \
        defineThermo,                                                          \
        rhoFluidThermo,                                                        \
        bRhoMulticomponentThermo,                                              \
        BInhomogeneousMixture,                                                 \
        ThermoPhysics                                                          \
    )


// ************************************************************************* //
