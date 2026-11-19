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

#include "pyfoamyIdealSpecieThermo.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BaseThermo>
template<class Method>
Foam::tmp<Foam::scalarField>
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::property
(
    Method method,
    const label speciei,
    const scalarField& p,
    const scalarField& T
) const
{
    const idealThermoType& thermo = idealThermos_[speciei];

    tmp<scalarField> tPsi(new scalarField(T.size()));
    scalarField& psi = tPsi.ref();

    forAll(psi, i)
    {
        psi[i] = (thermo.*method)(p[i], T[i]);
    }

    return tPsi;
}


template<class BaseThermo>
template<class Method>
Foam::tmp<Foam::volScalarField::Internal>
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::property
(
    const word& name,
    const dimensionSet& dims,
    Method method,
    const label speciei,
    const volScalarField::Internal& p,
    const volScalarField::Internal& T
) const
{
    tmp<volScalarField::Internal> tPsi
    (
        volScalarField::Internal::New
        (
            IOobject::groupName(name, this->T_.group()),
            this->T_.mesh(),
            dims
        )
    );

    tPsi.ref().primitiveFieldRef() = property(method, speciei, p, T);

    return tPsi;
}


template<class BaseThermo>
template<class Method>
Foam::tmp<Foam::volScalarField>
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::property
(
    const word& name,
    const dimensionSet& dims,
    Method method,
    const label speciei,
    const volScalarField& p,
    const volScalarField& T
) const
{
    tmp<volScalarField> tPsi
    (
        volScalarField::New
        (
            IOobject::groupName(name, this->T_.group()),
            this->T_.mesh(),
            dims
        )
    );
    volScalarField& psi = tPsi.ref();

    psi.primitiveFieldRef() =
        property(method, speciei, p.primitiveField(), T.primitiveField());

    volScalarField::BoundaryField& psiBf = psi.boundaryFieldRef();

    forAll(psiBf, patchi)
    {
        psiBf[patchi] =
            property
            (
                method,
                speciei,
                p.boundaryField()[patchi],
                T.boundaryField()[patchi]
            );
    }

    return tPsi;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BaseThermo>
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::pyfoamyIdealSpecieThermo
(
    const fvMesh& mesh,
    const word& phaseName
)
:
    BaseThermo(mesh, phaseName),
    idealThermos_(this->species().size())
{
    const speciesTable& species = this->species();

    forAll(species, i)
    {
        idealThermos_.set
        (
            i,
            new idealThermoType
            (
                species[i],
                this->properties().subDict(species[i])
            )
        );
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BaseThermo>
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::~pyfoamyIdealSpecieThermo()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BaseThermo>
Foam::scalar Foam::pyfoamyIdealSpecieThermo<BaseThermo>::rhoi
(
    const label speciei,
    const scalar p,
    const scalar T
) const
{
    return idealThermos_[speciei].rho(p, T);
}


template<class BaseThermo>
Foam::tmp<Foam::volScalarField> Foam::pyfoamyIdealSpecieThermo<BaseThermo>::rhoi
(
    const label speciei,
    const volScalarField& p,
    const volScalarField& T
) const
{
    return
        property("rho", dimDensity, &idealThermoType::rho, speciei, p, T);
}


template<class BaseThermo>
Foam::scalar Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Cpi
(
    const label speciei,
    const scalar p,
    const scalar T
) const
{
    return idealThermos_[speciei].Cp(p, T);
}


template<class BaseThermo>
Foam::tmp<Foam::volScalarField> Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Cpi
(
    const label speciei,
    const volScalarField& p,
    const volScalarField& T
) const
{
    return
        property
        (
            "Cp",
            dimEnergy/dimMass/dimTemperature,
            &idealThermoType::Cp,
            speciei,
            p,
            T
        );
}


#define pyfoamyIdealSpecieThermoEnthalpy(Name)                                     \
                                                                               \
template<class BaseThermo>                                                     \
Foam::scalar Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Name##i                   \
(                                                                              \
    const label speciei,                                                       \
    const scalar p,                                                            \
    const scalar T                                                             \
) const                                                                        \
{                                                                              \
    return idealThermos_[speciei].Name(p, T);                                  \
}                                                                              \
                                                                               \
                                                                               \
template<class BaseThermo>                                                     \
Foam::tmp<Foam::scalarField>                                                   \
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Name##i                                \
(                                                                              \
    const label speciei,                                                       \
    const scalarField& p,                                                      \
    const scalarField& T                                                       \
) const                                                                        \
{                                                                              \
    return property(&idealThermoType::Name, speciei, p, T);                    \
}                                                                              \
                                                                               \
                                                                               \
template<class BaseThermo>                                                     \
Foam::tmp<Foam::volScalarField::Internal>                                      \
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Name##i                                \
(                                                                              \
    const label speciei,                                                       \
    const volScalarField::Internal& p,                                         \
    const volScalarField::Internal& T                                          \
) const                                                                        \
{                                                                              \
    return                                                                     \
        property                                                               \
        (                                                                      \
            #Name, dimEnergy/dimMass, &idealThermoType::Name, speciei, p, T    \
        );                                                                     \
}                                                                              \
                                                                               \
                                                                               \
template<class BaseThermo>                                                     \
Foam::tmp<Foam::volScalarField>                                                \
Foam::pyfoamyIdealSpecieThermo<BaseThermo>::Name##i                                \
(                                                                              \
    const label speciei,                                                       \
    const volScalarField& p,                                                   \
    const volScalarField& T                                                    \
) const                                                                        \
{                                                                              \
    return                                                                     \
        property                                                               \
        (                                                                      \
            #Name, dimEnergy/dimMass, &idealThermoType::Name, speciei, p, T    \
        );                                                                     \
}

pyfoamyIdealSpecieThermoEnthalpy(he)
pyfoamyIdealSpecieThermoEnthalpy(hs)
pyfoamyIdealSpecieThermoEnthalpy(ha)

#undef pyfoamyIdealSpecieThermoEnthalpy


// ************************************************************************* //
