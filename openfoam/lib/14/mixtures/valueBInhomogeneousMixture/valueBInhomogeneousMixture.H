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

Class
    Foam::valueBInhomogeneousMixture

Description
    Burnt gas combustion mixture for inhomogeneous lean, stoichiometric
    or rich conditions - same fu/ox/pr split as bInhomogeneousMixture
    (via the shared, unmodified base class), but evaluating each of
    fuel/oxidant/products' own Cp/H/etc at its OWN native Tcommon and
    mass-weighted-SUMMING the resulting values (via
    Foam::valueMixtureThermoType/valueMixtureTransportType), rather
    than blending the three species' raw JANAF coefficients into one
    combined thermoType (as BInhomogeneousMixture's operator+=-based
    mixture_ does).

    Motivation: BInhomogeneousMixture's coefficient blend keeps
    whichever specie is assigned first (fuel_) as the combined
    mixture's Tcommon regardless of its own mass fraction, and only
    checks the other species' Tcommon against it in a debug build -
    mismatched Tcommon between fuel/oxidant/products produces a real
    Cp/H discontinuity at that inherited value, which can break
    OpenFOAM's T(h) Newton inversion right at that temperature
    (confirmed against real production data).

See also
    Foam::bInhomogeneousMixture
    Foam::BInhomogeneousMixture
    Foam::valueMixtureThermoType
    Foam::valueUInhomogeneousEGRMixture

SourceFiles
    valueBInhomogeneousMixture.C

\*---------------------------------------------------------------------------*/

#ifndef valueBInhomogeneousMixture_H
#define valueBInhomogeneousMixture_H

#include "bInhomogeneousMixture.H"
#include "valueMixtureThermo.H"
#include "FieldListSlice.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class valueBInhomogeneousMixture Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class valueBInhomogeneousMixture
:
    public bInhomogeneousMixture
{
public:

    // Public Typedefs

        //- The type of thermodynamics this mixture is instantiated for
        typedef ThermoType thermoType;

        //- Mixing type for thermodynamic properties
        typedef valueMixtureThermoType<ThermoType, 3> thermoMixtureType;

        //- Mixing type for transport properties
        typedef valueMixtureTransportType<ThermoType, 3> transportMixtureType;


private:

    // Private Data

        //- Fuel thermodynamic model
        thermoType fuel_;

        //- Oxidant thermodynamic model
        thermoType oxidant_;

        //- Product thermodynamic model
        thermoType products_;

        //- Mutable storage for the cell/face mixture thermo data
        mutable thermoMixtureType thermoMixture_;

        //- Mutable storage for the cell/face mixture transport data
        mutable transportMixtureType transportMixture_;


public:

    // Constructors

        //- Construct from a dictionary
        valueBInhomogeneousMixture(const dictionary&);

        //- Disallow default bitwise copy construction
        valueBInhomogeneousMixture
        (
            const valueBInhomogeneousMixture<ThermoType>&
        ) = delete;


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "valueBInhomogeneous<" + ThermoType::typeName() + '>';
        }

        inline const ThermoType& specieThermo(const label speciei) const
        {
            FatalErrorInFunction
                << "Cannot return specieThermo for specie " << speciei
                << exit(FatalError);
            return products_;
        }

        //- Return the mixture for thermodynamic properties
        const thermoMixtureType& thermoMixture
        (
            const scalarFieldListSlice&
        ) const;

        //- Return the mixture for transport properties
        const transportMixtureType& transportMixture
        (
            const scalarFieldListSlice&
        ) const;

        //- Return the mixture for transport properties
        const transportMixtureType& transportMixture
        (
            const scalarFieldListSlice&,
            const thermoMixtureType&
        ) const;

        //- Read dictionary
        void read(const dictionary&);


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const valueBInhomogeneousMixture<ThermoType>&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "valueBInhomogeneousMixture.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
