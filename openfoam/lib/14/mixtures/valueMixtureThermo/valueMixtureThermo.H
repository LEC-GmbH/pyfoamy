/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2025 OpenFOAM Foundation
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
    Foam::valueMixtureThermoType
    Foam::valueMixtureTransportType

Description
    Shared per-species-evaluated mixing framework for a FIXED, small
    number of named species (nSpecies known at compile time - e.g. the
    3 pseudo-species fuel/oxidant/products shared by
    valueBInhomogeneousMixture and valueUInhomogeneousEGRMixture),
    mirroring Foam::valueMulticomponentMixture's own
    massWeighted/harmonicMassWeighted/moleWeighted framework (which is
    built for a runtime-sized PtrList<ThermoType> instead - overkill,
    and a real source of code duplication, for a mixture that only
    ever combines 2-3 named species).

    Each mixture class using this only needs to:
      - construct the thermoMixtureType/transportMixtureType from its
        own fixed ThermoType members (fuel_, oxidant_, products_, ...)
      - compute the species WEIGHTS for a given composition (the one
        genuinely per-mixture bit of logic - e.g.
        BInhomogeneousMixture's ft-based fu/ox/pr split vs
        UInhomogeneousEGRMixture's fu/egr-based one) and hand them to
        setWeights()
    All the actual Cp/H/mu/kappa/etc mixing arithmetic - previously
    hand-duplicated per mixture class - lives here once.

See also
    Foam::valueMulticomponentMixture
    Foam::valueBInhomogeneousMixture
    Foam::valueUInhomogeneousEGRMixture

\*---------------------------------------------------------------------------*/

#ifndef valueMixtureThermo_H
#define valueMixtureThermo_H

#include "FixedList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                   Class valueMixtureThermoType Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType, int nSpecies>
class valueMixtureThermoType
{
    // Private Data

        //- The nSpecies fixed species thermo models (not owned)
        FixedList<const ThermoType*, nSpecies> species_;

        //- Current mass-fraction weights, one per species
        mutable FixedList<scalar, nSpecies> Y_;


    // Private Member Functions

        //- Calculate a mass-fraction-weighted property
        template<class Method, class ... Args>
        scalar massWeighted(Method psiMethod, const Args& ... args) const
        {
            scalar psi = 0;
            forAll(species_, i)
            {
                psi += Y_[i]*(species_[i]->*psiMethod)(args ...);
            }
            return psi;
        }

        //- Calculate a harmonic mass-fraction-weighted property
        template<class Method, class ... Args>
        scalar harmonicMassWeighted(Method psiMethod, const Args& ... args)
        const
        {
            scalar rPsi = 0;
            forAll(species_, i)
            {
                rPsi += Y_[i]/(species_[i]->*psiMethod)(args ...);
            }
            return 1/rPsi;
        }

        //- Limit the given temperature
        scalar limit(const scalar T) const
        {
            return T;
        }


public:

    // Constructors

        valueMixtureThermoType(const FixedList<const ThermoType*, nSpecies>& species)
        :
            species_(species)
        {
            forAll(Y_, i)
            {
                Y_[i] = 0;
            }
        }


    // Member Functions

        //- Set the current mass-fraction weights (one per species,
        //  in the same order as the species this was constructed
        //  with) - the one thing a caller updates per evaluation
        void setWeights(const FixedList<scalar, nSpecies>& Y) const
        {
            Y_ = Y;
        }

        //- The current mass-fraction weight for specie i
        scalar Y(const label i) const
        {
            return Y_[i];
        }

        //- The specie thermo for specie i
        const ThermoType& specieThermo(const label i) const
        {
            return *species_[i];
        }


        // Fundamental properties

            //- Molecular weight [kg/kmol]
            scalar W() const
            {
                return harmonicMassWeighted(&ThermoType::W);
            }

            //- Return density [kg/m^3]
            scalar rho(scalar p, scalar T) const
            {
                return harmonicMassWeighted(&ThermoType::rho, p, T);
            }

            //- Return compressibility [s^2/m^2]
            scalar psi(scalar p, scalar T) const
            {
                scalar oneByRho = 0;
                scalar psiByRho2 = 0;

                forAll(species_, i)
                {
                    const scalar rhoi = species_[i]->rho(p, T);
                    const scalar psii = species_[i]->psi(p, T);

                    oneByRho += Y_[i]/rhoi;

                    if (psii > 0)
                    {
                        psiByRho2 += Y_[i]*psii/sqr(rhoi);
                    }
                }

                return psiByRho2/sqr(oneByRho);
            }

            //- Return volumetric coefficient of thermal expansion [1/T]
            scalar alphav(const scalar p, const scalar T) const
            {
                return harmonicMassWeighted(&ThermoType::alphav, p, T);
            }

            //- Heat capacity at constant pressure [J/kg/K]
            scalar Cp(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::Cp, p, T);
            }

            //- Heat capacity at constant volume [J/kg/K]
            scalar Cv(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::Cv, p, T);
            }

            //- Sensible enthalpy [J/kg]
            scalar hs(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::hs, p, T);
            }

            //- Sensible internal energy [J/kg]
            scalar es(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::es, p, T);
            }

            //- Absolute enthalpy [J/kg]
            scalar ha(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::ha, p, T);
            }

            //- Enthalpy of formation [J/kg]
            scalar hf() const
            {
                return massWeighted(&ThermoType::hf);
            }


        // Mass specific derived properties

            //- Heat capacity at constant pressure/volume [J/kg/K]
            scalar Cpv(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::Cpv, p, T);
            }

            //- Gamma = Cp/Cv []
            scalar gamma(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::gamma, p, T);
            }

            //- Enthalpy/Internal energy [J/kg]
            scalar he(const scalar p, const scalar T) const
            {
                return massWeighted(&ThermoType::he, p, T);
            }


        // Energy->temperature inversion functions

            //- Temperature from enthalpy or internal energy
            //  given an initial temperature T0
            scalar The(const scalar he, const scalar p, const scalar T0)
            const
            {
                return ThermoType::T
                (
                    *this,
                    he,
                    p,
                    T0,
                    &valueMixtureThermoType::he,
                    &valueMixtureThermoType::Cpv,
                    &valueMixtureThermoType::limit
                );
            }

            //- Temperature from sensible internal energy
            //  given an initial temperature T0
            scalar Tes(const scalar he, const scalar p, const scalar T0)
            const
            {
                return ThermoType::T
                (
                    *this,
                    he,
                    p,
                    T0,
                    &valueMixtureThermoType::es,
                    &valueMixtureThermoType::Cv,
                    &valueMixtureThermoType::limit
                );
            }
};


/*---------------------------------------------------------------------------*\
                 Class valueMixtureTransportType Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType, int nSpecies>
class valueMixtureTransportType
{
    // Private Data

        //- The nSpecies fixed species thermo models (not owned)
        FixedList<const ThermoType*, nSpecies> species_;

        //- Current mole-fraction weights, one per species
        mutable FixedList<scalar, nSpecies> X_;


public:

    // Constructors

        valueMixtureTransportType
        (
            const FixedList<const ThermoType*, nSpecies>& species
        )
        :
            species_(species)
        {
            forAll(X_, i)
            {
                X_[i] = 0;
            }
        }


    // Member Functions

        //- Set the current mole-fraction weights (one per species) -
        //  see setMassWeights() to derive these from mass fractions
        //  instead, the normal case
        void setWeights(const FixedList<scalar, nSpecies>& X) const
        {
            X_ = X;
        }

        //- Set the current weights from mass fractions Y (one per
        //  species) - normalizes to mole fractions internally,
        //  mirroring valueMulticomponentMixture::transportMixture()
        void setMassWeights(const FixedList<scalar, nSpecies>& Y) const
        {
            scalar sumX = 0;
            forAll(species_, i)
            {
                X_[i] = Y[i]/species_[i]->W();
                sumX += X_[i];
            }
            forAll(X_, i)
            {
                X_[i] /= sumX;
            }
        }

        //- Dynamic viscosity [kg/m/s]
        scalar mu(const scalar p, const scalar T) const
        {
            scalar mu = 0;
            forAll(species_, i)
            {
                mu += X_[i]*species_[i]->mu(p, T);
            }
            return mu;
        }

        //- Thermal conductivity [W/m/K]
        scalar kappa(const scalar p, const scalar T) const
        {
            scalar kappa = 0;
            forAll(species_, i)
            {
                kappa += X_[i]*species_[i]->kappa(p, T);
            }
            return kappa;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
