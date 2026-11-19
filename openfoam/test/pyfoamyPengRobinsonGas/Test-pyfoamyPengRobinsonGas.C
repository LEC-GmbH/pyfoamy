/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2026 LEC - Large Engines Competence Center
-------------------------------------------------------------------------------
License
    This file is part of pyfoamy's OpenFOAM extensions.

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

Application
    Test-pyfoamyPengRobinsonGas

Description
    Evaluates pyfoamyPengRobinsonGas for tests/test_peng_robinson.py, from
    the dictionary "input" in the working directory:
    \verbatim
        species   { <name> { specie {..} equationOfState {..}
                             thermodynamics {..} } ... }
        mixtures  { <name> { <specie> <mole fraction>; ... } ... }
        states    ((<p> <T>) ...);
        roundTrip { mixture <name>; p <p>; T (...); T0Offsets (...); }
    \endverbatim
    Mixtures are built as coefficientMulticomponentMixture does: the first
    specie times its mass fraction, the others added.

    Writes results.dat: per mixture and state Z, the departures h/(R T),
    Cp/R, Cv/R, (Cp - Cv)/R and alphav T, and the errors against central
    differences (dT = 1e-4 T) of alphav, Cp (from h), Cv (from e at
    constant rho) and psi p/rho; roundTrip.dat: T, T0 and the temperature
    THs returns from hs(p, T) starting at T0 (nan if it failed);
    operatorEq.dat: Z of the first two species' operator== and of the
    first specie at 1 bar, 1000 K.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "IFstream.H"
#include "OFstream.H"
#include "specie.H"
#include "pyfoamyPengRobinsonGas.H"
#include "janafThermo.H"
#include "sensibleEnthalpy.H"
#include "thermo.H"
#include "scalarList.H"

using namespace Foam;

typedef pyfoamyPengRobinsonGas<specie> Gas;
typedef species::thermo<janafThermo<Gas>, sensibleEnthalpy> Thermo;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList args(argc, argv, false, false, false);

    const dictionary input((IFstream("input")()));

    PtrList<Thermo> thermos;
    HashTable<label> index;

    const dictionary& speciesDict = input.subDict("species");
    forAllConstIter(dictionary, speciesDict, iter)
    {
        index.insert(iter().keyword(), thermos.size());
        thermos.append(new Thermo(iter().keyword(), iter().dict()));
    }

    const dictionary& mixturesDict = input.subDict("mixtures");
    PtrList<Thermo> mixtures;
    HashTable<label> mixtureIndex;
    forAllConstIter(dictionary, mixturesDict, iter)
    {
        mixtureIndex.insert(iter().keyword(), mixtures.size());

        const dictionary& X = iter().dict();

        scalar XW = 0;
        forAllConstIter(dictionary, X, j)
        {
            XW +=
                X.lookup<scalar>(j().keyword())
               *thermos[index[j().keyword()]].W();
        }

        autoPtr<Thermo> mix;
        forAllConstIter(dictionary, X, j)
        {
            const Thermo& t = thermos[index[j().keyword()]];
            const scalar Y = X.lookup<scalar>(j().keyword())*t.W()/XW;

            if (mix.valid())
            {
                mix() += Y*t;
            }
            else
            {
                mix.reset(new Thermo(iter().keyword(), Y*t));
            }
        }
        mixtures.append(mix.ptr());
    }

    const List<Pair<scalar>> states(input.lookup("states"));

    OFstream dat("results.dat");
    dat.precision(17);
    dat << "# name p T Z hDepRT CpDepR CvDepR CpMCvR alphavT"
        << " errAlphavT errCpR errCvR errPsi" << endl;

    forAll(mixtures, mixi)
    {
        const Gas& g = mixtures[mixi];

        for (const Pair<scalar>& s : states)
        {
            const scalar p = s.first(), T = s.second(), dT = 1e-4*T;
            const scalar R = g.R();
            const scalar rho0 = g.rho(p, T);

            // Pressure at T1 for the state's density, by Newton
            const auto pAt = [&g, rho0, p](const scalar T1)
            {
                scalar p1 = p;
                for (label k = 0; k < 50; k++)
                {
                    const scalar dp = 1e-6*p1;
                    const scalar f = g.rho(p1, T1) - rho0;
                    p1 -= f*2*dp/(g.rho(p1 + dp, T1) - g.rho(p1 - dp, T1));
                    if (mag(f) < 1e-14*rho0) break;
                }
                return p1;
            };

            const scalar alphavFD =
                -(g.rho(p, T + dT) - g.rho(p, T - dT))/(2*dT*rho0);
            const scalar CpFD = (g.h(p, T + dT) - g.h(p, T - dT))/(2*dT);
            const scalar CvFD =
                (g.e(pAt(T + dT), T + dT) - g.e(pAt(T - dT), T - dT))/(2*dT);

            dat << mixtures[mixi].name() << ' ' << p << ' ' << T << ' '
                << g.Z(p, T) << ' '
                << g.h(p, T)/(R*T) << ' '
                << g.Cp(p, T)/R << ' '
                << g.Cv(p, T)/R << ' '
                << g.CpMCv(p, T)/R << ' '
                << g.alphav(p, T)*T << ' '
                << mag(g.alphav(p, T) - alphavFD)*T << ' '
                << mag(g.Cp(p, T) - CpFD)/R << ' '
                << mag(g.Cv(p, T) - CvFD)/R << ' '
                << mag(g.psi(p, T)*p - rho0)/rho0 << endl;
        }
    }

    if (input.found("roundTrip"))
    {
        const dictionary& rt = input.subDict("roundTrip");
        const Thermo& t = mixtures[mixtureIndex[rt.lookup<word>("mixture")]];
        const scalar p = rt.lookup<scalar>("p");
        const scalarList Ts(rt.lookup("T"));
        const scalarList offsets(rt.lookup("T0Offsets"));

        FatalError.throwExceptions();

        OFstream rtDat("roundTrip.dat");
        rtDat.precision(17);
        rtDat << "# T T0 THs" << endl;
        for (const scalar T : Ts)
        {
            for (const scalar dT0 : offsets)
            {
                scalar Tback = std::numeric_limits<scalar>::quiet_NaN();
                try
                {
                    Tback = t.THs(t.hs(p, T), p, T + dT0);
                }
                catch (const error&)
                {}
                rtDat << T << ' ' << T + dT0 << ' ' << Tback << endl;
            }
        }
    }

    if (thermos.size() > 1)
    {
        const Gas& g0 = thermos[0];
        const Gas& g1 = thermos[1];

        OFstream eq("operatorEq.dat");
        eq.precision(17);
        eq << Gas(g0 == g1).Z(1e5, 1000) << ' ' << g0.Z(1e5, 1000) << endl;
    }

    Info<< "End" << endl;

    return 0;
}


// ************************************************************************* //
