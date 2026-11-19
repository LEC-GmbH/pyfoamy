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
    Verifies pyfoamyIdealSpecieThermo on a one-cell N2/NC12H26 case (run in
    case/ after blockMesh): every per-specie rhoi, Cpi, hei, hsi, hai overload
    equals the perfect gas, the pure PR specie differs from it, and the
    mixture keeps its real-gas psi. Exits 1 if a check fails.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "fluidMulticomponentThermo.H"
#include "logPolynomialTransport.H"
#include "sensibleEnthalpy.H"
#include "janafThermo.H"
#include "pyfoamyPengRobinsonGas.H"
#include "perfectGasThermo.H"

using namespace Foam;

typedef
    logPolynomialTransport
    <
        species::thermo<janafThermo<pyfoamyPengRobinsonGas<specie>>, sensibleEnthalpy>,
        8
    > realThermo;

typedef perfectGasThermo<realThermo>::type idealThermo;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    autoPtr<fluidMulticomponentThermo> tThermo
    (
        fluidMulticomponentThermo::New(mesh)
    );
    const fluidMulticomponentThermo& thermo = tThermo();

    const volScalarField& p = thermo.p();
    const volScalarField& T = thermo.T();
    const scalar p0 = p[0], T0 = T[0];

    bool ok = true;
    auto check = [&ok](const word& what, const scalar err, const bool pass)
    {
        Info<< "    " << what << ": " << err << (pass ? "" : "  FAILED")
            << endl;
        ok = ok && pass;
    };

    Info<< "thermo " << thermo.type() << nl
        << "p " << p0 << " Pa, T " << T0 << " K" << endl;

    forAll(thermo.species(), i)
    {
        const word& name = thermo.species()[i];
        const dictionary& dict = thermo.properties().subDict(name);
        const idealThermo ideal(name, dict);
        const realThermo real(name, dict);

        Info<< name << ": hs ideal " << ideal.hs(p0, T0)
            << ", pure PR " << real.hs(p0, T0) << " J/kg" << endl;

        scalar err = 0;
        auto rel = [](const scalar a, const scalar b)
        {
            return mag(a - b)/max(mag(b), small);
        };

        err = max(err, rel(thermo.rhoi(i, p0, T0), ideal.rho(p0, T0)));
        err = max(err, rel(thermo.rhoi(i, p, T)()[0], ideal.rho(p0, T0)));
        err = max(err, rel(thermo.Cpi(i, p0, T0), ideal.Cp(p0, T0)));
        err = max(err, rel(thermo.Cpi(i, p, T)()[0], ideal.Cp(p0, T0)));

        #define checkEnthalpy(Name)                                            \
            err = max(err, rel(thermo.Name##i(i, p0, T0), ideal.Name(p0, T0))); \
            err = max                                                          \
            (                                                                  \
                err,                                                           \
                rel                                                            \
                (                                                              \
                    thermo.Name##i(i, p.primitiveField(), T.primitiveField())()[0], \
                    ideal.Name(p0, T0)                                         \
                )                                                              \
            );                                                                 \
            err = max                                                          \
            (                                                                  \
                err,                                                           \
                rel                                                            \
                (                                                              \
                    thermo.Name##i(i, p.internalField(), T.internalField())()[0], \
                    ideal.Name(p0, T0)                                         \
                )                                                              \
            );                                                                 \
            err = max(err, rel(thermo.Name##i(i, p, T)()[0], ideal.Name(p0, T0)));

        checkEnthalpy(he)
        checkEnthalpy(hs)
        checkEnthalpy(ha)

        #undef checkEnthalpy

        check("per-specie vs perfect gas, rel", err, err < 1e-12);
    }

    // The test means something only if PR is far off the ideal for the fuel
    {
        const label i = thermo.species()["NC12H26"];
        const dictionary& dict = thermo.properties().subDict("NC12H26");
        const scalar dep =
            realThermo("NC12H26", dict).hs(p0, T0)
          - idealThermo("NC12H26", dict).hs(p0, T0);
        check("NC12H26 pure PR departure [J/kg]", dep, dep < -1e5);
        Info<< "    (hsi " << thermo.hsi(i, p0, T0) << ")" << endl;
    }

    // The mixture stays real gas: psi differs from the perfect-gas 1/(R T)
    {
        const scalar psiIdeal = 1/(constant::thermodynamic::RR/thermo.W()()[0]*T0);
        const scalar dev = thermo.psi()[0]/psiIdeal - 1;
        check("mixture psi/psi_ideal - 1", dev, mag(dev) > 1e-3);
    }

    Info<< (ok ? "PASSED" : "FAILED") << endl;

    return ok ? 0 : 1;
}


// ************************************************************************* //
