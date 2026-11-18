/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020-2026 OpenFOAM Foundation
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

#include "nonUniformTable2.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

const Foam::NamedEnum<Foam::Function2s::nonUniformTable2::boundsHandling, 2>
Foam::Function2s::nonUniformTable2::boundsHandlingNames
{
    "error",
    "clamp"
};

namespace Foam
{
namespace Function2s
{
    // addToRunTimeSelectionTable can't token-paste a templated base
    // name (Function2<scalar>) directly - a local alias sidesteps
    // that, the same way the core addFunction2(SS, Type) macro does
    // (typedef Function2<Type> Type##Function2;) for every other
    // real Function2 type.
    typedef Function2<scalar> scalarFunction2;

    defineTypeNameAndDebug(nonUniformTable2, 0);

    addToRunTimeSelectionTable
    (
        scalarFunction2,
        nonUniformTable2,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::Function2s::nonUniformTable2::nonUniformTable2
(
    const word& name,
    const unitSets& units,
    const dictionary& dict
)
:
    FieldFunction2<scalar, nonUniformTable2>(name),
    x_(dict.lookup("x")),
    y_(dict.lookup("y")),
    values_(dict.lookup("values")),
    outOfBounds_
    (
        boundsHandlingNames[dict.lookupOrDefault<word>("outOfBounds", "error")]
    )
{
    assertNoConvertUnits(typeName, units, dict);

    if (x_.size() < 2 || y_.size() < 2)
    {
        FatalErrorInFunction
            << "Table " << this->name_ << nl
            << "    has fewer than 2 x or y breakpoints."
            << exit(FatalError);
    }

    if (values_.size() != x_.size())
    {
        FatalErrorInFunction
            << "Table " << this->name_ << nl
            << "    values has " << values_.size() << " row(s), "
               "expected " << x_.size() << " (matching x)."
            << exit(FatalError);
    }

    forAll(values_, i)
    {
        if (values_[i].size() != y_.size())
        {
            FatalErrorInFunction
                << "Table " << this->name_ << nl
                << "    values row " << i << " has "
                << values_[i].size() << " value(s), expected "
                << y_.size() << " (matching y)."
                << exit(FatalError);
        }
    }

    for (label i = 1; i < x_.size(); i++)
    {
        if (x_[i] <= x_[i - 1])
        {
            FatalErrorInFunction
                << "Table " << this->name_ << nl
                << "    x breakpoints are not strictly increasing."
                << exit(FatalError);
        }
    }

    for (label i = 1; i < y_.size(); i++)
    {
        if (y_[i] <= y_[i - 1])
        {
            FatalErrorInFunction
                << "Table " << this->name_ << nl
                << "    y breakpoints are not strictly increasing."
                << exit(FatalError);
        }
    }
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

inline void Foam::Function2s::nonUniformTable2::locate
(
    const List<scalar>& breakpoints,
    scalar v,
    label& i,
    scalar& lambda
) const
{
    if (v <= breakpoints.first())
    {
        if (v < breakpoints.first() && outOfBounds_ == boundsHandling::error)
        {
            FatalErrorInFunction
                << v << " out of range " << breakpoints.first()
                << " to " << breakpoints.last() << nl
                << "    of table " << this->name_
                << exit(FatalError);
        }
        i = 0;
        lambda = 0;
        return;
    }

    if (v >= breakpoints.last())
    {
        if (v > breakpoints.last() && outOfBounds_ == boundsHandling::error)
        {
            FatalErrorInFunction
                << v << " out of range " << breakpoints.first()
                << " to " << breakpoints.last() << nl
                << "    of table " << this->name_
                << exit(FatalError);
        }
        i = breakpoints.size() - 2;
        lambda = 1;
        return;
    }

    for (label k = 0; k < breakpoints.size() - 1; k++)
    {
        if (v >= breakpoints[k] && v <= breakpoints[k + 1])
        {
            i = k;
            lambda = (v - breakpoints[k])/(breakpoints[k + 1] - breakpoints[k]);
            return;
        }
    }

    // Should never happen - breakpoints are checked strictly increasing
    // at construction and v is already known to lie within their range.
    i = breakpoints.size() - 2;
    lambda = 1;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::scalar Foam::Function2s::nonUniformTable2::value
(
    scalar x,
    scalar y
) const
{
    label ix, iy;
    scalar lambdax, lambday;
    locate(x_, x, ix, lambdax);
    locate(y_, y, iy, lambday);

    const scalar fxi =
        values_[ix][iy]
      + lambdax*(values_[ix + 1][iy] - values_[ix][iy]);
    const scalar fxix1 =
        values_[ix][iy + 1]
      + lambdax*(values_[ix + 1][iy + 1] - values_[ix][iy + 1]);

    return fxi + lambday*(fxix1 - fxi);
}


void Foam::Function2s::nonUniformTable2::write
(
    Ostream& os,
    const unitSets& units
) const
{
    writeEntry(os, "x", x_);
    writeEntry(os, "y", y_);
    // Not writeEntry(os, "values", values_) - its no-keyword overload
    // resolves to writeListEntry() for any UList<T>, which writes a
    // "List<...>" compound tag needing pTraits<T>::typeName - List's
    // own element type here is List<scalar>, which has no such
    // pTraits specialisation (it isn't a primitive type), so that
    // path doesn't even compile. Streaming values_ directly uses
    // UList<T>::operator<< instead, which never needs pTraits<T> -
    // it only ever needs it (via contiguous<T>()) for the innermost
    // scalar rows, which are fine.
    os.writeKeyword("values") << values_ << token::END_STATEMENT << endl;
    writeEntry(os, "outOfBounds", boundsHandlingNames[outOfBounds_]);
}


// ************************************************************************* //
