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

Class
    Foam::Function2s::nonUniformTable2

Description
    Scalar function of two scalar arguments, tabulated on explicitly
    given, non-uniformly spaced x/y breakpoints, linearly interpolated
    between them - unlike the standard uniformTable, whose grid must be
    evenly spaced in both arguments.

    An (x, y) outside the tabulated range is handled per outOfBounds
    ("error", the default - a FatalError, matching uniformTable's own
    behaviour; or "clamp" - x/y are clamped to the nearest tabulated
    value before interpolating).

Usage
    \verbatim
    rho
    {
        type        nonUniformTable2;

        x           (1e5 2e5 3e5);
        y           (280 300 350);
        outOfBounds clamp;

        values
        (
            (991 992 993)
            (994 995 996)
            (997 998 999)
        );
    }
    \endverbatim

SourceFiles
    nonUniformTable2.C

\*---------------------------------------------------------------------------*/

#ifndef nonUniformTable2_H
#define nonUniformTable2_H

#include "Function2.H"
#include "NamedEnum.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace Function2s
{

/*---------------------------------------------------------------------------*\
                     Class nonUniformTable2 Declaration
\*---------------------------------------------------------------------------*/

class nonUniformTable2
:
    public FieldFunction2<scalar, nonUniformTable2>
{
public:

    // Public Data Types

        //- Enumeration for handling out-of-range values
        enum boundsHandling
        {
            error,  // Exit with a FatalError (default)
            clamp   // Clamp to the nearest tabulated breakpoint
        };

        //- Enumeration names for handling out-of-range values
        static const NamedEnum<boundsHandling, 2> boundsHandlingNames;


private:

    // Private Data

        //- x breakpoints, strictly increasing
        List<scalar> x_;

        //- y breakpoints, strictly increasing
        List<scalar> y_;

        //- Table values, x_.size() rows by y_.size() columns each -
        //  List<List<scalar>>, not RectangularMatrix, specifically so
        //  both read and write use List<T>'s own shape-free bracket
        //  form ( (row) (row) ... ), no leading "m n" needed (or
        //  written) at all - RectangularMatrix's own Istream/Ostream
        //  operators always require/emit that prefix.
        List<List<scalar>> values_;

        //- Out-of-range handling, shared by both axes
        boundsHandling outOfBounds_;


    // Private Member Functions

        //- Bracketing index and interpolation fraction of v within
        //  the given breakpoints, applying outOfBounds_
        inline void locate
        (
            const List<scalar>& breakpoints,
            scalar v,
            label& i,
            scalar& lambda
        ) const;


public:

    //- Runtime type information
    TypeName("nonUniformTable2");


    // Constructors

        //- Construct from name and dictionary
        nonUniformTable2
        (
            const word& name,
            const unitSets& units,
            const dictionary& dict
        );


    // Member Functions

        //- Evaluate the function and return the result
        virtual scalar value(scalar x, scalar y) const;

        //- Write the function coefficients
        virtual void write(Ostream& os, const unitSets& units) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Function2s
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
