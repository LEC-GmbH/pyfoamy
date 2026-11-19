/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2014-2026 OpenFOAM Foundation
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

#include "pyfoamyPengRobinsonGas.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

template<class Specie>
inline void Foam::pyfoamyPengRobinsonGas<Specie>::sqrtAAlpha
(
    const scalar T,
    scalar& s,
    scalar& dsdT,
    scalar& d2sdT2
) const
{
    const scalar sqrtT = sqrt(T);

    s = A0_ - A1_*sqrtT;
    dsdT = -A1_/(2*sqrtT);
    d2sdT2 = A1_/(4*T*sqrtT);
}


template<class Specie>
inline void Foam::pyfoamyPengRobinsonGas<Specie>::AB
(
    const scalar p,
    const scalar T,
    scalar& A,
    scalar& B
) const
{
    const scalar s = A0_ - A1_*sqrt(T);

    A = sqr(s)*p/sqr(RR*T);
    B = b_*p/(RR*T);
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::logTerm
(
    const scalar Z,
    const scalar B
) const
{
    const scalar root2 = sqrt(2.0);

    return log((Z + (1 + root2)*B)/(Z + (1 - root2)*B))/(2*root2*b_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Specie>
inline Foam::pyfoamyPengRobinsonGas<Specie>::pyfoamyPengRobinsonGas
(
    const Specie& sp,
    const scalar A0,
    const scalar A1,
    const scalar b
)
:
    Specie(sp),
    A0_(A0),
    A1_(A1),
    b_(b)
{}


template<class Specie>
inline Foam::pyfoamyPengRobinsonGas<Specie>::pyfoamyPengRobinsonGas
(
    const word& name,
    const pyfoamyPengRobinsonGas& pg
)
:
    Specie(name, pg),
    A0_(pg.A0_),
    A1_(pg.A1_),
    b_(pg.b_)
{}


template<class Specie>
inline Foam::autoPtr<Foam::pyfoamyPengRobinsonGas<Specie>>
Foam::pyfoamyPengRobinsonGas<Specie>::clone() const
{
    return autoPtr<pyfoamyPengRobinsonGas<Specie>>
    (
        new pyfoamyPengRobinsonGas<Specie>(*this)
    );
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::rho
(
    scalar p,
    scalar T
) const
{
    return p/(Z(p, T)*this->R()*T);
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::h
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);
    const scalar Z = this->Z(p, T);

    return
    (
        RR*T*(Z - 1)
      + (2*T*s*ds - sqr(s))*logTerm(Z, B)
    )/this->W();
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::Cp
(
    scalar p,
    scalar T
) const
{
    return Cv(p, T) + CpMCv(p, T) - this->R();
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::e
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);

    return (2*T*s*ds - sqr(s))*logTerm(Z(p, T), B)/this->W();
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::Cv
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);

    return 2*T*(sqr(ds) + s*d2s)*logTerm(Z(p, T), B)/this->W();
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::sp
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);
    const scalar Z = this->Z(p, T);

    return
        this->R()*(log(Z - B) - log(p/pStd))
      + 2*s*ds*logTerm(Z, B)/this->W();
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::sv
(
    scalar p,
    scalar T
) const
{
    NotImplemented;
    return 0;
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::psi
(
    scalar p,
    scalar T
) const
{
    return 1/(Z(p, T)*this->R()*T);
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::Z
(
    scalar p,
    scalar T
) const
{
    scalar A, B;
    AB(p, T, A, B);

    const scalar a2 = B - 1;
    const scalar a1 = A - 2*B - 3*sqr(B);
    const scalar a0 = -A*B + sqr(B) + pow3(B);

    const scalar Q = (3*a1 - a2*a2)/9.0;
    const scalar Rl = (9*a2*a1 - 27*a0 - 2*a2*a2*a2)/54.0;

    const scalar Q3 = Q*Q*Q;
    const scalar D = Q3 + Rl*Rl;

    if (D <= 0)
    {
        // Three real roots: the largest, the vapour's
        const scalar th = ::acos(min(max(Rl/sqrt(-Q3), -1), 1));
        const scalar qm = 2*sqrt(-Q);

        return qm*cos(th/3.0) - a2/3.0;
    }
    else
    {
        const scalar D05 = sqrt(D);

        return cbrt(Rl + D05) + cbrt(Rl - D05) - a2/3.0;
    }
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::CpMCv
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);
    const scalar Z = this->Z(p, T);

    const scalar M = (sqr(Z) + 2*B*Z - sqr(B))/(Z - B);
    const scalar N = 2*s*ds*B/(b_*RR);

    return this->R()*sqr(M - N)/(sqr(M) - 2*A*(Z + B));
}


template<class Specie>
inline Foam::scalar Foam::pyfoamyPengRobinsonGas<Specie>::alphav
(
    scalar p,
    scalar T
) const
{
    scalar s, ds, d2s, A, B;
    sqrtAAlpha(T, s, ds, d2s);
    AB(p, T, A, B);
    const scalar Z = this->Z(p, T);

    // dZ/dT at constant p, by implicit differentiation of the cubic in Z
    const scalar dAdT = p*(2*s*ds - 2*sqr(s)/T)/sqr(RR*T);
    const scalar dBdT = -B/T;
    const scalar FA = Z - B;
    const scalar FB = sqr(Z) - (2 + 6*B)*Z - A + 2*B + 3*sqr(B);
    const scalar FZ = 3*sqr(Z) + 2*(B - 1)*Z + A - 2*B - 3*sqr(B);

    return 1/T - (FA*dAdT + FB*dBdT)/(FZ*Z);
}


// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Specie>
inline void Foam::pyfoamyPengRobinsonGas<Specie>::operator+=
(
    const pyfoamyPengRobinsonGas<Specie>& pg
)
{
    scalar X1 = this->Y()/this->W();
    Specie::operator+=(pg);

    if (mag(this->Y()) > small)
    {
        X1 *= this->W()/this->Y();
        const scalar X2 = this->W()*pg.Y()/(pg.W()*this->Y());

        A0_ = X1*A0_ + X2*pg.A0_;
        A1_ = X1*A1_ + X2*pg.A1_;
        b_ = X1*b_ + X2*pg.b_;
    }
}


template<class Specie>
inline void Foam::pyfoamyPengRobinsonGas<Specie>::operator*=(const scalar s)
{
     Specie::operator*=(s);
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

template<class Specie>
Foam::pyfoamyPengRobinsonGas<Specie> Foam::operator+
(
    const pyfoamyPengRobinsonGas<Specie>& pg1,
    const pyfoamyPengRobinsonGas<Specie>& pg2
)
{
    const Specie sp
    (
        static_cast<const Specie&>(pg1)
      + static_cast<const Specie&>(pg2)
    );

    if (mag(sp.Y()) < small)
    {
        return pyfoamyPengRobinsonGas<Specie>(sp, pg1.A0_, pg1.A1_, pg1.b_);
    }

    const scalar X1 = sp.W()*pg1.Y()/(pg1.W()*sp.Y());
    const scalar X2 = sp.W()*pg2.Y()/(pg2.W()*sp.Y());

    return pyfoamyPengRobinsonGas<Specie>
    (
        sp,
        X1*pg1.A0_ + X2*pg2.A0_,
        X1*pg1.A1_ + X2*pg2.A1_,
        X1*pg1.b_ + X2*pg2.b_
    );
}


template<class Specie>
Foam::pyfoamyPengRobinsonGas<Specie> Foam::operator*
(
    const scalar s,
    const pyfoamyPengRobinsonGas<Specie>& pg
)
{
    return pyfoamyPengRobinsonGas<Specie>
    (
        s*static_cast<const Specie&>(pg),
        pg.A0_,
        pg.A1_,
        pg.b_
    );
}


template<class Specie>
Foam::pyfoamyPengRobinsonGas<Specie> Foam::operator==
(
    const pyfoamyPengRobinsonGas<Specie>& pg1,
    const pyfoamyPengRobinsonGas<Specie>& pg2
)
{
    return pyfoamyPengRobinsonGas<Specie>
    (
        static_cast<const Specie&>(pg1) == static_cast<const Specie&>(pg2),
        pg1.A0_,
        pg1.A1_,
        pg1.b_
    );
}


// ************************************************************************* //
