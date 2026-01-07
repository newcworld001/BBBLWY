r"""
Quantum Torus
based on `q`-Commuting Polynomials implementation in Sage by Travis Scrimshaw

AUTHORS:

- Tao Yu (2025-07-08): Initial version
"""

# ****************************************************************************
#       Copyright (C) 2025 Tao Yu <yut6 at sustech.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from collections.abc import Mapping

from sage.misc.cachefunc import cached_method
from sage.misc.latex import latex
from sage.structure.category_object import normalize_names
from sage.misc.misc_c import prod

from sage.rings.integer_ring import ZZ
from sage.rings.infinity import infinity

from sage.categories.algebras import Algebras
from sage.categories.commutative_rings import CommutativeRings

from sage.combinat.free_module import CombinatorialFreeModule
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing

from sage.sets.family import Family
from sage.groups.free_group import FreeGroup
from sage.modules.free_module import FreeModule
from sage.modules.free_module_element import FreeModuleElement, vector

from sage.structure.element import Matrix
from sage.matrix.constructor import matrix

from .util import Weyl_on_gens

# TODO: change docstrings
class QuantumTorus(CombinatorialFreeModule):
    r"""
    The algebra of `q`-commuting Laurent polynomials.

    Let `R` be a commutative ring, and fix an element `q \in R`. Let
    `B = (B_{xy})_{x,y \in I}`  be a skew-symmetric bilinear form with
    index set `I`. Let `R[I]_{q,B}` denote the Laurent polynomial ring in
    the variables `I` such that we have the `q`-*commuting* relation
    for `x, y \in I`:

    .. MATH::

        x y = q^{2B_{xy}} \cdot y x.

    This is a graded `R`-algebra with a natural basis given by Weyl-ordered
    monomials.
    """

    @staticmethod
    def __classcall__(cls, q, Q=None, names=None, subgroup=None):
        r"""
        Normalize input to ensure a unique representation.
        """
        if Q is None:
            if subgroup is None:
                raise ValueError("must provide at least one of B and subgroup")
            Q = subgroup.inner_product_matrix()
        if not Q.is_skew_symmetric():
            raise ValueError("the matrix must be skew symmetric")
        n = Q.nrows()
        if subgroup is None:
            subgroup = FreeModule(ZZ, n)
        elif isinstance(subgroup, Matrix):
            if subgroup.rank() < subgroup.nrows():
                raise ValueError("basis matrix must have full row rank")
            subgroup = FreeModule(ZZ, n).submodule_with_basis(subgroup.rows())
        if names is not None:
            names = normalize_names(n, names)
            assert n == len(names)
        elif subgroup.is_ambient():
            raise ValueError("names must be given when subgroup is ambient")
        if isinstance(q, str):
            q = LaurentPolynomialRing(ZZ, q).gen()
        return super().__classcall__(cls, q=q, Q=matrix(Q, immutable=True), names=names, subgroup=subgroup)

    def __init__(self, q, Q, names=None, subgroup=None):
        r"""
        Initialize ``self``.
        """
        assert subgroup is not None
        base_ring = q.parent()
        if base_ring not in CommutativeRings:
            raise ValueError("the base ring must be a commutative ring")
        self._q = q
        self._Q = Q
        P = subgroup.basis_matrix()
        try:
            self._Q_on_gens = (P * self._Q * P.transpose()).change_ring(ZZ)
        except TypeError:
            raise TypeError("q-commuting matrix must be integral at least on generators")
        self._ngens = subgroup.rank()
        if subgroup.is_ambient():
            assert names is not None
            self._display_group = FreeGroup(names=names, abelian=True, bracket=False)
        CombinatorialFreeModule.__init__(
            self,
            base_ring,
            basis_keys=subgroup,
            category=Algebras(base_ring).WithBasis().Graded(),
            prefix="",
            names=names,
            bracket=False,
            sorting_key=QuantumTorus._term_key,
        )

    def is_ambient(self):
        return self._indices.is_ambient()

    @staticmethod
    def _term_key(x):
        r"""
        Compute a key for ``x`` for comparisons.
        """
        L = x.list()
        L.reverse()
        return (sum(L), L)

    def _exponent_cast(self, x):
        if not isinstance(x, FreeModuleElement):
            x = vector(x, immutable=True)
        if x not in self._indices:
            raise ValueError("exponent vector must be in the subgroup in construction")
        return x if x.is_immutable() else vector(x, immutable=True)

    def _element_constructor_(self, x):
        if isinstance(x, Mapping):
            return self._from_dict({self._exponent_cast(k): x[k] for k in x})
        else:
            return self._from_dict({self._exponent_cast(x): 1})

    def _repr_(self):
        r"""
        Return a string representation of ``self``.
        """
        names = ", ".join(self.variable_names())
        return f"Quantum torus in {names} over {self.base_ring()} with q={self._q} and matrix:\n{self._Q}"

    def _latex_(self):
        r"""
        Return a latex representation of ``self``.
        """
        names = ", ".join(r"{}^{{\pm}}".format(v) for v in self.variable_names())
        return "{}[{}]_{{{}}}".format(latex(self.base_ring()), names, self._q)

    def _Weyl_wrap(self, m, f, use_generator_form: bool = None):
        if not m:
            return "1"
        if use_generator_form is None:
            use_generator_form = self._indices.is_ambient()
        if use_generator_form:
            G = self._display_group
            proxy_prod = G.prod(g**val for g, val in zip(G.gens(), m) if val != 0)
            res = f(proxy_prod)
        else:
            res = f(m)
        return "[" + res + "]"

    def _repr_term(self, m):
        r"""
        Return a latex representation of the basis element indexed by ``m``.
        """
        return self._Weyl_wrap(m, repr)

    def _latex_term(self, m):
        r"""
        Return a latex representation of the basis element indexed by ``m``.
        """
        return self._Weyl_wrap(m, latex)

    def gen(self, i) -> "QuantumTorus.Element":
        r"""
        Return the ``i``-generator of ``self``.
        """
        return self.monomial(self._indices.gen(i))

    @cached_method
    def gens(self) -> tuple["QuantumTorus.Element", ...]:
        r"""
        Return the generators of ``self``.
        """
        return tuple(self.monomial(g) for g in self._indices.gens())

    def ngens(self):
        return self._ngens

    @cached_method
    def algebra_generators(self):
        r"""
        Return the algebra generators of ``self``.
        """
        d = {v: self.gen(i) for i, v in enumerate(self.variable_names())}
        return Family(self.variable_names(), d.__getitem__, name="generator")

    def degree_on_basis(self, m):
        r"""
        Return the degree of the monomial index by ``m``.
        """
        return sum(m.list())

    def dimension(self):
        r"""
        Return the dimension of ``self``, which is `\infty`.
        """
        return infinity

    def q(self):
        """
        Return the parameter `q`.
        """
        return self._q

    def Q(self):
        return self._Q
    
    def Q_on_gens(self):
        return self._Q_on_gens

    @cached_method
    def one_basis(self):
        r"""
        Return the basis index of the element `1`.
        """
        return self._indices.zero()
    
    def _Weyl_exp(self, k, l):
        return k * self._Q * l

    def product_on_basis(self, x, y):
        r"""
        Return the product of two monomials whose exponents are given by ``x`` and ``y``.
        """
        # Special case for multiplying by 1
        if x == self.one_basis():
            return self.monomial(y)
        if y == self.one_basis():
            return self.monomial(x)

        qpow = self._Weyl_exp(x, y)
        ret = x + y
        ret.set_immutable()
        return self.term(ret, self._q**qpow)
    
    def _is_valid_homomorphism_(self, codomain, im_gens, base_map):
        if not (codomain.is_exact() and codomain.base_ring().is_exact()):
            return True
        return all(
            im_gens[i] * im_gens[j] == base_map(self._q**(2 * self._Q_on_gens[i, j])) * im_gens[j] * im_gens[i]
            for i in range(self._ngens)
            for j in range(i + 1, self._ngens)
        )

    class Element(CombinatorialFreeModule.Element):
        def decompose_if_monomial(self):
            """Return `(k, c)` if `self` is `c*X^k`, or raise a ValueError."""
            if len(self._monomial_coefficients) == 1:
                return next(iter(self._monomial_coefficients.items()))
            raise ValueError("self is not a monomial.")

        def __invert__(self):
            r"""
            Return the (multiplicative) inverse of ``self``.
            """
            try:
                m, c = self.decompose_if_monomial()
                ret = -m
                ret.set_immutable()
                return self.parent().term(ret, c.inverse())
            except ValueError as e:
                e.add_note("Only monomials are invertible.")
                raise e
        
        def _im_gens_(self, codomain, im_gens, base_map):
            res = codomain.zero()
            T = self.parent()
            q = T._q
            L = T._indices
            for m, c in self:   # self.__iter__ gives _monomial_coefficients.items()
                egens = L.coordinates(m)
                weyl = Weyl_on_gens(T._Q_on_gens, egens)
                res += base_map(c * q**weyl) * prod(g**k for g, k in zip(im_gens, egens))
            return res
