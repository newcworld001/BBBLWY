from typing import Any, NamedTuple, cast

from networkx import MultiGraph, bfs_edges
from flipper import create_triangulation
from flipper.kernel import (
    Triangulation,
    Encoding,
    Isometry,
    EdgeFlip,
    Vertex,
    Edge,
    Corner,
)

from sage.misc.cachefunc import cached_function
from sage.misc.misc_c import prod

from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from sage.categories.homset import Hom

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.matrix.special import zero_matrix, identity_matrix, block_matrix

from .quantum_torus import QuantumTorus
from .util import EE, framed_symplectic_basis, Weyl_on_gens, ScaledMatrix

#######################################
# Basic surface triangulation functions
#######################################


def triangulation_from_tuples(
    triangles: list[tuple[int, int, int]], vertex_states: dict[int, bool] = None
):
    r"""
    Example:
    >>> S = triangulation_from_tuples([(0,1,2), (0,5,4), (1,3,5), (2,4,3)]); S
    [(~5, ~1, 3), (~4, ~3, ~2), (~0, 5, 4), (0, 1, 2)]
    """
    nFace = len(triangles)
    nEdge = nFace * 3 // 2
    # create a copy and convert to Python ints to prevent argument changes and errors with ~
    faces = [[int(i) for i in t] for t in triangles]
    edge_first_face = [False] * nEdge
    for f in faces:
        for i in range(len(f)):
            e = f[i]
            if not edge_first_face[e]:
                edge_first_face[e] = True
            else:
                f[i] = ~e
    return create_triangulation(faces, vertex_states=vertex_states)


def act_on_lamination(f: Encoding, v: list[int]):
    S = f.source_triangulation
    A, _ = f.applied_geometric(S.lamination(v))
    return matrix(A) * vector(v)


def make_surface_form(t: Triangulation):
    n = t.zeta
    Q = zero_matrix(n, n)
    for face in t.triangles:
        a, b, c = face.indices
        # the edges are ordered CCW
        # so by convention of GY2, CCW == -1
        Q[a, b] += -1
        Q[b, c] += -1
        Q[c, a] += -1
    return Q - Q.transpose()


def make_surface_balancing(t: Triangulation):
    E = t.zeta
    F = t.num_triangles
    H = zero_matrix(F, E)
    for i, face in enumerate(t.triangles):
        for e in face.indices:
            H[i, e] = 1
    return H


def make_peripheral(t: Triangulation):
    n = t.zeta
    E = [t.edge_lookup[i] for i in range(n)]
    one_peri = lambda v: vector(
        ZZ, (int(e.source_vertex is v) + int(e.target_vertex is v) for e in E)
    )
    return {v: one_peri(v) for v in t.vertices if not v.filled}


def surface_type(t: Triangulation):
    return t.genus, t.num_unfilled_vertices


#######################################
# CF and balanced algebras
#######################################


def CF_decomposition(t: Triangulation, e: int = None, drop_puncture: int | None = -1):
    """
    :param Triangulation t: Triangulation of a surface
    :param int | None e: Optional. If given, the function attempts to make the corresponding unit vector part of the basis.
    :param int | None drop_puncture: Optional. If given, the corresponding vector is replaced by the all 1 vector. Negative index is allowed for lookup from the end.
    :return Matrix: A block matrix whose partition gives the 1, 2, 0 blocks. Symplectic duals are placed next to each other.

    Example:
    >>> S0 = triangulation_from_tuples([[0,1,4],[1,2,5],[2,3,4],[3,0,5]])
    >>> Q = make_surface_form(S0)
    >>> C = CF_decomposition(S0, 3); C
    [0 0 0 1 0 0]
    [0 0 0 0 0 1]
    [-----------]
    [1 2 1 0 1 1]
    [2 1 1 2 1 2]
    [-----------]
    [1 1 1 1 2 2]
    [1 1 1 1 0 0]
    >>> C * Q * C.transpose()
    [ 0  1  0  0  0  0]
    [-1  0  0  0  0  0]
    [ 0  0  0  2  0  0]
    [ 0  0 -2  0  0  0]
    [ 0  0  0  0  0  0]
    [ 0  0  0  0  0  0]
    """
    Q = make_surface_form(t)
    n = Q.nrows()
    g = t.genus
    basis = framed_symplectic_basis(Q, e)
    # note framed_symplectic_basis does not include the kernel part
    # then by general theory, basis[-2g:] can be made balanced
    one = vector(ZZ, [1] * n)
    H = make_surface_balancing(t)
    for i in range(-2 * g, 0):
        s = H * basis[i]
        if s % 2 != 0:
            basis[i] += one
    K = list(make_peripheral(t).values())
    if drop_puncture is not None:
        K[drop_puncture] = one
    return block_matrix(
        [[matrix(ZZ, len(B), n, B)] for B in (basis[: -2 * g], basis[-2 * g :], K)]
    )


def balanced_basis(t: Triangulation, e: int = None):
    """
    :param Triangulation t: Triangulation of a surface
    :param int | None e: Optional. If given, the function attempts to make the corresponding unit vector part of the basis.
    :return Matrix: A block matrix whose partition gives the 4, 2, 0 blocks. Symplectic duals are placed next to each other.

    Example:
    >>> S0 = triangulation_from_tuples([[0,1,4],[1,2,5],[2,3,4],[3,0,5]])
    >>> Q = make_surface_form(S0)

    >>> C = balanced_basis(S0, 3); C
    [0 0 0 2 0 0]
    [0 0 0 0 0 2]
    [-----------]
    [1 2 1 0 1 1]
    [2 1 1 2 1 2]
    [-----------]
    [1 1 1 1 2 2]
    [1 1 1 1 0 0]
    >>> C * Q * C.transpose()
    [ 0  4  0  0  0  0]
    [-4  0  0  0  0  0]
    [ 0  0  0  2  0  0]
    [ 0  0 -2  0  0  0]
    [ 0  0  0  0  0  0]
    [ 0  0  0  0  0  0]

    >>> H = make_surface_balancing(S0)
    >>> D, U, V = H.smith_form(); D == U * H * V
    True
    >>> D.diagonal()
    [1, 1, 1, 2]
    >>> C.row_module() == (V * diagonal_matrix([2,2,2,1,1,1])).column_module()
    True
    """
    g, p = surface_type(t)
    C = CF_decomposition(t, e, None)
    C[: -2 * g - p] *= 2
    return C


def make_CF_algebra(t: Triangulation, q="q", names="X"):
    # if kwds:
    #     return QuantumTorus(q, make_surface_form(t), names, CF_decomposition(t, **kwds))
    # else:
    return QuantumTorus(q, make_surface_form(t), names)


def make_balanced_algebra(t: Triangulation, A="A", names="Y", e: int = None):
    return QuantumTorus(A, make_surface_form(t) / 2, names, balanced_basis(t, e))


def puncture_weight_to_edge(t: Triangulation, weights: list[int], modulus: int):
    if modulus % 2 != 1:
        raise ValueError("Modulus must be odd.")
    p = t.num_vertices
    if len(weights) != p:
        raise ValueError("Number of weights does not match number of vertices.")
    KEY = "flipper"
    G: MultiGraph[Vertex] = MultiGraph(
        (e.source_vertex, e.target_vertex, {KEY: e}) for e in t.positive_edges
    )
    edge_weights = [0] * t.zeta
    remaining_weights = weights.copy()
    root_corner: Corner = t.corners[0]
    root: Vertex = root_corner.vertex
    tree = [
        (a, b, cast(Edge, G[a][b][0][KEY]))  # pyright: ignore[reportArgumentType]
        for a, b in bfs_edges(G, root)
    ]
    for a, b, e in reversed(tree):
        # e is an edge a--b, b is farther from the root
        weight_delta = remaining_weights[b.label]
        edge_weights[e.index] += weight_delta
        # weight on b is not useful anymore, but checked by assert later
        remaining_weights[b.label] -= weight_delta
        remaining_weights[a.label] -= weight_delta
    assert all(remaining_weights[i] == 0 for i in range(p) if i != root.label)
    half_w = remaining_weights[root.label] * (modulus + 1) // 2
    # a = opposite, counterclockwise order
    a, b, c = cast(list[Edge], root_corner.edges)
    if root_corner.vertices[1] is root:
        edge_weights[c.index] += half_w
    elif root_corner.vertices[2] is root:
        edge_weights[b.index] += half_w
    if root_corner.vertices.count(root) == 1:
        edge_weights[a.index] -= half_w
        edge_weights[b.index] += half_w
        edge_weights[c.index] += half_w
    return edge_weights


def edge_weights_to_puncture(t: Triangulation, weights: list[int], modulus: int = None):
    w = [0] * t.num_vertices
    for c in cast(list[list[Corner]], t.corner_classes):
        puncture: Vertex = c[0].vertex
        w[puncture.label] = sum(weights[corner.edges[1].index] for corner in c)
    return w if modulus is None else [x % modulus for x in w]


#######################################
# Rep class and operations
#######################################


class TeichRep:
    def __init__(
        self,
        domain: QuantumTorus,
        dimension: int,
        scaled_gens: list[ScaledMatrix],
        x,
        order: int,
        sympl_basis,
    ):
        self.domain = domain
        self.dimension = dimension
        self.scaled_gens = scaled_gens
        self.x = x
        self.order = order
        F = CyclotomicField(order)
        self.cyclo_field = F
        self.q = EE(x, F)
        self.base_map = Hom(domain.base_ring(), F)(self.q)
        self.sympl_basis = sympl_basis

    def _call_monomial(self, k, c) -> ScaledMatrix:
        q = self.domain._q
        L = self.domain._indices
        egens = L.coordinates(k)
        weyl = Weyl_on_gens(self.domain._Q_on_gens, egens)
        return self.base_map(c * q**weyl) * prod(
            g**e for g, e in zip(self.scaled_gens, egens)
        )

    def __call__(self, g: QuantumTorus.Element | list):
        """Specialization of `r._im_gens_`."""
        if isinstance(g, QuantumTorus.Element):
            terms = [self._call_monomial(k, c) for k, c in g]
            if len(terms) == 1:
                return terms[0]
            elif len(terms) > 1:
                return ScaledMatrix(sum(A.normal() for A in terms), 1)
            else:
                return ScaledMatrix(zero_matrix(self.dimension), 1)
        else:
            return self._call_monomial(g, 1)

    def replace(
        self,
        new_domain=None,
        scaled_gens: list[ScaledMatrix] = None,
        sympl_basis=None,
    ):
        if new_domain is None:
            new_domain = self.domain
        if scaled_gens is None:
            scaled_gens = self.scaled_gens
        if sympl_basis is None:
            sympl_basis = self.sympl_basis
        return TeichRep(
            new_domain,
            self.dimension,
            scaled_gens,
            self.x,
            self.order,
            sympl_basis,
        )

    def is_underlying_exact(self):
        return all(A.is_underlying_exact() for A in self.scaled_gens)

    def conjugation(self, C):
        new_gens = [A.conjugation(C) for A in self.scaled_gens]
        return self.replace(scaled_gens=new_gens)


def pullback_by_isom(
    m: QuantumTorus.Element, codomain: QuantumTorus, iso: Isometry, check=True
) -> QuantumTorus.Element:
    """
    Given `iso: t -> t'`, the pullback is `Fr(R) -> Fr(codomain)`,
    where `R = m.parent()` is for `t'` and `codomain` is for `t`
    """
    R1: QuantumTorus = m.parent()
    if check:
        Q = codomain.Q()
        Q1 = R1.Q()
        # P = perm_from_isom(iso)
        flag = (
            Q == make_surface_form(iso.source_triangulation)
            and Q1 == make_surface_form(iso.target_triangulation)
            # and P * Q1 * P.transpose() == Q
        )
        if not flag:
            raise ValueError("Inconsist input.")
    r = R1.ngens()
    coeff_dict: dict = m.monomial_coefficients()
    exp_pullback = lambda k1: vector(
        ZZ, (k1[iso.index_map[i]] for i in range(r)), immutable=True
    )
    return codomain({exp_pullback(k1): c for k1, c in coeff_dict.items()})


def push_rep_with_isom(
    r: TeichRep, new_domain: QuantumTorus, iso: Isometry, check=True
):
    """
    Given `iso: t -> t'`, the pullback is `Fr(new_domain) -> Fr(R)`,
    where `R = r.domain` is for `t` and `new_domain` is for `t'`,
    so the return is `new_domain -> R -> End`.
    """
    R = r.domain
    scaled_gens = [r(pullback_by_isom(g, R, iso, check)) for g in new_domain.gens()]
    B = r.sympl_basis
    n = B.ncols()
    f = iso.inverse()
    newB = B[:, [f.index_map[i] for i in range(n)]]
    newB.subdivide(B.subdivisions()[0])
    return r.replace(new_domain, scaled_gens, newB)


def _flip_linear_part(k1, Q1, e: int):
    n = len(k1)
    comp_fn = lambda a: (
        k1[a] if a != e else -k1[e] + sum(max(Q1[b, e], 0) * k1[b] for b in range(n))
    )
    return vector(comp_fn(a) for a in range(n))


@cached_function
def _flip_adjoint_vars():
    R = LaurentPolynomialRing(ZZ, "q")
    A = PolynomialRing(R, "x")
    return A, R.gen(), A.gen()


@cached_function
def _flip_adjoint_factor(r: int):
    assert r >= 0
    A, q, x = _flip_adjoint_vars()
    if r == 0:
        return A(1)
    return (1 + q ** (2 * r - 1) * x) * _flip_adjoint_factor(r - 1)


class FlipPullback(NamedTuple):
    poly: Any
    gen: QuantumTorus.Element
    mon: QuantumTorus.Element
    inv: bool

    def normal(self) -> QuantumTorus.Element:
        if not self.inv:
            R = self.mon.parent()
            return self.poly(q=R._q, x=self.gen) * self.mon
        else:
            raise ValueError("Inverse not supported.")


def pullback_by_flip(
    m1: QuantumTorus.Element, codomain: QuantumTorus, f: EdgeFlip, check=True
):
    """
    Pullback a monomial by a flip.
    Only the Weyl-normalized part is calculated. Coefficient is simply ignored.
    """
    try:
        k1 = m1.decompose_if_monomial()[0]
    except ValueError:
        raise NotImplementedError(
            "Common denominator in qtorus is too hard, so non-monomials are not yet implemented."
        )
    e = f.edge_index
    n = len(k1)
    Q1 = m1.parent().Q()
    if check:
        # note P^2=1
        P = matrix(_flip_linear_part(r, Q1, e) for r in identity_matrix(n))
        if P * Q1 * P.transpose() != codomain.Q():
            raise ValueError("Inconsist input.")
    k = _flip_linear_part(k1, Q1, e)
    # cheap trick to determine if we have CF or balanced
    d = 1 if codomain.is_ambient() else 2
    r = (Q1[e] * k1) // d
    inverse = r < 0
    if inverse:
        k = -k
        r = -r
    Xe = codomain(vector(ZZ, (d if i == e else 0 for i in range(n))))
    return FlipPullback(
        _flip_adjoint_factor(r)(q=codomain.q() ** d), Xe, codomain(k), inverse
    )
