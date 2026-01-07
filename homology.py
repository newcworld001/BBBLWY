from collections.abc import Iterable

from sage.misc.misc_c import prod
from sage.misc.mrange import xmrange

from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import CyclotomicField

from sage.modules.free_module_element import vector

from sage.matrix.constructor import matrix
from sage.matrix.special import (
    identity_matrix,
    zero_matrix,
    block_diagonal_matrix,
    block_matrix,
)
from sage.matrix.symplectic_basis import symplectic_basis_over_ZZ

from flipper.kernel import Encoding, Lamination, Triangulation, norm

from .util import ScaledMatrix, make_standard_rep, EE
from .quantum_torus import QuantumTorus
from .surface import TeichRep, surface_type


def _path_to_lamination(t: Triangulation, path: Iterable[int]) -> Lamination:
    geometric = [0] * t.zeta
    algebraic = [0] * t.zeta
    for step in path:
        i = norm(step)
        geometric[i] += 1
        algebraic[i] += +1 if step >= 0 else -1
    return t.lamination(geometric, algebraic, remove_peripheral=False)


def _rotate_to_common_prefix(a: list[int], b: list[int]):
    sign = 0
    for i, ea in enumerate(a):
        for j, eb in enumerate(b):
            if ea == eb:
                sign = 1
                a = a[i:] + a[:i]
                b = b[j:] + b[:j]
            elif ea == ~eb:
                sign = -1
                a = a[i:] + a[:i]
                b = list(~x for x in reversed(b))
                b = b[~j:] + b[:~j]
        if sign:
            break
    if sign != 0:
        count = 0
        for ea, eb in zip(a, b):
            if ea == eb:
                count += 1
            else:
                return a, count


def _path_algebraic_intersection(
    t: Triangulation, p0: list[int], p1: list[int], L0: Lamination, L1: Lamination
):
    """Given paths in `t.homology_basis()`, return the algebraic intersection number."""
    p = _rotate_to_common_prefix(p0, p1)
    if p is None:
        return 0
    p0, count = p  # p0[:count] == p1[:count]

    a, b = t.square_about_edge(p0[count - 1])[:2]
    c, d = t.square_about_edge(p0[0])[2:]
    # #<-----#
    # |  a  ^^
    # |    / |
    # |b  / d|
    # |  /e  |
    # | /    |
    # V/ c   |
    # #----->#
    # __getitem__ is algebraic
    nac0 = L0[c] if L0[a] else 0  # positive if L0 is up
    nbd1 = L1[b] if L1[d] else 0  # positive if L1 is right
    if nac0 and nbd1:
        return -nac0 * nbd1
    nbd0 = L0[b] if L0[d] else 0  # positive if L0 is right
    nac1 = L1[c] if L1[a] else 0  # positive if L1 is up
    return nbd0 * nac1


# part of Lamination.is_homologous_to
def _homology_reduction_matrix(t: Triangulation):
    """Return a matrix M such that whose image is isomorphic to H_1.
    Apply to algebraic intersections to determine homology class."""

    matrix = identity_matrix(t.zeta)
    tree, dual_tree = t.tree_and_dual_tree(True)
    vertices_used = {vertex: False for vertex in t.vertices}
    # Get some starting vertices.
    for vertex in t.vertices:
        if not vertex.filled:
            vertices_used[vertex] = True

    while True:
        for edge in t.edges:
            if not tree[edge.index]:
                continue
            source, target = edge.source_vertex, edge.target_vertex
            if vertices_used[source] and not vertices_used[target]:
                # This implies edge goes between distinct vertices.
                vertices_used[target] = True
                for edge2 in t.edges:
                    # We have to skip the edge2 == ~edge case at this point as we are still
                    # removing it from various places.
                    if edge2.source_vertex == target and edge2 != ~edge:
                        s = +1 if edge2.is_positive() == edge.is_positive() else -1
                        matrix.add_multiple_of_row(edge2.index, edge.index, s)
                # Don't forget to go back and do edge which we skipped before.
                # matrix = matrix.elementary(edge.index, edge.index, -1)
                matrix[edge.index] = 0
                break
        else:
            break  # If there are no more to add then we've dealt with every edge.
    return matrix.matrix_from_rows(
        i for i in range(t.zeta) if not tree[i] and not dual_tree[i]
    )


class HomologyBasis:
    def __init__(self, t: Triangulation):
        """Construct a list of laminations representing a homology basis (peripherals included),
        the intersection matrix, and a matrix that takes the basis to a symplectic one.
        """
        g, p = surface_type(t)
        paths: list[list[int]] = t.homology_basis()
        assert len(paths) == 2 * g
        laminations = [_path_to_lamination(t, p) for p in paths]
        intersection_matrix = matrix(
            len(paths),
            lambda i, j: _path_algebraic_intersection(
                t, paths[i], paths[j], laminations[i], laminations[j]
            ),
        )
        # symplectic_basis_over_ZZ(M): Returns a pair (F, C) such that
        # the rows of C form a symplectic basis for M and F = C * M * C.transpose().
        # order of the basis is L+L'+K where L and L' are complementary Lagrangian and K is kernel.
        F, C = symplectic_basis_over_ZZ(intersection_matrix)
        peripherals: list[list[int]] = [
            [corner.edges[2].label for corner in t.corner_class_of_vertex(v)]  # type: ignore
            for v in t.vertices
            if not v.filled
        ][:-1]
        assert len(peripherals) == p - 1
        laminations += [_path_to_lamination(t, path) for path in peripherals]
        intersection_matrix = block_diagonal_matrix(
            [intersection_matrix, zero_matrix(p - 1)]
        )
        permutation = [i // 2 + (i % 2) * g for i in range(2 * g)]
        C = block_diagonal_matrix([C[permutation], identity_matrix(p - 1)])
        self.triangulation = t
        self.laminations = laminations
        self.intersection_matrix = intersection_matrix
        self.symplectic_basis = C
        self.reduction_matrix = _homology_reduction_matrix(t)


def homology_action(f: Encoding, basis: HomologyBasis = None, symplectic=True):
    if basis is None:
        basis = HomologyBasis(f.source_triangulation)
    source_basis = matrix(l.algebraic for l in basis.laminations)
    target_basis = matrix(f(l).algebraic for l in basis.laminations)
    reduction_matrix = basis.reduction_matrix
    source_matrix = reduction_matrix * source_basis.transpose()
    target_matrix = reduction_matrix * target_basis.transpose()
    # P = A.solve_right(B) means AP=B
    P = source_matrix.solve_right(target_matrix)
    if symplectic:
        C = basis.symplectic_basis.transpose()
        return (C.inverse() * P * C).change_ring(ZZ)
    else:
        return P


def make_homology_torus(
    t: Triangulation, basis: HomologyBasis = None, q="q", names="X"
):
    if basis is None:
        basis = HomologyBasis(t)
    return QuantumTorus(q, basis.intersection_matrix, names=names)


def make_homology_rep(
    t: Triangulation, x, weights: list[int] = None, basis: HomologyBasis = None
):
    if basis is None:
        basis = HomologyBasis(t)
    g, p = surface_type(t)
    if weights is None:
        weights = [0] * (p - 1)
    elif len(weights) == p and (s := sum(weights)):
        raise ValueError(f"Sum of puncture weights should be zero, got {s}.")
    R = make_homology_torus(t, basis=basis)
    B = basis.symplectic_basis
    n = (2 * x).denominator()
    D = n**g
    order, sympl_gens = make_standard_rep([1] * g, x)
    scaled_gens = []
    q = EE(x, CyclotomicField(x.denominator()))
    for c in B.inverse():
        # TODO: take powers before tensor product
        # especially because pow does not preserve sparseness
        raw_gen = prod((A**k).sparse_matrix() for A, k in zip(sympl_gens, c))
        weyl = -sum(c[2 * i] * c[2 * i + 1] for i in range(g))
        im_scale = q ** sum(w * e for w, e in zip(weights, c[2 * g :]))
        scaled_gens.append(ScaledMatrix(q**weyl * raw_gen, im_scale))
    return TeichRep(R, D, scaled_gens, x, order, sympl_basis=B)


def make_bundle_homology_rep(
    f: Encoding, x, weights: list[int] = None, basis: HomologyBasis = None
):
    t = f.source_triangulation
    if basis is None:
        basis = HomologyBasis(t)
    rho0 = make_homology_rep(t, x, weights, basis)
    action = homology_action(f, basis, False)
    target_gens = [rho0(c) for c in action.columns()]
    rho1 = rho0.replace(scaled_gens=target_gens)
    return rho0, rho1


def AH(
    f: Encoding, x, weights: list[int] = None, *, rho=None, basis: HomologyBasis = None
):
    """Return homology intertwiner `H`. Determinant is not normalized.
    Does not check if weight are invariant under mapping class."""
    t: Triangulation = f.source_triangulation
    g = t.genus
    if basis is None:
        basis = HomologyBasis(t)
    if rho is not None and weights is not None:
        raise UserWarning("Both rho and weights are given, weights is ignored")
    if rho is None:
        rho = make_bundle_homology_rep(f, x, weights, basis)
    rho0, rho1 = rho
    A0 = block_matrix(
        [rho1(gen).normal() - 1 for gen in rho0.sympl_basis[: 2 * g : 2]],
        ncols=1,
        subdivide=False,
    )
    K = A0.right_kernel()
    assert K.dimension() == 1
    v = K.basis()[0]
    B = lambda k: rho1(sum(c * g for c, g in zip(k, rho0.sympl_basis[1 : 2 * g : 2]))).normal()  # type: ignore
    H = matrix(B(k) * v for k in xmrange([rho0.order] * g, vector))
    return H.transpose()
