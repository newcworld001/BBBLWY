from flipper.kernel import EdgeFlip

from sage.structure.element import get_coercion_model

from sage.rings.integer import Integer
from sage.rings.number_field.number_field import CyclotomicField

from sage.matrix.constructor import matrix
from sage.matrix.special import (
    identity_matrix,
    diagonal_matrix,
    block_matrix,
)

from sage.misc.misc_c import prod
from sage.functions.all import exp
from sage.symbolic.constants import pi, I
_2piI = 2 * pi * I

from .util import (
    EE,
    symplectic_factors,
    ScaledMatrix,
    root_of_unity_list_diag_ize,
    make_standard_rep,
    pivoted_R,
)
from .mapping_torus import Immersion, MappingTorus
from .surface import (
    CF_decomposition,
    TeichRep,
    make_CF_algebra,
    pullback_by_flip,
    push_rep_with_isom,
    surface_type,
    puncture_weight_to_edge
)


#######################################
# Concrete reps
#######################################


def make_CF_rep(imm: Immersion, x, weights: list[int] = None, **kwds):
    r"""
    Return value has an extra attribute `sympl_basis` which is the symplectic basis used in the construction.
    In particualar, the image of `_sympl_basis[0], _sympl_basis[2], ...` are diagonal.
    Note the root for the punctures are unspecified.
    >>> S0 = triangulation_from_tuples([[0,1,4],[1,2,5],[2,3,4],[3,0,5]])
    >>> f = S0.encode(to_python([{0:~1r}, 5, 3, 2, 5]))
    >>> m = MappingTorus(f)
    >>> R0 = make_CF_algebra(S0)
    >>> rho = make_CF_rep(m.immersions[0], 1/3)
    >>> rho.check_error()
    """
    if x.denominator() % 2 == 0:
        raise NotImplementedError(
            "Even denominator for Chekhov--Fock is not supported yet."
        )
    S = imm.surface
    g, p = surface_type(S)
    R = make_CF_algebra(S)
    B = CF_decomposition(S, **kwds)
    n = (2 * x).denominator()
    m = (4 * x).denominator()
    nEven = 2 * g - 3 + p
    sympl_rank = nEven + g
    D = n**nEven * m**g
    ds = [1] * nEven + [2] * g
    order, sympl_gens = make_standard_rep(ds, x)
    if kwds.pop("debug", False):
        row_dividers = B.subdivisions()[0]
        assert row_dividers[0] == 2 * nEven and row_dividers[1] == 2 * sympl_rank
        assert ds == symplectic_factors(B * R.Q() * B.transpose())
        assert all(A.nrows() == D for A in sympl_gens)
    scaled_gens = []
    q = EE(x, CyclotomicField(x.denominator()))
    if weights:
        edge_adjust = puncture_weight_to_edge(S, weights, n)
        adjusted_shear_bend = imm.shear_bend.copy()
        F = adjusted_shear_bend[0].parent()
        for i, k in enumerate(edge_adjust):
            adjusted_shear_bend[i] += F(k * _2piI)
    else:
        adjusted_shear_bend = imm.shear_bend
    for i, c in enumerate(B.inverse()):
        # TODO: take powers before tensor product
        # especially because pow does not preserve sparseness
        raw_gen = prod((A**k).sparse_matrix() for A, k in zip(sympl_gens, c))
        weyl = -sum(ds[i] * c[2 * i] * c[2 * i + 1] for i in range(sympl_rank))
        im_scale = exp(adjusted_shear_bend[i] / n)
        scaled_gens.append(ScaledMatrix(q**weyl * raw_gen, im_scale))
    return TeichRep(R, D, scaled_gens, x, order, sympl_basis=B)


def make_bundle_CF_rep(m: MappingTorus, x, weights: list[int] = None) -> list[TeichRep]:
    """
    Return a list of `TeichRep` of the CF algebras using data from `m: MappingTorus`.
    Entries correspond to `m.immersions`.
    """
    f = m.mapping_class
    if not isinstance(f[-1], EdgeFlip):
        raise NotImplementedError("m.mapping_class must start with a flip")
    l = len(f)
    res = []
    # since each step uses the target, the list starts at index 1
    for i in range(l):
        move = f[~i]
        if isinstance(move, EdgeFlip):
            target_i = (i + 1) % l
            imm = m.immersions[target_i]
            res.append(make_CF_rep(imm, x, weights=weights, e=move.edge_index))
        else:
            R0 = make_CF_algebra(move.target_triangulation)
            src_rep = res[-1]
            res.append(push_rep_with_isom(src_rep, R0, move))
    # after construction, we rotate the list so index start at 0
    return [res[-1]] + res[:-1]


def _get_sympl_rank(rho: TeichRep) -> int:
    return rho.sympl_basis.subdivisions()[0][1] // 2


def _make_inter_check(rho0, rho1, target_ab_vec, target_diag_mat, f, pullback_ab):
    assert _get_sympl_rank(rho0) == _get_sympl_rank(rho1)
    if not all(gen.is_diagonal() for gen in target_diag_mat):
        raise ValueError(
            "Even rows of sympl_basis must be diagonal in the target TeichRep."
        )
    if (target_ab_vec * rho1.domain._Q).column(f.edge_index) != 0:
        raise ValueError("Flip edge must be diagonal in target TeichRep.")
    assert all(p.poly == 1 and not p.inv for p in pullback_ab)


def _make_diag_comm_matrix(A, B, F=None):
    """Find the coefficient matrix for `A * D == D * B` where `D` is diagonal
    and the variables are the diagonal elements of `D`."""
    if F is None:
        F = get_coercion_model().common_parent(A.base_ring(), B.base_ring())
    dictA: dict = A.dict()
    P = matrix(F, len(dictA), A.ncols())
    for k, ((i, j), a) in enumerate(dictA.items()):
        P[k, j] = A[i, j]
        P[k, i] = -B[i, j]
    return P


def make_intertwiner(
    rho0: TeichRep, rho1: TeichRep, f: EdgeFlip, separate=False, check=True
):
    """
    Suppose `f: L0 -> L1` and `rhoi: Teich(Li) -> End(V)` is irreducible.
    If `separate` is `True`, return `C, D, eps` such that `rho1(x) == H^{-1} rho0(f^*(x)) H`
    for all `x` in `Teich(L1)`
    where `H == C^{-1}D` and `eps` is the discarded pivot of QR in the implementation.
    Otherwise, return `H` as above.
    """
    R1 = rho1.domain
    R0 = rho0.domain
    order = rho0.order
    F = CyclotomicField(order)
    #####
    # diagonalize the pullback of the generators that map to diagonal by rho1
    sympl_rank = _get_sympl_rank(rho1)
    target_ab_vec = rho1.sympl_basis[list(range(0, 2 * sympl_rank, 2))]
    target_diag_mat = [rho1(r) for r in target_ab_vec]
    pullback_ab = [pullback_by_flip(R1(k), R0, f, check) for k in target_ab_vec]
    if check:
        _make_inter_check(rho0, rho1, target_ab_vec, target_diag_mat, f, pullback_ab)
    # print(target_ab_vec, [p.mon for p in pullback_ab])
    rebalanced = [
        rho0(p.mon).rebalance_to(A.scale, order, F)
        for p, A in zip(pullback_ab, target_diag_mat)
    ]
    # eig_mats = dict()
    # for p, A in zip(pullback_ab, target_diag_mat):
    #     x = rho0(p.mon).rebalance_to(A.scale, order, F)
    #     eig_mats = x.root_of_unity_eigenspaces(order, eig_mats)
    eig_mats = root_of_unity_list_diag_ize(order, rebalanced)
    target_eig_zip = zip(*[A.matrix.diagonal() for A in target_diag_mat])
    eig_mat_sorted = [eig_mats[eig_tuple] for eig_tuple in target_eig_zip]
    C = block_matrix(eig_mat_sorted, ncols=1, subdivide=False).inverse()
    rhoC = rho0.conjugation(C)
    # at this point, rhoC(p) == A for p, A in zip(pullback_ab_gens, target_diag_mat)])
    # so only the symplectic dual generators need to be matched
    #####
    # find a diagonal matrix D such that rho1 == ~D * rhoC(pullback) * D
    target_dual_vec = rho1.sympl_basis[list(range(1, 2 * sympl_rank, 2))]
    pullback_dual = [pullback_by_flip(R1(k), R0, f, check) for k in target_dual_vec]
    Xe = pullback_dual[0].gen
    Xe_mat = rhoC(Xe)
    if check:
        # due to the pullback formula, we have the following
        assert all(p.gen == Xe and not p.inv for p in pullback_dual)
        assert all(p.poly == 1 for p in pullback_dual[1:])
        # and pullback_dual[0].poly == 1 + q * x
        assert Xe_mat.matrix.is_diagonal()
    # print(target_dual_vec)
    # print(pullback_dual)
    target_dual_mat = [rho1(r) for r in target_dual_vec]
    pullback_dual_mat = [rhoC(p.mon) for p in pullback_dual]
    A0 = (
        pullback_dual[0].poly(q=rhoC.q, x=Xe_mat.normal())
        * pullback_dual_mat[0].normal()
    )
    P0 = _make_diag_comm_matrix(A0, target_dual_mat[0].normal())
    if sympl_rank > 1:
        # By rebalancing, these calculations are exact
        P = []
        for A, B in zip(pullback_dual_mat[1:], target_dual_mat[1:]):
            A.rebalance_to(B.scale, order, F)
            P.append(_make_diag_comm_matrix(A.matrix, B.matrix, F))
        P1 = block_matrix(P, ncols=1)
        B = P1.right_kernel_matrix(basis="computed")
    else:
        B = identity_matrix(P0.ncols())
    R, P = pivoted_R(P0 * B.transpose())
    r = R.ncols() - 1
    d = R[:r, P].right_kernel_matrix("computed") * B
    assert d.nrows() == 1
    d = d[0]
    det = C.det() * prod(d)
    D = diagonal_matrix(d / det ** (Integer(1) / len(d)))
    if separate:
        return C, D, R[r, r], rebalanced
    else:
        return C * D


def test_intertwiner(rho0: TeichRep, rho1: TeichRep, f: EdgeFlip, H):
    """
    Suppose `f: L0 -> L1` and `rhoi: Teich(Li) -> End(V)`.
    Check `rho1(x) == H^{-1} rho0(f^*(x)) H` for some generating set of `Teich(L1)`.
    """
    errors = []
    for g1 in rho1.domain.gens():
        # for k in rho1.sympl_basis:
        #     g1 = rho1.domain(k)
        gt = pullback_by_flip(g1, rho0.domain, f)
        if gt.inv:
            g1 = g1.inverse()
            gt = gt._replace(inv=False)
        g0 = gt.normal()
        A = H.inverse() * rho0(g0).normal() * H
        B = rho1(g1).normal()
        errors.append((A - B).norm(1))
    return max(errors)


def ACF(m: MappingTorus, x, weights: list[int] = None, *, rho=None):
    """Return BWY intertwiner `H` normalized by `det(H) == 1`."""
    f = m.mapping_class
    if rho is not None and weights is not None:
        raise UserWarning("Both rho and weights are given, weights is ignored")
    if rho is None:
        rho = make_bundle_CF_rep(m, x, weights)
    Hs = [make_intertwiner(rho[i], rho[i + 1], f[-i - 1]) for i in range(len(f) - 1)]
    return prod(Hs)
