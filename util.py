from typing import cast
from collections.abc import Iterable, Generator
from copy import copy

from flipper.kernel import (
    Isometry,
    Encoding,
    Move,
    EdgeFlip,
    Triangulation,
    AssumptionError,
)

from sage.rings.integer_ring import ZZ
from sage.rings.complex_double import CDF

from sage.rings.integer import Integer
from sage.rings.number_field.number_field import CyclotomicField

from sage.structure.element import CommutativeRingElement

from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.matrix.special import identity_matrix, diagonal_matrix

from sage.rings.universal_cyclotomic_field import E
from sage.arith.misc import xgcd
from sage.matrix.symplectic_basis import symplectic_basis_over_ZZ

from sage.misc.functional import round
from sage.functions.all import arg
from sage.symbolic.constants import pi


def to_python(x):
    if isinstance(x, (list, tuple)):
        T = type(x)
        return T(to_python(e) for e in x)
    elif isinstance(x, dict):
        return {int(k): to_python(x[k]) for k in x}
    else:
        return int(x)


def perm_from_isom(f: Isometry):
    """
    Example:
    >>> S = triangulation_from_tuples([(0,1,2), (0,5,4), (1,3,5), (2,4,3)])
    >>> S1 = triangulation_from_tuples([(1,2,0), (1,5,4), (2,3,5), (0,4,3)])
    >>> f = _find_isometry_noexcept(S, S1, int(1))
    >>> Qs = make_surface_form(f.source_triangulation)
    >>> Qt = make_surface_form(f.target_triangulation)
    >>> P = perm_from_isom(f)
    >>> P * Qt * P.transpose() == Qs
    True
    >>> P * Qs * P.transpose() == Qt
    False
    """
    n = f.source_triangulation.zeta
    return matrix(n, n, {p: 1 for p in f.index_map.items()})


def find_symplectic_dual(Q, v):
    r = v * Q
    d, u = r[0], [1]
    for i in range(1, len(r)):
        d0, s, t = xgcd(d, r[i])
        u = [s * x for x in u] + [t]
        d = d0
    return (d, vector(ZZ, u)) if d >= 0 else (-d, -vector(ZZ, u))


def symplectic_factors(F):
    r = F.nrows() // 2
    return [d for i in range(r) if (d := F[2 * i, 2 * i + 1]) != 0]


def is_standard_symplectic_pairing(F):
    if not F.is_skew_symmetric():
        return False
    n = F.nrows()
    factors = symplectic_factors(F)
    for i in range(n - 1):
        b, r = i // 2, i % 2
        for j in range(i + 1, n):
            if r == 0 and j == i + 1:
                if F[i, j] != factors[b]:
                    return False
            else:
                if F[i, j] != 0:
                    return False
    return True


def merge_symplectic_basis(*args):
    res = []
    for S in args:
        n = len(S) // 2
        for i in range(n):
            res.append(S[i])
            res.append(S[i + n])
    return res


def framed_symplectic_basis(Q, e: int = None):
    n = Q.nrows()
    K = Q.kernel()
    if e is not None:
        a = vector(ZZ, n, {e: 1})
        d, b = find_symplectic_dual(Q, a)
        Qe = matrix(ZZ, K.basis() + [a, b]) * Q
        L = [v.lift() for v in (Qe.right_kernel() / K).gens()]
        L = matrix(ZZ, len(L), n, L)
        QL = L * Q * L.transpose()
        F, C = symplectic_basis_over_ZZ(QL)
        # by sage doc, F == C * QL * C.transpose() == (C * L) * Q * (C * L).transpose()
        C = C * L
        assert all(x == 1 or x == 2 for x in symplectic_factors(F) + [d])
        if d == 1:
            return merge_symplectic_basis([a, b], C.rows())
        else:
            return merge_symplectic_basis(C.rows(), [a, b])
    else:
        L = matrix(ZZ, [v.lift() for v in (K.ambient_module() / K).gens()])
        C = symplectic_basis_over_ZZ(L * Q * L.transpose())[1] * L
        return merge_symplectic_basis(C.rows())


def matrix_tensor(l: list):
    """
    The order of the matrix elements is induced by vector_tensor.
    Example:
    >>> A = [random_matrix(ZZ, 2, 3) for i in range(3)]
    >>> v = [random_vector(ZZ, 3) for i in range(3)]
    >>> matrix_tensor(A) * vector_tensor(v) == vector_tensor([B * x for B, x in zip(A, v)])
    True
    """
    res = l[0]
    for A in l[1:]:
        res = res.tensor_product(A)
    return res


def vector_tensor(l: Iterable):
    """
    The tensor product is in row major order, i.e., res[1] == l0[0]*l1[0]*...*lk[1] and so on
    Example:
    >>> v = [vector([1,2]), vector([3,pi]), vector([5,I])]
    >>> vector_tensor(v)
    (15, 3*I, 5*pi, I*pi, 30, 6*I, 10*pi, 2*I*pi)
    """
    return matrix_tensor([x.row() for x in l]).row(0)


def extend_by_identity(A, sizes: list[int], slot: int):
    factors = [identity_matrix(s) for s in sizes[:slot]] if slot > 0 else []
    factors.append(A)
    if slot < len(sizes) - 1:
        factors += [identity_matrix(s, sparse=True) for s in sizes[slot + 1 :]]
    return matrix_tensor(factors)


def EE(x, base_ring=CDF):
    return base_ring(E(x.denominator(), x.numerator()))


def Weyl_on_gens(B_on_gens, egens):
    n = len(egens)
    return -sum(
        B_on_gens[i, j] * egens[i] * egens[j] for i in range(n) for j in range(i + 1, n)
    )


def as_root_of_unity(r, order, F):
    i = round(arg(r) / (2 * pi) * order)
    return EE(i / order, F)


class ScaledMatrix:
    def __init__(self, matrix, scale):
        self.matrix = matrix
        self.scale = scale

    def __eq__(self, other):
        return self.matrix == other.matrix and self.scale == other.scale

    def __mul__(self, other):
        if isinstance(other, ScaledMatrix):
            return ScaledMatrix(self.matrix * other.matrix, self.scale * other.scale)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, CommutativeRingElement):
            return ScaledMatrix(self.matrix, self.scale * other)
        return NotImplemented

    def __pow__(self, exponent: int):
        return ScaledMatrix(
            (self.matrix**exponent).sparse_matrix(), self.scale**exponent
        )

    def inverse(self):
        return ScaledMatrix(self.matrix.inverse(), 1 / self.scale)

    def normal(self):
        return self.scale * self.matrix

    def is_diagonal(self) -> bool:
        return self.matrix.is_diagonal()

    def underlying_parent(self):
        return self.matrix.parent()

    def is_underlying_exact(self) -> bool:
        return self.underlying_parent().is_exact()

    def root_of_unity_eigenspaces(self, order: int, subspace_matrices: dict):
        """Note these are left eigenspaces"""
        F = self.underlying_parent().base_ring()
        n = self.matrix.nrows()
        if len(subspace_matrices) == 0:
            subspace_matrices = {tuple(): identity_matrix(n)}
        eig_vals = [EE(Integer(i) / order, F) for i in range(order)]
        eig_matrices = dict()
        for other_eig, B in subspace_matrices.items():
            for eig_value in eig_vals:
                A = B.solve_left(B * (self.matrix - eig_value))
                eig_matrix = A.left_kernel_matrix(basis="computed") * B
                eig_matrices[other_eig + (eig_value,)] = eig_matrix
        return eig_matrices

    def conjugation(self, C):
        new_mat = C.inverse() * self.matrix * C
        return ScaledMatrix(new_mat, self.scale)

    def rebalance_to(self, s, order, F):
        d = as_root_of_unity(s / self.scale, order, F)
        self.matrix /= d
        self.scale *= d
        return self


def root_of_unity_list_diag_ize(order: int, mats: list[ScaledMatrix]):
    """Note these are left eigenspaces"""
    eig_mats = dict()
    for A in mats:
        eig_mats = A.root_of_unity_eigenspaces(order, eig_mats)
    return eig_mats


def scale_matrix(q, s: int):
    return diagonal_matrix(q**i for i in range(s))


def shift_matrix(s: int):
    entries = {((j + 1) % s, j): 1 for j in range(s)}
    return matrix(s, s, entries)


def make_standard_rep(ds: list[int], x):
    # The relation between generators only depend on q^2.
    # In particular, dimensions only depend on q^2.
    qexp = [2 * d * x for d in ds]
    sizes: list[int] = [a.denominator() for a in qexp]
    order = max(sizes)
    F = CyclotomicField(order)
    blocks = []
    for a, s in zip(qexp, sizes):
        q2d = EE(a, F)
        blocks += [scale_matrix(q2d, s), shift_matrix(s)]
    gens = [extend_by_identity(A, sizes, i // 2) for i, A in enumerate(blocks)]
    return order, gens


def _QR_col_pivoting(R):
    norms = [v.norm() for v in R.columns()]
    return max(enumerate(norms), key=lambda p: p[1])


def _perm_inverse(P):
    res = [-1] * len(P)
    for i, x in enumerate(P):
        res[x] = i
    return res


def pivoted_R(A):
    """
    Since sage refuses to QR numerical matrices except RDF and CDF,
    we need to write our own.
    This method returns `R, P` where `A == Q * R[:, P]`.
    """
    R = matrix(A)  # mutable copy
    P = list(range(R.ncols()))
    s = min(R.nrows(), R.ncols())
    for i in range(s):
        p, norm = _QR_col_pivoting(R[i:, i:])
        p += i
        if i != p:
            R.swap_columns(i, p)
            P[i], P[p] = P[p], P[i]
        # form Householder data
        v = R.column(i)[i:]
        s = norm if v[0] >= 0 else -norm
        v[0] += s
        b = 1 / (s * v[0])
        # Q_i = 1 - b * v * v^t
        R[i, i] = -s
        R[i + 1 :, i] = 0
        if i < R.ncols():
            diff = v.column() * (v.conjugate() * R[i:, i + 1 :]).row()
            R[i:, i + 1 :] -= b * diff
    return R, _perm_inverse(P)


def scaleness(A):
    n = A.nrows()
    s = A.trace() / n
    return s, (A - s).norm(1) / A.norm(1)


def qcomm_error(A, B, q):
    s, e = scaleness((q * B * A).solve_right(A * B))
    return max(abs(s - 1), e)


def check_commuting_gens(L: list, n: int):
    r = len(L)
    se = [scaleness(A**n) for A in L]
    error = max(s[1] for s in se)
    error += max(
        qcomm_error(L[i], L[j], 1) for i in range(r - 1) for j in range(i + 1, r)
    )
    return [s[0] for s in se], error


def _index_to_normalize(seq: list[Move]) -> Generator[int]:
    n = len(seq)
    for _ in range(n * n * n):
        idx = -1
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            if isinstance(a, EdgeFlip) and isinstance(b, Isometry):
                idx = i
            elif isinstance(a, Isometry) and isinstance(b, Isometry):
                idx = i
            elif isinstance(a, EdgeFlip) and isinstance(b, EdgeFlip):
                if a.edge_index == b.edge_index:
                    idx = i
        if idx >= 0:
            yield idx
        else:
            return


def normalize_mapping_class(f: Encoding) -> Encoding:
    seq = cast(list[Move], copy(f.sequence))
    for i in _index_to_normalize(seq):
        a = seq[i]
        b = seq[i + 1]
        if isinstance(a, EdgeFlip) and isinstance(b, Isometry):
            e = cast(int, a.edge_label)
            e0 = cast(int, b.inverse_label_map[e])
            new_surface = cast(Triangulation, b.source_triangulation).flip_edge(e0)
            new_flip = EdgeFlip(b.source_triangulation, new_surface, e0)
            new_iso_map = copy(b.label_map)
            if (e0 >= 0 and e < 0) or (e0 < 0 and e >= 0):
                new_iso_map[e0] = ~new_iso_map[e0]
                new_iso_map[~e0] = ~new_iso_map[~e0]
            seq[i] = Isometry(new_surface, a.target_triangulation, new_iso_map)
            seq[i + 1] = new_flip
        elif isinstance(a, Isometry) and isinstance(b, Isometry):
            new_iso_map = {i: a.label_map[j] for i, j in b.label_map.items()}
            seq[i] = Isometry(
                b.source_triangulation, a.target_triangulation, new_iso_map
            )
            seq.pop(i + 1)
        else:
            assert isinstance(a, EdgeFlip) and isinstance(b, EdgeFlip)
            assert a.edge_index == b.edge_index
            if a.edge_label == ~b.edge_label:
                seq.pop(i + 1)
                seq.pop(i)
            elif a.edge_label == b.edge_label:
                num_edges = b.source_triangulation.zeta
                e = b.edge_index
                new_iso_map = {i: i for i in range(-num_edges, num_edges)}
                new_iso_map[e] = ~e
                new_iso_map[~e] = e
                seq[i] = Isometry(
                    b.source_triangulation, a.target_triangulation, new_iso_map
                )
                seq.pop(i + 1)
    # at this point, seq is [iso, flips...], we just need to normalize flip labels
    S0 = f.source_triangulation
    e0 = cast(Isometry, seq[0]).label_map[0]
    flip_indices = [f.edge_index for f in cast(list[EdgeFlip], seq[1:])]
    try:
        f1 = S0.encode([{0: e0}] + flip_indices)
    except AssumptionError:
        f1 = None
    if f1 is None or f != f1:
        f1 = S0.encode([{0: ~e0}] + flip_indices)
        assert f == f1
    return f1
