from copy import copy
from typing import assert_never, cast, overload

from flipper.kernel import Triangulation, Triangle, Corner, Encoding, Isometry, EdgeFlip

from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import CyclotomicField

from sage.matrix.constructor import matrix
from sage.matrix.special import (
    zero_matrix,
    identity_matrix,
    diagonal_matrix,
)

from sage.misc.misc_c import prod
from sage.functions.all import exp

from .util import EE, ScaledMatrix, make_standard_rep, matrix_tensor
from .mapping_torus import Immersion, MappingTorus
from .surface import make_CF_algebra, TeichRep


def corner_lookup(S: Triangulation, e: int) -> Corner:
    return S.corner_lookup[e]


class DotDecoration:
    """
    Implementing Ishibashi's "Cyclic quantum Teichmuller theory" conventions.
    """

    def __init__(
        self,
        surface: Triangulation,
        corner_indices: dict[Triangle, int] = None,
        triangle_order: list[Triangle] = None,
    ):
        if corner_indices is None:
            corner_indices = {t: 0 for t in surface.triangles}
        if triangle_order is None:
            triangle_order = copy(surface.triangles)
        key = lambda t: [e.label for e in t]
        if not (
            surface.triangles
            == sorted(triangle_order, key=key)
            == sorted(corner_indices, key=key)
        ):
            raise ValueError("some triangles are not in surface")
        self.surface = surface
        self.corner_indices = corner_indices
        self.triangle_order = triangle_order

    def __repr__(self):
        res = [f"{t}: {self.corner_indices[t]}" for t in self.triangle_order]
        return "[" + ", ".join(res) + "]"

    def __copy__(self):
        return DotDecoration(self.surface, copy(self.corner_indices))

    def __or__(self, other: dict[Triangle, int]):
        if not all(x in self.corner_indices for x in other):
            raise ValueError("some new triangles not in surface")
        return DotDecoration(
            self.surface, self.corner_indices | other, self.triangle_order
        )

    def corner(self, t: Triangle):
        return t.corners[self.corner_indices[t]]

    def triangle_index(self, t: Triangle | Corner):
        if isinstance(t, Corner):
            t = t.triangle
        return self.triangle_order.index(t)

    @property
    def num_triangles(self):
        return self.surface.num_triangles

    def diff_with_dot(self, c: Corner):
        return (cast(int, c.side) - self.corner_indices[c.triangle]) % 3


class DotRotPerm:
    def __init__(self, source_dots: DotDecoration, target_dots: DotDecoration):
        if source_dots.surface != target_dots.surface:
            raise ValueError("dot decorations do not live on the same surface")
        self.source_dots = source_dots
        self.target_dots = target_dots
        self.rotations = [
            (target_dots.corner_indices[t] - source_dots.corner_indices[t]) % 3
            for t in source_dots.triangle_order
        ]
        self.permutation = [
            source_dots.triangle_order.index(t) for t in target_dots.triangle_order
        ]
        """i-th triangle in target_dots is permutation[i]-th triangle in source_dots."""

    def has_rotation(self):
        return any(x != 0 for x in self.rotations)

    def rotation_count(self):
        return sum(x != 0 for x in self.rotations)

    def has_permutation(self):
        return any(self.permutation[i] != i for i in range(len(self.permutation)))

    def __bool__(self):
        return self.has_rotation() or self.has_permutation()

    def __repr__(self):
        res = []
        if self.has_permutation():
            res.append(f"Triangle permutation {self.permutation}")
        if self.has_rotation():
            res.append(f"Rotations {self.rotations}")
        return " * ".join(res) if res else "Identity"


class DottedIsometry(Isometry):
    def __init__(
        self,
        source_triangulation,
        target_triangulation,
        label_map,
        source_dots: DotDecoration,
        target_dots: DotDecoration,
    ):
        if (
            source_dots.surface is not source_triangulation
            or target_dots.surface is not target_triangulation
        ):
            raise ValueError("dot decorations do not match surfaces")
        self.source_dots = source_dots
        self.target_dots = target_dots
        super().__init__(source_triangulation, target_triangulation, label_map)

    @staticmethod
    def enhance(f: Isometry, prev: DotDecoration):
        S0: Triangulation = f.source_triangulation
        S1: Triangulation = f.target_triangulation
        next_indices = dict()
        next_order = list()
        for triangle in prev.triangle_order:
            corner = prev.corner(triangle)
            next_corner = corner_lookup(S1, f.label_map[corner.label])
            next_triangle: Triangle = next_corner.triangle
            next_indices[next_triangle] = next_corner.side
            next_order.append(next_triangle)
        next = DotDecoration(S1, next_indices, next_order)
        return DottedIsometry(
            f.source_triangulation, f.target_triangulation, f.label_map, prev, next
        )


def rotate_corner(S: Triangulation, c: int | Corner, direction=1) -> Corner:
    if not isinstance(c, Corner):
        c = corner_lookup(S, c)
    return corner_lookup(S, c.labels[direction])


class DottedEdgeFlip(EdgeFlip):
    def __init__(
        self,
        source_triangulation,
        target_triangulation,
        edge_label,
        source_dots: DotDecoration,
        target_dots: DotDecoration,
    ):
        if (
            source_dots.surface is not source_triangulation
            or target_dots.surface is not target_triangulation
        ):
            raise ValueError("dot decorations do not match surfaces")
        self.source_dots = source_dots
        self.target_dots = target_dots
        self.v_index = source_dots.triangle_index(
            corner_lookup(source_triangulation, edge_label)
        )
        self.w_index = source_dots.triangle_index(
            corner_lookup(source_triangulation, ~edge_label)
        )
        super().__init__(source_triangulation, target_triangulation, edge_label)

    @staticmethod
    def enhance(f: EdgeFlip, prev: DotDecoration):
        S0: Triangulation = f.source_triangulation
        S1: Triangulation = f.target_triangulation
        r"""
        +---+            +---+
        |+w |   T_{vw}   |v +|
        |v\.|  ------->  |./w|
        |. \|            |/ .|
        +---+            +---+
        """
        e = cast(int, f.edge_label)
        cv = corner_lookup(S0, e)
        cw = rotate_corner(S0, ~e, -1)
        prev |= {c.triangle: cast(int, c.side) for c in (cv, cw)}
        vix = prev.triangle_index(cv)
        wix = prev.triangle_index(cw)
        cv = rotate_corner(S1, e)
        cw = corner_lookup(S1, ~e)
        next_order = copy(prev.triangle_order)
        next_order[vix] = cv.triangle
        next_order[wix] = cw.triangle
        next = DotDecoration(
            S1,
            {t: i for t, i in prev.corner_indices.items() if t in S1}
            | {c.triangle: cast(int, c.side) for c in (cv, cw)},
            next_order,
        )
        return DottedEdgeFlip(
            f.source_triangulation, f.target_triangulation, f.edge_label, prev, next
        )


class DottedEncoding(Encoding):
    def __init__(self, sequence: list[DottedEdgeFlip | DottedIsometry]):
        self.rot_perms = [
            DotRotPerm(f0.target_dots, f1.source_dots)
            for f0, f1 in zip(sequence, [sequence[-1]] + sequence[:-1])
        ]
        super().__init__(sequence)

    def __getitem__(self, key: int) -> DottedEdgeFlip | DottedIsometry:
        return super().__getitem__(key)

    @staticmethod
    def enhance(f: Encoding, prev: DotDecoration):
        sequence = []
        for move in reversed(f.sequence):
            emove = dot_enhance(move, prev)
            sequence.append(emove)
            prev = emove.target_dots
        sequence.reverse()
        return DottedEncoding(sequence)

    @property
    def source_dots(self) -> DotDecoration:
        return self.sequence[-1].source_dots

    @property
    def target_dots(self) -> DotDecoration:
        return self.sequence[0].target_dots

    def info(self):
        print(self.source_dots)
        for mv, rp in zip(reversed(self.sequence), reversed(self.rot_perms)):
            print(mv)
            print(mv.target_dots)
            if rp:
                print(rp)
                print(rp.target_dots)


@overload
def dot_enhance(f: EdgeFlip, prev: DotDecoration = None) -> DottedEdgeFlip: ...
@overload
def dot_enhance(f: Isometry, prev: DotDecoration = None) -> DottedIsometry: ...
@overload
def dot_enhance(f: Encoding, prev: DotDecoration = None) -> DottedEncoding: ...
def dot_enhance(f: EdgeFlip | Isometry | Encoding, prev: DotDecoration = None):
    if prev is None:
        prev = DotDecoration(f.source_triangulation)
    if isinstance(f, EdgeFlip):
        return DottedEdgeFlip.enhance(f, prev)
    elif isinstance(f, Isometry):
        return DottedIsometry.enhance(f, prev)
    elif isinstance(f, Encoding):
        return DottedEncoding.enhance(f, prev)
    assert_never(f)


def local_gen_exp(dots: DotDecoration, e: int):
    res = [0] * (2 * dots.num_triangles)
    for i in (e, ~e):
        c = corner_lookup(dots.surface, i)
        ix = dots.triangle_index(c)
        diff = dots.diff_with_dot(c)
        if diff == 0:
            res[2 * ix] += 1
            res[2 * ix + 1] += -1
        elif diff == 1:
            res[2 * ix + 1] += 1
        else:
            res[2 * ix] += -1
    return res


class Fermat:
    def __init__(self, ratio, n):
        # TODO: it seems it's better to do
        # plus  = (1 + Y^-1)^(-1/n) and
        # minus = (1 + Y)^(-1/n)
        # so that ratio^n == Y
        self.n = n
        self.ratio = ratio
        self.minus = (1 + ratio**n) ** (-1 / n)
        self.plus = self.ratio * self.minus

    def check(self):
        return abs(self.plus**self.n + self.minus**self.n - 1)

    def __repr__(self):
        return f"{self.ratio.n()} == {self.plus.n()} / {self.minus.n()}"


def make_local_rep(imm: Immersion, dots: DotDecoration, x):
    """
    Note CF algebra uses BWY orientation, which is opposite of Ishibashi.
    Therefore, we need to adjust `q` to `q^-1`.
    """
    # copied from make_CF_rep
    S = imm.surface
    if dots.surface != S:
        raise ValueError("decorations do not match surface")
    R = make_CF_algebra(S)
    n = (2 * x).denominator()
    sympl_rank = S.num_triangles
    D = n**sympl_rank
    order, sympl_gens = make_standard_rep([1] * sympl_rank, -x)
    # convention: sympl_gens is ordered U_0, P_0, U_1, ...
    scaled_gens = []
    q = EE(-x, CyclotomicField(x.denominator()))
    for i in range(R.ngens()):
        c = local_gen_exp(dots, i)
        # TODO: take powers before tensor product
        # especially because pow does not preserve sparseness
        raw_gen = prod((A**k).sparse_matrix() for A, k in zip(sympl_gens, c))
        weyl = -sum(c[2 * i] * c[2 * i + 1] for i in range(sympl_rank))
        im_scale = exp(imm.shear_bend[i] / n)
        scaled_gens.append(ScaledMatrix(q**weyl * raw_gen, im_scale))
    return TeichRep(R, D, scaled_gens, x, order, sympl_basis=None)


def _dot_rot_block(rot: int, n: int, q):
    """
    Block for rotation in a single triangle.
    Note we use `q^-1` in Ishibashi's formula.
    Return has abs(det) == sqrt(n) if the rotation is nontrivial, otherwise it is identity.
    """
    if rot == 0:
        return identity_matrix(n)
    assert rot == 1 or rot == 2
    A = matrix(n, n, lambda m, k: q ** (m**2 + 2 * k * m))
    return A if rot == 1 else A.conjugate_transpose()


def dot_rot_intertwiner(rho0: TeichRep, f: DotRotPerm):
    """
    Suppose `f: L0 -> L1` is a dot rotation and `rho0: Teich(L0) -> End(V)` is a local rep.
    Return `H` such that `rho1(x) == H^{-1} rho0(f^*(x)) H` for all `x` in `Teich(L1)`.
    """
    order = rho0.order
    q = rho0.q
    D = rho0.dimension
    E = f.source_dots.num_triangles
    assert D == order**E
    # assuming ord(q) is odd, in which case ord(q) == order
    H = 1
    if f.has_rotation():
        rot_blocks = [_dot_rot_block(r, order, q) for r in f.rotations]
        H = matrix_tensor(rot_blocks)
    if f.has_permutation():
        # If rho1(x) == matrix_tensor(A_0, A_1, ...),
        # then rho0(f^*(x)) == matrix_tensor(A_perm[0], ...).
        # Therefore, H_{i,j} = delta_{i_perm[0], j_0}...
        # Recall matrix_tensor(A_0, ...)_{0,1} == ... * (A_{-1})_{0,1} and so on
        perm_entries = dict()
        for i in range(D):
            i_list = Integer(i).digits(base=order, padto=E)
            # need to reverse because digits is little-endian
            i_list.reverse()
            # logically the next step is the delta and reverse again to fix endianness
            # j_list = [i_list[k] for k in f.permutation]
            # j_list.reverse()
            j_list = [i_list[k] for k in reversed(f.permutation)]
            j = ZZ(j_list, base=order)
            perm_entries[(i, j)] = 1
        H *= matrix(D, D, perm_entries)
    return H


def cyclic_dilog(p: Fermat, x, q):
    """Literally Ishibashi's formula (q not inverted)"""
    # TODO: not very careful about branches. need to check
    return prod(
        (p.minus + q ** (2 * j + 1) * p.plus * x) ** (j / p.n) for j in range(1, p.n)
    )


def local_flip_intertwiner(rho0: TeichRep, f: DottedEdgeFlip):
    """
    Suppose `f: L0 -> L1` is a dotted flip and `rho0: Teich(L0) -> End(V)` is a local rep.
    Return `H` such that `rho1(x) == H^{-1} rho0(f^*(x)) H` for all `x` in `Teich(L1)`.
    """
    e = cast(int, f.edge_index)
    Xe = rho0.scaled_gens[e]
    v = f.v_index
    w = f.w_index
    # first we make S_{vw}
    sympl_rank = f.source_triangulation.num_triangles
    order, sympl_gens = make_standard_rep([1] * sympl_rank, -rho0.x)
    # convention: symple_gens is ordered U_0, P_0, U_1, ...
    assert rho0.dimension == order**sympl_rank
    Uw = sympl_gens[2 * w]
    Pv = sympl_gens[2 * v + 1]
    S_vw = zero_matrix(rho0.dimension)
    for i in range(order):
        for j in range(order):
            S_vw += rho0.q ** (-2 * i * j) * Uw**i * Pv**j
    # next we make cyclic_dilog(Xe_bar)
    # ~P * Xe.matrix * P == D
    D, P = Xe.matrix.diagonalization()
    p = Fermat(Xe.scale, order)
    D = diagonal_matrix(cyclic_dilog(p, x, 1 / rho0.q) for x in D.diagonal())
    F = P * D * ~P
    return F * S_vw / order


def make_bundle_local_rep(m: MappingTorus, x) -> list[TeichRep]:
    """
    Return a list of `TeichRep` of the CF algebras using data from `m: MappingTorus`.
    Entries correspond to `m.immersions`.
    """
    f = m.mapping_class
    if not isinstance(f, DottedEncoding):
        f = dot_enhance(f)
    if not isinstance(f[-1], EdgeFlip):
        raise NotImplementedError("m.mapping_class must start with a flip")
    l = len(f)
    res = []
    # since each step uses the target, the list starts at index 1
    for i in range(l):
        move = f[~i]
        target_i = (i + 1) % l
        imm = m.immersions[target_i]
        res.append(make_local_rep(imm, move.target_dots, x))
        rp = f.rot_perms[~i]
        res.append(make_local_rep(imm, rp.target_dots, x))
    # after construction, we rotate the list so index start at 0
    return [res[-1]] + res[:-1]


def combine_Hs(Hs: list, x, f):
    total_rots = Integer(sum(rp.rotation_count() for rp in f.rot_perms))
    n = (2 * x).denominator()
    H = prod(Hs)
    return H / H.base_ring()(n ** (total_rots / 2))


def AK(m: MappingTorus, x, *, rho=None, separate=False):
    """Return BWY intertwiner `H` normalized by `det(H) == 1`."""
    f = m.mapping_class
    if not isinstance(f, DottedEncoding):
        f = dot_enhance(f)
    if rho is None:
        rho = make_bundle_local_rep(m, x)
    Hs = []
    for i in range(len(f)):
        if isinstance(move := f[~i], DottedEdgeFlip):
            Hs.append(local_flip_intertwiner(rho[2 * i], move))
        else:
            Hs.append(1)
        if rp := f.rot_perms[~i]:
            Hs.append(dot_rot_intertwiner(rho[2 * i + 1], rp))
        else:
            Hs.append(1)
    return Hs if separate else combine_Hs(Hs, x, f)


def test_perm(surface: str, mapping_class: str):
    import flipper
    from .util import normalize_mapping_class
    S = flipper.load(surface)
    f = normalize_mapping_class(S.mapping_class(mapping_class))
    g = dot_enhance(f)
    m = MappingTorus(g)
    x = ~Integer(3)
    rho3 = make_bundle_local_rep(m, x)
    rp = g.rot_perms[0]
    mid = DotDecoration(rp.source_dots.surface, rp.target_dots.corner_indices, rp.source_dots.triangle_order)
    rho_mid = make_local_rep(m.immersions[0], mid, x)
    rot = DotRotPerm(rp.source_dots, mid)
    perm = DotRotPerm(mid, rp.target_dots)
    assert not rot.has_permutation() and not perm.has_rotation()
    Hrot = dot_rot_intertwiner(rho3[-1], rot)
    rhoC = rho3[-1].conjugation(Hrot)
    print(rhoC.scaled_gens == rho_mid.scaled_gens)
    Hperm = dot_rot_intertwiner(rhoC, perm)
    rho1 = rho3[0]
    rhoC = rho_mid.conjugation(Hperm)
    print(rhoC.scaled_gens == rho1.scaled_gens)
    rhoC = rho_mid.conjugation(~Hperm)
    print(rhoC.scaled_gens == rho1.scaled_gens)