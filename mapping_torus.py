from typing import cast

from flipper.kernel import (
    Encoding,
    Move,
    Isometry,
    EdgeFlip,
    Triangulation,
    Triangle,
    AssumptionError,
)

from snappy import Manifold
from snappy.snap.t3mlite import Mcomplex, Tetrahedron, Perm4
from snappy.snap.t3mlite.simplex import ZeroSubsimplices, TwoSubsimplices
from snappy.snap.t3mlite.simplex import E01, E02, E21, E03, E13, E32
from snappy.snap.t3mlite.arrow import Arrow

from sage.structure.element import Matrix
from sage.matrix.constructor import matrix
from sage.matrix.special import block_matrix
from sage.modules.free_module_element import vector
from sage.misc.functional import round
from sage.functions.all import log
from sage.symbolic.constants import pi, I

_piI = pi * I


def _side_index_in_triangle(t: Triangle, edge_label: int):
    """Find the index of `e`-labeled edge in `t`. Assumes that `e` is in `t`."""
    return t.labels.index(edge_label)


def _get_left_triangle(S: Triangulation, edge_label: int) -> Triangle:
    return S.triangle_lookup[edge_label]


def _move_first_to_index(t: list[int], i: int):
    """
    Given [a0,a1,a2,f] representing a Perm4 which is part of a face embedding,
    rotate the first 3 entries so that ai is in the 0-th slot, and f is fixed.
    """
    return Perm4(t[3 - i : 3] + t[: 3 - i] + [t[3]])


def _other_side_face_imm(tet: Tetrahedron, p: Perm4):
    """
    Given the embedding of a triangle defined by Tetrahedron and Perm4 pair `(tet, p)`,
    return the other representation of the embedding using the face pairing.
    """
    face = TwoSubsimplices[p[3]]
    other_tet = cast(dict[int, Tetrahedron], tet.Neighbor)[face]
    return other_tet, cast(dict[int, Perm4], tet.Gluing)[face] * p


def _make_flip_perms(flip_index: int, n0: Triangle, n1: Triangle, tet: Tetrahedron):
    assert flip_index >= 0
    j0 = _side_index_in_triangle(n0, flip_index)
    j1 = _side_index_in_triangle(n1, ~flip_index)
    r"""
    #      ^ 2               ^ 2
    #   d / \ c             /|\
    #    /f1 \             / ^ \
    # 3 <-<--> 1  ==>   3 <n0|n1> 1
    #    \f0 /             \ | /
    #   a \ / b             \|/
    #    0 v               0 v
    """
    return {
        n0: _other_side_face_imm(tet, _move_first_to_index([3, 0, 2, 1], j0)),
        n1: _other_side_face_imm(tet, _move_first_to_index([1, 2, 0, 3], j1)),
    }


def _apply_isometry(isom: Isometry, t: Triangle, p: Perm4):
    """Return the image of the Triangle `t` under the Isometry `isom` by the label map."""
    image_label = [isom.label_map[e.label] for e in t]
    next_t = Triangle([isom.target_triangulation.edge_lookup[l] for l in image_label])
    # Triangle constructor rotates the labels
    # suppose t: p is (abc) |-> (ijkl), isom(abc) = (a'b'c')
    # then new pair should be (a'b'c') |-> (ijkl)
    # but Triangle(a'b'c') = (uvw), so we need to find the index of a' in (uvw) and move first rto index
    j = _side_index_in_triangle(next_t, image_label[0])
    return next_t, _move_first_to_index(list(p.tuple()), j)


def _col_index(tetra: int, component: int):
    return 3 * tetra + component % 3


def makeNZ(gluing: Matrix):
    N = gluing.ncols() // 3
    r = gluing.nrows()
    gauge = [2] * N
    preB = matrix(r, N, lambda i, j: gluing[i, _col_index(j, gauge[j])])
    A = matrix(r, N, lambda i, j: gluing[i, _col_index(j, gauge[j] - 1)]) - preB
    B = matrix(r, N, lambda i, j: gluing[i, _col_index(j, gauge[j] + 1)]) - preB
    nu = vector(2 if i < N else 0 for i in range(r)) - sum(preB.columns())
    return A, B, nu


class LogShapes:
    def __init__(self, gluing: Matrix, shapes: list[complex]):
        F = shapes[0].parent()  # type: ignore
        piI = F(_piI)
        A, B, nu = makeNZ(gluing)
        N = len(shapes)
        z0 = vector(F, (log(z) for z in shapes))
        z1 = vector(F, (-log(1 - z) for z in shapes))
        log_error = (A * z1 + B * z0) / piI - nu
        rounded_error = vector(round(x.real()) for x in log_error)
        delta = block_matrix((A, B), nrows=1, subdivide=False).solve_right(
            rounded_error
        )
        z1 -= delta[:N] * piI
        z0 -= delta[N:] * piI
        log_error = (A * z1 + B * z0) / piI - nu
        rounded_error = vector(round(x.real()) for x in log_error)
        if rounded_error != 0:
            raise ValueError("Cannot find log branches for gluing equations.")
        self.rounding_error = (log_error - rounded_error).norm(1)
        self.data = (z0, z1, vector(F, (piI - a - b for a, b in zip(z0, z1))))
        self._base_ring = F

    def __getitem__(self, key):
        tetra, E = key
        if E == E01 or E == E32:
            component = 0
        elif E == E02 or E == E13:
            component = 1
        elif E == E03 or E == E21:
            component = 2
        else:
            raise ValueError(f"Simplex E = {E} is not an edge.")
        return self.data[component][tetra]


def _snappy_triangle_map(imm: dict[Triangle, tuple], mcx: Mcomplex):
    """Take flipper Immersion face map and turn keys and values into SnapPy objects."""
    return {
        f: (mcx.Tetrahedra[t.label], Perm4(p.permutation)) for f, (t, p) in imm.items()
    }


class Immersion:
    def __init__(
        self,
        surface: Triangulation,
        mcx: Mcomplex,
        triangle_map: dict[Triangle, tuple[Tetrahedron, Perm4]],
        log_shapes: LogShapes,
    ):
        self.surface = surface
        self.complex = mcx
        self.triangle_map = triangle_map
        self.log_shapes = log_shapes
        self.shear_bend = [self._shear_bend_at(i) for i in range(surface.zeta)]

    def __repr__(self):
        return repr(self.triangle_map)

    def __eq__(self, other):
        if not isinstance(other, Immersion):
            return False
        # Perm4 does not have custom __eq__, which means it compares by identity
        # so default comparision of dict[...] is not what we want
        keys = self.triangle_map.keys()
        return (
            self.surface == other.surface
            and self.complex is other.complex
            and keys == other.triangle_map.keys()
            and all(self.triangle_map[k][0] is other.triangle_map[k][0] for k in keys)
            and all(
                self.triangle_map[k][1].tuple() == other.triangle_map[k][1].tuple()
                for k in keys
            )
        )

    def _get_arrow(self, edge_label: int, use_surface_orientation=True):
        """
        Get an Arrow around the edge in the immersed surface.
        edge_label chooses which triangle the arrow is in.
        Recalll (I assume that) surface orientation is into the tetrahedron.
        """
        f = _get_left_triangle(self.surface, edge_label)
        i = _side_index_in_triangle(f, edge_label)
        T, p = self.triangle_map[f]
        face = TwoSubsimplices[p[3]]
        edge = face ^ ZeroSubsimplices[p[i]]
        arrow = Arrow(edge, face, T)
        if use_surface_orientation:
            return arrow.reverse()
        else:
            return arrow

    def _shear_bend_at(self, edge_index: int):
        edge_index = int(edge_index)
        assert edge_index >= 0
        start = self._get_arrow(edge_index)
        end = self._get_arrow(~edge_index, False).next()
        assert end is not None
        assert start.Tetrahedron.Class[start.Edge] is end.Tetrahedron.Class[end.Edge]
        res = 0
        while start != end:
            res += self.log_shapes[start.Tetrahedron.Index, start.Edge]
            start.next()
        return res - self.log_shapes._base_ring(_piI)

    def _get_flip(self, move: EdgeFlip):
        assert self.surface is move.source_triangulation
        next_map = {
            f: v for f, v in self.triangle_map.items() if f in move.target_triangulation
        }
        flip_index = move.edge_index
        f0 = _get_left_triangle(self.surface, flip_index)
        # f1 = self._get_left_triangle(~flip_index)
        n0 = _get_left_triangle(move.target_triangulation, flip_index)
        n1 = _get_left_triangle(move.target_triangulation, ~flip_index)
        # TODO: this assumes that the normal to the surface always points into the tetra selected by flipper
        tet = self.triangle_map[f0][0]
        next_map |= _make_flip_perms(flip_index, n0, n1, tet)
        return Immersion(
            move.target_triangulation, self.complex, next_map, self.log_shapes
        )

    def _get_isom(self, move: Isometry):
        assert self.surface is move.source_triangulation
        next_map = dict()
        for t, v in self.triangle_map.items():
            next_t, next_p = _apply_isometry(move, t, v[1])
            next_map[next_t] = v[0], next_p
        # print(self)
        # print(next_map)
        return Immersion(
            move.target_triangulation, self.complex, next_map, self.log_shapes
        )

    def get_move(self, move: Move):
        if isinstance(move, EdgeFlip):
            return self._get_flip(move)
        elif isinstance(move, Isometry):
            return self._get_isom(move)
        else:
            raise TypeError("Unrecognized move type")


class MappingTorus:
    def __init__(
        self,
        mapping_class: Encoding,
        *,
        snappy_variant=Manifold,
        check=True,
        find_field_args=None,
        prec=None,
    ):
        B = mapping_class.bundle(False, False)
        M = snappy_variant(B)
        T = Mcomplex(M)
        if check and not M.is_isometric_to(snappy_variant(mapping_class.bundle())):
            raise ValueError(
                "Cannot find layered triangulation that implements the sequence in the given mapping class encoding"
            )
        if find_field_args is None:
            shapes = M.tetrahedra_shapes("rect")
        else:
            find_result = M.tetrahedra_field_gens().find_field(*find_field_args)
            if find_result is None:
                raise ValueError(f"Cannot find field with arguments {find_field_args}")
            shapes = find_result[2]
            shapes = [z.n(prec) for z in shapes]
        self.surface = cast(Triangulation, mapping_class.source_triangulation)
        self.mapping_class = mapping_class
        self.bundle = B
        self.manifold = M
        self.complex = T
        self.log_shapes = LogShapes(M.gluing_equations(), shapes)
        tri_map = _snappy_triangle_map(B.immersion, T)
        immersions = [Immersion(self.surface, T, tri_map, self.log_shapes)]
        for f in reversed(mapping_class):
            immersions.append(immersions[-1].get_move(f))
        assert immersions[0] == immersions[-1]
        self.immersions = immersions[:-1]
        self.filled = M.copy()
        self.filled.dehn_fill(B.fibre_slopes())


def _find_isometry_noexcept(
    src: Triangulation, tar: Triangulation, e: int
) -> Isometry | None:
    try:
        return src.find_isometry(tar, {0: e})
    except AssumptionError:
        return None


def sequence_isoms(L: Encoding | list[Triangulation]):
    """
    :param Encoding | list[Triangulation] L: A mapping class as `L: Encoding` or `L = [f.source_triangulation for f in g]` for a mapping class `g`
    :return list[tuple[int, Isometry]]:\n
    `all(i[1].source_triangulation is L[i[0]] for i in sequence_isoms(L))`\n
    `all(sequence_isoms(L)[i][1].target_triangulation is L[i] for i in range(len(L)))`
    """
    if isinstance(L, Encoding):
        L = [f.source_triangulation for f in L]
    sigs = [f.iso_sig() for f in reversed(L)]
    isoms: list[tuple[int, Isometry]] = []
    for l in L:
        i = ~sigs.index(l.iso_sig())
        src = L[i]
        h = l.id_isometry() if src is l else None
        for e in l.edges:
            if h is not None:
                break
            h = _find_isometry_noexcept(src, l, e.label)
        assert h is not None
        isoms.append((i, h))
    return isoms
