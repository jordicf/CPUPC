# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"This module implements a class to represent STROPs (Single-Trunk Orthogonal Polygons)."

from __future__ import annotations
import copy
import portion as p
from rportion import RPolygon, rclosedopen
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional
from cpupc.geometry.fpolygon import RPoly2Box, XY_Box, FPolygon


class BranchDir(Enum):
    """Enum to represent the direction of a branch in a STROP."""

    N = 0
    S = 1
    E = 2
    W = 3

    def __repr__(self) -> str:
        return self.name


@dataclass
class StropBranches:
    """Class to represent the branches of a STROP. It contains the set of
    rectangles in each direction and the total area of the branches.
    The branches are disjoint and do not overlap with the trunk.
    The branches are sorted by the x-dimension (north/south) or y-dimension (east/west)."""

    _branches: dict[BranchDir, list[XY_Box]]
    _area: float

    def __init__(self) -> None:
        self._branches = {d: [] for d in BranchDir}
        self._area = -1.0

    def dup(self) -> StropBranches:
        """Returns a duplication of the StropBranches."""
        c = StropBranches()
        c._branches = copy.deepcopy(self._branches)
        c._area = self._area
        return c
    
    def __getitem__(self, direction: BranchDir) -> list[XY_Box]:
        """Returns the list of branches in the given direction."""
        return self._branches[direction]

    @property
    def num_branches(self) -> int:
        """Returns the total number of branches."""
        return sum(len(br) for br in self._branches.values())

    @property
    def area(self) -> float:
        """Returns the total area of the branches. If area is already computed, it returns the cached value."""
        if self._area < 0.0:
            self._area = sum(r.area for r, d in self.all_branches)
        return self._area

    @property
    def all_branches(self) -> Iterator[tuple[XY_Box, BranchDir]]:
        """Iterator over all branches in the decomposition, returning the branch and the direction."""
        for d in BranchDir:
            for b in self._branches[d]:
                yield (b, d)

    def add(self, r: XY_Box, direction: BranchDir) -> None:
        """Adds a rectangle to the list of branches, where direction
        indicates in which list (NSEW)."""
        assert direction in BranchDir, f"Invalid direction {direction} in add"
        self[direction].append(r)
        if direction in [BranchDir.N, BranchDir.S]:
            self[direction].sort(key=lambda b: b.xmin)
        else:
            self[direction].sort(key=lambda b: b.ymin)
        self._merge_branches(direction)

    def clear(self, direction: BranchDir) -> None:
        """Clears the list of branches in the given direction."""
        assert direction in BranchDir, f"Invalid direction {direction} in clear"
        self[direction].clear()
        self._area = -1.0  # Invalidate area

    def substitute(self, old_b1: XY_Box, old_b2: XY_Box, new_b: XY_Box, direction: BranchDir) -> None:
        """Replaces old1 and old2 with new in the list of branches."""
        assert direction in BranchDir, f"Invalid direction {direction} in substitute"
        self[direction].remove(old_b1)
        self[direction].remove(old_b2)
        self.add(new_b, direction)
        self._merge_branches(direction)

    def _merge_branches(self, direction: BranchDir) -> None:
        """Merges branches in the given direction that are adjacent and with the same width/height."""
        assert direction in BranchDir, (
            f"Invalid direction {direction} in _merge_branches"
        )
        self._area = -1.0  # Invalidate area
        br = self[direction]
        i = 0
        if direction in [BranchDir.N, BranchDir.S]:
            while i < len(br) - 1:
                assert br[i].xmax <= br[i + 1].xmin, (
                    "Branches in direction N/S must be disjoint"
                )
                if (
                    br[i].ymin == br[i + 1].ymin
                    and br[i].ymax == br[i + 1].ymax
                    and br[i].xmax == br[i + 1].xmin
                ):
                    # Merge the two branches
                    br[i] = XY_Box(br[i].xmin, br[i + 1].xmax, br[i].ymin, br[i].ymax)
                    del br[i + 1]
                else:
                    i += 1
        else:
            while i < len(br) - 1:
                assert br[i].ymax <= br[i + 1].ymin, (
                    "Branches in direction E/W must be disjoint"
                )
                if (
                    br[i].xmin == br[i + 1].xmin
                    and br[i].xmax == br[i + 1].xmax
                    and br[i].ymax == br[i + 1].ymin
                ):
                    # Merge the two branches
                    br[i] = XY_Box(br[i].xmin, br[i].xmax, br[i].ymin, br[i + 1].ymax)
                    del br[i + 1]
                else:
                    i += 1

    def __eq__(self, other: object) -> bool:
        """Checks equality of two StropBranches."""
        if not isinstance(other, StropBranches):
            return NotImplemented
        return all(self._branches[d] == other._branches[d] for d in BranchDir)

    def __hash__(self) -> int:
        """Returns a hash value for the StropBranches."""
        h = 0
        for b, d in self.all_branches:
            h ^= hash(b)
        return h


class Strop:
    """Class to represent the STROP of a polygon represented by a trunk
    (a rectangle) and a set of branches (rectangles) in the four directions.
    The branches are disjoint and do not overlap with the trunk.
    The class provides methods to reduce the number of branches by merging
    branches in a given direction. The class also provides a method to
    compute the Jaccard similarity of the STROP with respect to the original
    polygon."""

    _ref: FPolygon
    _polygon: Optional[FPolygon]
    _trunk: XY_Box
    _branches: StropBranches
    _similarity: float  # Jaccard similarity wrt the reference polygon

    def __init__(self, ref: FPolygon, trunk: Optional[XY_Box] = None) -> None:
        self._ref = ref
        if trunk:
            self._trunk = trunk
            self._branches = self._largest_strop_trunk(trunk)
        else:
            self._find_best_strop()
        self._invalidate()

    def _invalidate(self) -> None:
        """Invalidates the metrics."""
        self._similarity = -1.0
        self._polygon = None

    def __eq__(self, other: object) -> bool:
        """Checks equality of two STROPs"""
        if not isinstance(other, Strop):
            return NotImplemented
        return self._trunk == other._trunk and self._branches == other._branches

    def __hash__(self) -> int:
        """Returns a hash value for the STROP."""
        return hash(self._trunk) ^ hash(self._branches)

    def dup(self) -> Strop:
        """Returns a duplication of the Strop."""
        c = Strop(self._ref, self._trunk)
        c._branches = self._branches.dup()
        return c

    def __repr__(self) -> str:
        s = f"T:{self._trunk}"
        for b, d in self._branches.all_branches:
            s += f"|{d}:{b}"
        return s

    @property
    def trunk(self) -> XY_Box:
        """Returns the trunk of the decomposition."""
        return self._trunk

    @trunk.setter
    def trunk(self, r: XY_Box) -> None:
        """Sets the trunk of the decomposition."""
        self._trunk = r
        self._invalidate()

    @property
    def area(self) -> float:
        """Returns the area of the decomposition.
        If area is already computed, it returns the cached value."""
        return self.trunk.area + self._branches.area
    
    def area_branches(self, direction: BranchDir) -> float:
        """Returns the area of the branches in the given direction."""
        return sum(b.area for b in self._branches[direction])

    @property
    def similarity(self) -> float:
        """Returns the Jaccard similarity of the decomposition with
        respect to the original polygon."""
        if self._similarity < 0.0:
            self._similarity = round(self.reference.jaccard_similarity(self.polygon), 8)
        return self._similarity

    @property
    def num_branches(self) -> int:
        """Returns the number of branches in the decomposition."""
        return self._branches.num_branches

    def branches(self, type: Optional[BranchDir] = None) -> Iterator[XY_Box]:
        """Iterator over all branches.
        type indicates the type of rectangles to be returned (N, S, E, W, None for all)"""
        if type is None:
            for b, d in self._branches.all_branches:
                yield b
        else:
            for b in self._branches[type]:
                yield b

    def all_rectangles(self) -> Iterator[XY_Box]:
        """Iterator over all rectangles in the decomposition."""
        yield self._trunk
        for b, d in self._branches.all_branches:
            yield b

    @property
    def polygon(self) -> FPolygon:
        """Returns the polygon represented by the STROP decomposition."""
        if self._polygon is None:
            self._polygon = FPolygon(
                [self._trunk] + [b for b, d in self._branches.all_branches]
            )
        return self._polygon

    @property
    def reference(self) -> FPolygon:
        """Returns the original polygon represented by the STROP decomposition."""
        return self._ref

    def add_branch(self, r: XY_Box, direction: BranchDir) -> None:
        """Adds a branch to the decomposition."""
        self._branches.add(r, direction)
        self._invalidate()

    def _largest_strop_trunk(self, trunk: XY_Box) -> StropBranches:
        """Returns the branches of the largest strop included in self
        that has the associated trunk.
        """
        # Build the corners that must be subtracted
        ne = rclosedopen(trunk.xmax, p.inf, trunk.ymax, p.inf)
        nw = rclosedopen(-p.inf, trunk.xmin, trunk.ymax, p.inf)
        se = rclosedopen(trunk.xmax, p.inf, -p.inf, trunk.ymin)
        sw = rclosedopen(-p.inf, trunk.xmin, -p.inf, trunk.ymin)

        remainder = self.reference.rpolygon - trunk.rpolygon - ne - nw - se - sw

        # The remainder covers all branches.
        # Classify the branches according to the direction of the farthest edge to the trunk.
        # Add the coordinate of the farthest edge to the trunk
        branches = list[tuple[RPolygon, BranchDir, float]]()
        for r in remainder.maximal_rectangles():
            b = RPoly2Box(r)
            if not trunk.touch(b):
                continue

            if b.xmax > trunk.xmax:
                branches.append((r, BranchDir.E, b.xmax))
            elif b.xmin < trunk.xmin:
                branches.append((r, BranchDir.W, -b.xmin))
            elif b.ymax > trunk.ymax:
                branches.append((r, BranchDir.N, b.ymax))
            elif b.ymin < trunk.ymin:
                branches.append((r, BranchDir.S, -b.ymin))
            else:
                raise ValueError("Unexpected branch")

        # Compute a disjoint set of branches.
        # Sort the branches by distance of the farthest edge to the trunk.
        # Then visit the branches and remove those that overlap
        # with previous branches
        branches.sort(reverse=True, key=lambda r: r[2])
        prev_branches = RPolygon()
        strop_branches: StropBranches = StropBranches()

        for br, d, _ in branches:
            new_b = br - prev_branches
            if not new_b.empty:
                strop_branches.add(RPoly2Box(new_b), d)
            prev_branches |= br

        return strop_branches

    def _find_best_strop(self) -> None:
        """Finds the best strop included in self, by trying all possible trunks."""
        best_strop: Optional[Strop] = None
        for trunk in self.reference.maximal_rectangles():
            strop = Strop(self.reference, trunk)
            if (
                best_strop is None
                or strop.similarity > best_strop.similarity
                or (
                    strop.similarity == best_strop.similarity
                    and strop.num_branches < best_strop.num_branches
                )
            ):
                best_strop = strop
        assert best_strop is not None
        self.trunk = best_strop.trunk
        self._branches = best_strop._branches.dup()

    def reduce(self) -> Strop:
        """Reduces the number of branches of the STROP by reducing the
        branches in one of the directions. It selects the direction that
        maximizes the similarity with the original polygon.
        Returns a new Strop with the same original area."""
        assert self.num_branches > 0, "No branches to reduce"
        similarity = -1.0
        best_strop: Optional[Strop] = None
        for d in BranchDir:
            if len(self._branches[d]) > 0:
                new_strop = self.reduce_branches(d)
                if new_strop.similarity > similarity:
                    similarity = new_strop.similarity
                    best_strop = new_strop
        assert best_strop is not None
        return best_strop

    def reduce_branches(self, direction: BranchDir) -> Strop:
        """Reduces branches in the given direction by merging the
        closest pair of branches. In case only one branch remains, the branch
        is removed and the trunk is enlarged to keep the same area.
        Returns a new Strop with the same original area."""
        num_branches = len(self._branches[direction])
        assert num_branches > 0, f"No branches in direction {direction} to reduce"
        new_strop = self.dup()
        if len(new_strop._branches[direction]) == 1:
            tr = self.trunk
            xmin, xmax, ymin, ymax = tr.xmin, tr.xmax, tr.ymin, tr.ymax
            area = self.area_branches(direction)
            new_strop._branches.clear(direction)
            # Enlarge the trunk in the direction of the removed branches
            if direction == BranchDir.N:
                ymax += area / tr.width
            elif direction == BranchDir.S:
                ymin -= area / tr.width
            elif direction == BranchDir.E:
                xmax += area / tr.height
            else:  # direction == BranchDir.W
                xmin -= area / tr.height
            new_strop._trunk = XY_Box(xmin, xmax, ymin, ymax)
            return new_strop

        # Find the smallest branch in the direction
        b1 = min(new_strop._branches[direction], key=lambda b: b.area)
        # Find the closest branch to b1 in the direction
        b2 = min(
            (b for b in new_strop._branches[direction] if b != b1),
            key=lambda b: b1.distance(b),
        )
        # Merge b1 and b2
        new_b = b1.iso_area_absorb(b2)
        new_strop._branches.substitute(b1, b2, new_b, direction)
        new_strop._invalidate()
        return new_strop
