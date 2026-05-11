# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"This module implements a polygon class based on the rportion library."

from __future__ import annotations
import copy
from typing import Iterable, Iterator, Optional
from dataclasses import dataclass
from rportion import RPolygon, rclosedopen

# Some auxiliary types

RPoint = tuple[float, float]

# A list of 2D vertices to represent a polygon without holes.
# The edges are represented by segments between consecutive vertices.
# The polygon is assumed to be closed (i.e., the last vertex is connected
# to the first vertex).
Vertices = list[RPoint]


# Tuple to represent a rectangle in the interval [xmin, xmax, ymin, ymax]
# It is an immutable class with frozen=True and slots=True to save memory and improve performance.
# It is also hashable and comparable for equality, so it can be used as a key in dictionaries and sets.
@dataclass(frozen=True, slots=True)
class XY_Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def rpolygon(self) -> RPolygon:
        """Returns the rectangle represented as an RPolygon."""
        return rclosedopen(self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def center(self) -> RPoint:
        """Returns the center of the box."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        """Returns the width of the box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Returns the height of the box."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Returns the area of the box."""
        return self.width * self.height

    @property
    def is_vertical_edge(self) -> bool:
        """Returns True if the box is a vertical edge (width == 0)."""
        return self.xmax == self.xmin

    @property
    def is_horizontal_edge(self) -> bool:
        """Returns True if the box is a horizontal edge (height == 0)."""
        return self.ymax == self.ymin

    @property
    def is_edge(self) -> bool:
        """Returns True if the box is an edge (width == 0 or height == 0)."""
        return self.is_vertical_edge or self.is_horizontal_edge

    def dup(self) -> XY_Box:
        """Returns a duplication of the box."""
        return XY_Box(self.xmin, self.xmax, self.ymin, self.ymax)

    def __repr__(self) -> str:
        return f"[x=({self.xmin},{self.xmax}),y=({self.ymin},{self.ymax})]"

    def __hash__(self) -> int:
        return hash((self.xmin, self.xmax, self.ymin, self.ymax))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, XY_Box):
            return NotImplemented
        return (
            self.xmin == other.xmin
            and self.xmax == other.xmax
            and self.ymin == other.ymin
            and self.ymax == other.ymax
        )

    def distance(self, other: XY_Box) -> float:
        """Returns the Manhattan distance between the two rectangles."""
        dx = max(0.0, other.xmin - self.xmax, self.xmin - other.xmax)
        dy = max(0.0, other.ymin - self.ymax, self.ymin - other.ymax)
        return dx + dy

    def intersects(self, other: XY_Box) -> bool:
        """Returns True if the two rectangles intersect (i.e., they have a non-empty intersection)."""
        return (
            self.xmin < other.xmax
            and self.xmax > other.xmin
            and self.ymin < other.ymax
            and self.ymax > other.ymin
        )

    def __and__(self, other: XY_Box) -> Optional[XY_Box]:
        """Returns the intersection of the two rectangles, or None if they do not intersect."""
        if not self.intersects(other):
            return None
        return XY_Box(
            max(self.xmin, other.xmin),
            min(self.xmax, other.xmax),
            max(self.ymin, other.ymin),
            min(self.ymax, other.ymax),
        )

    def bbox(self, other: XY_Box) -> XY_Box:
        """Returns the bounding box of the two rectangles."""
        return XY_Box(
            min(self.xmin, other.xmin),
            max(self.xmax, other.xmax),
            min(self.ymin, other.ymin),
            max(self.ymax, other.ymax),
        )

    def touch(self, other: XY_Box) -> bool:
        """Returns True if the two rectangles touch each other (i.e., they have
        at least one point in common)."""
        return (
            self.xmin <= other.xmax
            and self.xmax >= other.xmin
            and self.ymin <= other.ymax
            and self.ymax >= other.ymin
        )

    def iso_area_absorb(self, other: XY_Box) -> XY_Box:
        """Returns a rectangle with the same area as the sum of the two
        rectangles. The rectangle is aligned with the common edge of the two rectangles,
        corresponding to the direction (ymin for N, ymax for S, xmin for E, xmax for W).
        The biggest rectangle absorbs the smallest one by enlarging the corresponding edge
        without exceeding the boundary of the other rectangle.
        The two rectangles must be disjoint (i.e., they cannot intersect, but they can touch)."""

        inter = self & other
        assert inter is None or inter.area == 0.0, (
            "The rectangles must be disjoint to be absorbed"
        )

        location = "*"
        if self.ymin == other.ymin:
            location = "N"
        elif self.ymax == other.ymax:
            location = "S"
        elif self.xmin == other.xmin:
            location = "E"
        elif self.xmax == other.xmax:
            location = "W"

        assert location in "NSEW", "Invalid location in iso_area_merge"

        # This function is tedious, since it has many cases (NSEW, left/right/top/bottom)
        # pprint(f"Absorbing {self} and {other} in direction {direction}")

        big, small = (self, other) if self.area >= other.area else (other, self)
        xmin, xmax, ymin, ymax = big.xmin, big.xmax, big.ymin, big.ymax
        if location in "NS":
            left = big.xmin < small.xmin
            if left:  # big is to the left of small
                xmax += small.area / big.height
                if xmax > small.xmax:
                    inc_area = (xmax - small.xmax) * ymax - ymin
                    xmax = small.xmax
                    if location == "N":
                        ymax += inc_area / (xmax - xmin)
                    else:  # location == "S"
                        ymin -= inc_area / (xmax - xmin)
            else:  # big is to the right of small
                xmin -= small.area / big.height
                if xmin < small.xmin:
                    inc_area = (small.xmin - xmin) * ymax - ymin
                    xmin = small.xmin
                    if location == "N":
                        ymax += inc_area / (xmax - xmin)
                    else:  # location == "S"
                        ymin -= inc_area / (xmax - xmin)
        else:  # location in "EW"
            top = big.ymin > small.ymin
            if top:  # big is above small
                ymin -= small.area / big.width
                if ymin < small.ymin:
                    inc_area = (small.ymin - ymin) * ymax - ymin
                    ymin = small.ymin
                    if location == "E":
                        xmax += inc_area / (xmax - xmin)
                    else:  # location == "W"
                        xmin -= inc_area / (xmax - xmin)
            else:  # big is below small
                ymax += small.area / big.width
                if ymax > small.ymax:
                    inc_area = (ymax - small.ymax) * (xmax - xmin)
                    ymax = small.ymax
                    if location == "E":
                        xmax += inc_area / (xmax - xmin)
                    else:  # location == "W"
                        xmin -= inc_area / (xmax - xmin)
        # pprint(f"Result: {new_b}")

        return XY_Box(xmin, xmax, ymin, ymax)


# Some methods to extend the functionality of rportion


def RPoly2Box(r: RPolygon) -> XY_Box:
    """Converts a rectangle in RPolygon format to XY_Box."""
    x_interval = r.x_enclosure_interval
    xmin, xmax = x_interval.lower, x_interval.upper
    y_interval = r.y_enclosure_interval
    ymin, ymax = y_interval.lower, y_interval.upper
    assert (
        isinstance(xmin, (float, int))
        and isinstance(xmax, (float, int))
        and isinstance(ymin, (float, int))
        and isinstance(ymax, (float, int))
    )
    return XY_Box(float(xmin), float(xmax), float(ymin), float(ymax))


class FPolygon:
    """Class for polygons based on the rportion library.
    The polygon is represented as a union of rectangles with
    closed-open intervals."""

    _polygon: RPolygon  # The polygon represented in rportion
    _area: float  # The area of the polygon
    _vertices: Optional[Vertices]  # The vertices of the polygon (if simple)
    _convex: Optional[
        list[bool]
    ]  # List of booleans indicating if the vertex is convex (True) or concave (False)

    def __init__(self, rectangles: Iterable[XY_Box] = list()):
        self._polygon = RPolygon()
        self._area = -1.0
        self._vertices = None
        self._convex = None
        for r in rectangles:
            self._polygon |= rclosedopen(
                float(r.xmin), float(r.xmax), float(r.ymin), float(r.ymax)
            )

    @property
    def rpolygon(self) -> RPolygon:
        """Returns the polygon represented as an RPolygon."""
        return self._polygon

    def dup(self) -> FPolygon:
        """Returns a deep copy of the Polygon."""
        c = FPolygon()
        c._polygon = copy.deepcopy(self._polygon)
        return c

    @property
    def area(self) -> float:
        """Returns the area of the polygon"""
        if self._area < 0.0:
            self._area = sum(
                RPoly2Box(r).area for r in self._polygon.rectangle_partitioning()
            )
        return self._area

    @property
    def is_simple(self) -> bool:
        """Returns True if the polygon is simple (connected and without holes)."""
        try:
            self.vertices()
            return True
        except AssertionError:
            return False

    def __or__(self, other: FPolygon) -> FPolygon:
        """Returns the union of polygons"""
        rec_copy = FPolygon()
        rec_copy._polygon = self.rpolygon | other.rpolygon
        return rec_copy

    def __and__(self, other: FPolygon) -> FPolygon:
        """Returns the intersection of polygons"""
        rec_copy = FPolygon()
        rec_copy._polygon = self.rpolygon & other.rpolygon
        return rec_copy

    def __sub__(self, other: FPolygon) -> FPolygon:
        """Returns the difference self-other"""
        rec_copy = FPolygon()
        rec_copy._polygon = self.rpolygon - other.rpolygon
        return rec_copy

    def __eq__(self, other: object) -> bool:
        """Checks equality of two polygons"""
        if not isinstance(other, FPolygon):
            return NotImplemented
        return self.rpolygon == other.rpolygon

    def __repr__(self) -> str:
        """Returns the representation of the polygon"""
        return repr(self.rpolygon)

    def maximal_rectangles(self) -> Iterator[XY_Box]:
        """Returns an iterator of the maximal rectangles included in the polygon."""
        for r in self.rpolygon.maximal_rectangles():
            yield RPoly2Box(r)

    def jaccard_similarity(self, other: FPolygon) -> float:
        """Returns the Jaccard similarity between two polygons.
        The Jaccard similarity between two polygons P1 and P2 is a value
        in [0,1] defined as Area(P1&P2)/Area(P1|P2)."""
        return (self & other).area / (self | other).area

    def vertices(self) -> Vertices:
        """Converts a polygon into a sequence of vertices.
        The sequence represents de boundaries of the polygon in clockwise order.
        The edges are represented by segments between consecutive vertices.
        The first vertex is the leftmost lowest vertex.
        The polygon must be a simple polygon (connected and without holes).
        An exception is raised if the polygon is not simple."""

        # Check if already computed
        if self._vertices:
            return self._vertices

        boundary = self._polygon.boundary()
        edges = list(boundary.rectangle_partitioning())

        assert len(edges) >= 4, "A polygon must have at least 4 edges"

        # Create the contour as a mapping from each vertex to its
        # adjacent vertices (each vertex should have 2 adjacent vertices)
        cont = dict[RPoint, set[RPoint]]()
        for e in edges:
            box = RPoly2Box(e)
            if box.is_vertical_edge:
                p1 = (box.xmin, box.ymin)
                p2 = (box.xmin, box.ymax)
            elif box.is_horizontal_edge:
                p1 = (box.xmin, box.ymin)
                p2 = (box.xmax, box.ymin)
            else:
                assert False, "Internal error: neither vertical nor horizontal edge"

            if p1 not in cont:
                cont[p1] = set()
            if p2 not in cont:
                cont[p2] = set()
            cont[p1].add(p2)
            cont[p2].add(p1)

        # Sanity check that all vertices have degree 2
        for v, adj in cont.items():
            assert len(adj) == 2, f"Vertex {v} has degree {len(adj)} != 2"

        # Let just make it deterministic. Let us pick the lowest-leftmost vertex as start
        # The second vertex will be the adjacent one with lowest-leftmost coordinates
        start_vertex = min(cont.keys(), key=lambda p: (p[0], p[1]))
        self._vertices = [start_vertex]
        second_vertex = max(cont[start_vertex], key=lambda p: (p[1], p[0]))
        cont[start_vertex].remove(second_vertex)
        cont[second_vertex].remove(start_vertex)
        self._vertices.append(second_vertex)
        current = second_vertex
        while True:
            assert current is not None
            next_vertices = cont[current]
            if len(next_vertices) == 0:
                assert self._vertices[0] == self._vertices[-1], "Not a simple polygon"
                self._vertices.pop()  # Remove the last vertex (equal to the first)
                break
            new_vertex = next_vertices.pop()
            cont[new_vertex].remove(current)
            self._vertices.append(new_vertex)
            current = new_vertex

        # Check that all points have been visited
        assert len(self._vertices) == len(cont), "Not a simple polygon"
        return self._vertices

    def convex(self) -> list[bool]:
        """Returns a list of booleans indicating if the vertex is convex (True) or concave (False).
        The list is in the same order as the vertices returned by vertices()."""
        if self._convex:
            return self._convex

        vertices = self.vertices()
        n = len(vertices)
        self._convex = [False] * n
        for i in range(n):
            p_prev = vertices[i - 1]
            p_curr = vertices[i]
            p_next = vertices[(i + 1) % n]
            cross_product = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (
                p_curr[1] - p_prev[1]
            ) * (p_next[0] - p_curr[0])
            self._convex[i] = (
                cross_product < 0
            )  # Convex if cross product is negative (clockwise order)
        return self._convex


def vertices2polygon(vertices: Iterable[RPoint]) -> FPolygon:
    """Converts a set of vertices into a simple FPolygon.
    The vertices represent a simple polygon (without holes).
    Based on the paper A Polygon-to-Rectangle Conversion Algorithm
    by Kevin D. Gourley and Douglas M. Green."""

    v = set(vertices)
    rectangles: list[XY_Box] = []

    while v:
        # Find the two leftmost vertices with minimum y coordinate
        pk = min(v, key=lambda p: (p[1], p[0]))
        v.remove(pk)
        pl = min(v, key=lambda p: (p[1], p[0]))
        v.remove(pl)
        assert pk[1] == pl[1], "Sanity check: same y coordinate"
        # Points in the [pk.x,pl.x] interval with y > pk.y
        new_v = {p for p in v if pk[0] <= p[0] <= pl[0] and p[1] > pk[1]}
        # Pick the point with minimum y coordinate
        pm = min(new_v, key=lambda p: (p[1], p[0]))
        rectangles.append(XY_Box(pk[0], pl[0], pk[1], pm[1]))
        # Update the vertex set
        pkm = (pk[0], pm[1])
        plm = (pl[0], pm[1])
        if pkm in v:
            v.remove(pkm)
        else:
            v.add(pkm)
        if plm in v:
            v.remove(plm)
        else:
            v.add(plm)

    return FPolygon(rectangles)
