# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import unittest
from cpupc.geometry.fpolygon import (
    FPolygon,
    RPoint,
    XY_Box,
    vertices2polygon,
)
from cpupc.geometry.strop import Strop


class TestFPolygon(unittest.TestCase):
    def setUp(self) -> None:
        self.r1 = FPolygon([XY_Box(1, 5, 1, 4)])
        self.r2 = FPolygon([XY_Box(2, 4, 3, 6)])
        self.r3 = self.r1 | self.r2
        self.r4 = self.r1 & self.r2
        self.r5 = self.r1 - self.r2
        self.empty = FPolygon()
        points: list[RPoint] = [
            (0, 3),
            (6, 10),
            (11, 3),
            (1, 4),
            (0, 5),
            (10, 10),
            (11, 0),
            (1, 3),
            (4, 5),
            (10, 5),
            (8, 0),
            (4, 6),
            (14, 5),
            (8, 2),
            (0, 6),
            (14, 8),
            (5, 2),
            (0, 9),
            (12, 8),
            (1, 9),
            (12, 10),
            (5, 1),
            (1, 7),
            (16, 10),
            (3, 1),
            (6, 7),
            (16, 3),
            (3, 4),
        ]
        self.p = vertices2polygon(points)
        self.strop = Strop(self.p)

    def test_area(self) -> None:
        self.assertEqual(self.r1.area, 12)
        self.assertEqual(self.r2.area, 6)
        self.assertEqual(self.r3.area, 16)
        self.assertEqual(self.r4.area, 2)
        self.assertEqual(self.r5.area, 10)
        self.assertEqual(self.empty.area, 0)
        self.assertEqual(self.p.area, 90)

    def test_strop(self) -> None:
        self.assertEqual(self.strop.area, 80)
        self.assertEqual(self.strop.num_branches, 8)
        self.assertAlmostEqual(self.strop.similarity, 80 / 90, 7)

    def test_strop_reduction(self) -> None:
        strop = self.strop.dup()
        print(f"Initial strop: area={strop.area}, branches={strop.num_branches}, similarity={strop.similarity}")
        while strop.num_branches > 0:
            new_strop = strop.reduce()
            print(f"Reduced strop: area={new_strop.area}, branches={new_strop.num_branches}, similarity={new_strop.similarity}")
            self.assertEqual(new_strop.reference, strop.reference)
            self.assertAlmostEqual(new_strop.area, strop.area, 7)
            self.assertTrue(new_strop.similarity <= strop.similarity + 0.02)
            self.assertEqual(new_strop.num_branches, strop.num_branches - 1)
            strop = new_strop


class TestVertices(unittest.TestCase):
    p1: FPolygon
    p2: FPolygon
    p3: FPolygon
    p4: FPolygon
    solution1: list[tuple[float, float]]

    @classmethod
    def setUpClass(cls) -> None:
        # A simple polygon
        cls.p1 = FPolygon(
            [
                XY_Box(0, 4, 2, 5),
                XY_Box(2, 4, 2, 7),
                XY_Box(3, 6, 0, 2),
                XY_Box(4, 6, 0, 8),
            ]
        )

        cls.solution1 = [
            (0.0, 2.0),
            (0.0, 5.0),
            (2.0, 5.0),
            (2.0, 7.0),
            (4.0, 7.0),
            (4.0, 8.0),
            (6.0, 8.0),
            (6.0, 0.0),
            (3.0, 0.0),
            (3.0, 2.0),
        ]

        # A disconnected polygon
        cls.p2 = FPolygon(
            [
                XY_Box(0, 2, 0, 2),
                XY_Box(3, 5, 3, 5),
            ]
        )

        # A polygon with a hole
        cls.p3 = FPolygon(
            [
                XY_Box(0, 1, 0, 3),
                XY_Box(2, 3, 0, 3),
                XY_Box(0, 3, 0, 1),
                XY_Box(0, 3, 2, 3),
            ]
        )

        # A complex polygon
        cls.p4 = FPolygon(
            [
                XY_Box(1, 4, 6, 7),
                XY_Box(0, 2, 2, 4),
                XY_Box(2, 4, 2, 6),
                XY_Box(4, 7, 3, 5),
                XY_Box(3, 5, 0, 2),
                XY_Box(5, 6, 1, 2),
                XY_Box(4, 5, 0, 3),
            ]
        )

    def test_polygon2vertices(self) -> None:
        self.assertEqual(self.p1.vertices, self.solution1)
        self.assertTrue(self.p2.vertices is None)
        self.assertTrue(self.p3.vertices is None)

    def test_vertices2polygon(self) -> None:
        new_poly = vertices2polygon(self.solution1)
        self.assertEqual(new_poly, self.p1)

        vertices = self.p4.vertices
        assert vertices is not None, "Vertices should not be None for p4"
        new_poly = vertices2polygon(vertices)
        self.assertEqual(new_poly, self.p4)

    def test_convexity(self) -> None:
        expected_convexity = [
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
        ]
        self.assertEqual(self.p1.convex, expected_convexity)

        convexity = self.p4.convex
        expected_convexity = [
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ]
        self.assertEqual(convexity, expected_convexity)


if __name__ == "__main__":
    unittest.main()
