# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import unittest
from pathlib import Path

import torch

from cpupc.netlist.netlist import Netlist
from tools.floorset.writenetlist import decompose_hyperedges, write_netlist

MODULES = """
Modules:
  A: {area: 4.0, center: [0.0, 0.0]}
  B: {area: 4.0, center: [2.0, 0.0]}
  C: {area: 4.0, center: [2.0, 2.0]}
  D: {area: 4.0, center: [0.0, 2.0]}
  p1: {center: [0.0, 4.0], io_pin: true, length: 0.0}
  p2: {center: [4.0, 4.0], io_pin: true, length: 0.0}
"""


def edges_of(net: Netlist):
    """Decomposes the edges of a netlist into (src, dst, weight) name tuples."""
    triplets = []
    edges = decompose_hyperedges(net.edges)
    for e in edges:
        assert len(e.modules) == 2
        triplets.append((e.modules[0].name, e.modules[1].name, e.weight))
    return triplets


class TestDecomposeHyperedges(unittest.TestCase):

    def _net(self, nets: str) -> Netlist:
        return Netlist(MODULES + "Nets:\n" + nets)

    def test_clique_hyperedge(self):
        # 4 distinct modules, weight 3 -> 6 pairs of weight 3/C(4,2)=0.5
        net = self._net("  - [A, B, C, p1, 3.0]\n")
        self.assertEqual(
            edges_of(net),
            [
                ("A", "B", 0.5),
                ("A", "C", 0.5),
                ("p1", "A", 0.5),
                ("B", "C", 0.5),
                ("p1", "B", 0.5),
                ("p1", "C", 0.5),
            ],
        )

    def test_pair_weight_summed_across_hyperedges(self):
        # Pair (A, B) appears in both nets: 6/C(3,2)=2 from the first net and
        # 4/C(2,2)=4 from the second -> a single edge of weight 2+4=6
        net = self._net("  - [A, B, C, 6.0]\n  - [A, B, 4.0]\n")
        self.assertEqual(
            edges_of(net),
            [("A", "B", 6.0), ("A", "C", 2.0), ("B", "C", 2.0)],
        )

    def test_two_pin_nets(self):
        # Block-block order preserved; the pin is moved to the source
        net = self._net("  - [A, B, 2.0]\n  - [A, p1]\n")
        self.assertEqual(edges_of(net), [("A", "B", 2.0), ("p1", "A", 1.0)])

    def test_dedupe(self):
        # [A, A, B] has p=2 distinct modules: a single edge of weight w
        self.assertEqual(edges_of(self._net("  - [A, A, B, 2.0]\n")), [("A", "B", 2.0)])

    def test_self_loop_dropped(self):
        # [A, A] dedupes to a single distinct module (a self-loop) and is
        # dropped. Self-loops occur in the MCNC ami33/ami49 benchmarks.
        self.assertEqual(edges_of(self._net("  - [A, A, 2.0]\n")), [])


class TestWriteNetlist(unittest.TestCase):

    netlist_file = str(Path(__file__).resolve().parent / "netlist.yml")

    def test_regression_two_pin_netlist(self):
        # All the nets are 2-pin block-block edges, so enabling
        # the clique decomposition must not change the floorset output
        ref_data, ref_label, ref_names = write_netlist(Netlist(self.netlist_file))
        data, label, names = write_netlist(Netlist(self.netlist_file), hyperedge=True)
        self.assertEqual(names, ref_names)
        for t, t_ref in zip(data, ref_data, strict=True):
            self.assertTrue(torch.equal(t, t_ref))
        self.assertTrue(torch.equal(label[0], ref_label[0]))
        for s, s_ref in zip(label[1], ref_label[1], strict=True):
            self.assertTrue(torch.equal(s, s_ref))

    def test_clique_in_floorset(self):
        net = Netlist(MODULES + "Nets:\n  - [A, B, C, D, 3.0]\n  - [A, p1]\n")
        net.create_squares()
        data, label, names = write_netlist(net, hyperedge=True)
        t_module, t_b2b, t_p2b, t_pin = data
        # There are 4 blocks
        self.assertEqual(t_module.shape, (4, 6))
        self.assertEqual(names["modules"], ["A", "B", "C", "D"])
        # The 4-block net gives C(4,2)=6 b2b edges of weight 3/C(4,2)=0.5
        self.assertEqual(t_b2b.shape, (6, 3))
        self.assertTrue(torch.all(t_b2b[:, 2] == 0.5))
        # The [A, p1] net gives a single pin-block edge
        self.assertEqual(t_p2b.shape, (1, 3))
        self.assertEqual(t_pin.shape, (2, 2))
        # The solution has one polygon per real block
        self.assertEqual(len(label[1]), 4)
        metrics = label[0]
        self.assertEqual(metrics[0].item(), 16.0)  # bounding-box area
        self.assertEqual(metrics[2].item(), 7.0)  # number of edges (6 b2b + 1 p2b)


if __name__ == "__main__":
    unittest.main()
