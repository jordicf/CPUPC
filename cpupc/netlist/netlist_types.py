# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""
Types used in the netlist
"""

from dataclasses import dataclass
from math import sqrt
from typing import Iterator

from .module import Module
from ..geometry.geometry import Point


@dataclass
class Edge:
    """
    Representation of an edge in a graph (possibly obtained from a hypergraph).
    The edge represents the target node in an adjacency list
    """

    node: str  # Name of the target
    weight: float  # Weight of the edge


@dataclass
class NamedHyperEdge:
    """Representation of a hyperedge (list of module names)"""

    modules: list[str]  # List of module names of the hyperedge
    weight: float  # Weight of the hyperedge

    def __repr__(self) -> str:
        if self.weight == 1:
            return f"Edge<modules={self.modules}>"
        return f"Edge<modules={self.modules}, weight={self.weight}>"


@dataclass
class HyperEdge:
    """Representation of a hyperedge (list of modules)"""

    modules: list[Module]  # List of modules of the hyperedge
    weight: float  # Weight of the hyperedge

    @property
    def wire_length(self) -> float:
        """
        Returns the wire length of the hyperedge.
        The wire length is the distance between the center of each module and
        the centroid of the modules (without taking into account module areas)
        multiplied by the hyperedge weight.
        """
        intersection_point = Point(0, 0)
        for b in self.modules:
            assert (
                b.center is not None
            ), f"Center must be defined for module {b.name}. Maybe execute b.calculate_center_from_rectangles()?"
            intersection_point += b.center
        intersection_point /= len(self.modules)
        wire_length = 0.0
        for b in self.modules:
            # redundant assertion (will never happen) for type checking
            assert b.center is not None
            v = intersection_point - b.center
            wire_length += sqrt(v & v)
        return wire_length * self.weight

    def __repr__(self) -> str:
        names = [b.name for b in self.modules]
        if self.weight == 1:
            return f"Edge<modules={names}>"
        return f"Edge<modules={names}, weight={self.weight}>"


class ModuleClusters:
    """
    Representation of clusters of modules in the netlist.
    Each cluster is a list of module names.
    """

    _clusters: dict[str, set[Module]]  # Mapping from cluster name to set of modules

    def __init__(self):
        self._clusters = {}

    def add_module_to_cluster(self, cluster_name: str, module: Module):
        """Adds a module to a cluster."""
        if cluster_name not in self._clusters:
            self._clusters[cluster_name] = set()
        self._clusters[cluster_name].add(module)

    def get_cluster_names(self) -> list[str]:
        """Returns the list of cluster names."""
        return list(self._clusters.keys())

    def get_modules_in_cluster(self, cluster_name: str) -> set[Module]:
        """Returns the set of modules in a cluster."""
        assert (
            cluster_name in self._clusters
        ), f"Cluster {cluster_name} does not exist in the netlist"
        return self._clusters[cluster_name]
    
    def get_clusters(self) -> Iterator[set[Module]]:
        """Returns an iterator over the clusters."""
        for cluster in self._clusters.values():
            yield cluster

    def __repr__(self) -> str:
        return f"ModuleClusters<clusters={self._clusters}>"
