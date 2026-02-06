# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

from dataclasses import dataclass
from cpupc.netlist.netlist import Netlist
from cpupc.netlist.module import Module, Point


@dataclass(slots=True)
class swapPoint:
    """A point in the netlist representing the position of a module."""

    x: float
    y: float
    nets: list[int]  # List of net IDs that this point belongs to (sorted and unique)
    x_orig: float # Original coordinates before splitting
    y_orig: float
    def __init__(self, x: float, y: float) -> None:
        self.x = self.x_orig = x
        self.y = self.y_orig = y
        self.nets = []


@dataclass(slots=True)
class swapNet:
    """A net in the netlist."""

    weight: float  # Weight of the net
    points: list[int]  # List of point IDs that belong to this net
    hpwl: float  # Half-perimeter wire length (initialized to 0.0)
    internal: bool # if the net connects splits of a module, instead of distinct modules

    def __init__(self, weight: float, points: list[int], internal: bool = False) -> None:
        self.weight = weight
        self.points = points
        self.hpwl = 0.0
        self.internal = internal


class swapNetlist:
    """A netlist consisting of points and nets."""

    __slots__ = (
        "_netlist",
        "_points",
        "_nets",
        "_external_nets",
        "_name2idx",
        "_idx2name",
        "_movable",
        "_areas",
        "_hpwl",
        "_avg_area",
        "_num_subblocks",
        "_split_net_factor",
        "_split_threshold",
        "_star_model",
        "_verbose",
    )

    _netlist: Netlist  # Original netlist
    _points: list[swapPoint]
    _nets: list[swapNet]
    _external_nets: int # Number of non-internal nets
    _name2idx: dict[str, int]  # Mapping from names to indices of points
    _idx2name: list[str]  # Mapping from indices to names of points
    _movable: list[int]  # List of movable point indices
    _areas: list[float] # List of module areas, movable or not
    _hpwl: float  # Total HPWL of the netlist
    _avg_area: float  # Average area of the movable modules
    _num_subblocks: int  # Number of fake subblocks created by splits
    _split_net_factor: float  # Weight factor for nets created by splits
    _split_threshold: float # Percentage of area to be split, in [0,1] range
    _star_model: bool # Whether to use star model for internal nets, grid model otherwise
    _verbose: bool  # Verbosity flag

    def __init__(
        self, filename: str, split_net_factor: float = 1.0,
        split_threshold: float = 0.5, star_model: bool = False,
        verbose: bool = False
    ) -> None:
        self._points = []
        self._nets = []
        self._name2idx = {}
        self._idx2name = []
        self._movable = []
        self._areas = []
        self._netlist = Netlist(filename)
        self._avg_area = 0.0
        self._num_subblocks = 0
        self._split_net_factor = split_net_factor
        self._split_threshold = split_threshold
        self._star_model = star_model
        self._verbose = verbose

        # Read modules
        self._netlist.calculate_centers_from_rectangles()
        movable_area = 0.0
        for m in self.netlist.modules:
            idx = len(self.points)
            self._name2idx[m.name] = idx
            self._idx2name.append(m.name)
            assert m.center is not None
            area = sum(r.area for r in m.rectangles)
            self.points.append(swapPoint(x=m.center.x, y=m.center.y))
            self._areas.append(area)
            if not m.is_hard:
                assert (
                    len(m.rectangles) == 1
                ), "Only one rectangle per soft module is supported"
                self._movable.append(idx)
                movable_area += area

        assert len(self._movable) > 0, "No movable modules found"
        self._avg_area = movable_area / len(self._movable)

        # Read nets
        for e in self._netlist.edges:
            point_indices = []
            for m in e.modules:
                idx = self._name2idx[m.name]
                point_indices.append(idx)
                self.points[idx].nets.append(len(self.nets))
            self.nets.append(swapNet(weight=e.weight, points=point_indices))
        self._external_nets = len(self.nets)

        # Split movable modules that are too large
        if self._split_net_factor > 0.0 and self._split_threshold > 0.0:
            self._split_modules()

        # Sort the nets of each point
        for p in self.points:
            # Sorted and unique nets (to avoid repeated nets)
            p.nets = sorted(set(p.nets))

        # Compute initial HPWL (this includes the internal 'fake' nets)
        self.hpwl = sum(self._compute_net_hpwl(net) for net in self.nets)

    @property
    def netlist(self) -> Netlist:
        """Original netlist."""
        return self._netlist

    @property
    def movable(self) -> list[int]:
        """List of movable point indices."""
        return self._movable

    @property
    def points(self) -> list[swapPoint]:
        """List of points in the netlist."""
        return self._points

    @property
    def nets(self) -> list[swapNet]:
        """List of nets in the netlist."""
        return self._nets

    @property
    def hpwl(self) -> float:
        """Total half-perimeter wire length (HPWL) of the netlist."""
        return self._hpwl

    @hpwl.setter
    def hpwl(self, value: float) -> None:
        self._hpwl = value

    def idx2name(self, i: int) -> str:
        """Mapping from point index to module name."""
        return self._idx2name[i]

    def idx2module(self, i: int) -> Module:
        """Get the module corresponding to a point index."""
        name = self.idx2name(i)
        return self._netlist.get_module(name)

    def name2idx(self, name: str) -> int:
        """Mapping from module name to point index."""
        return self._name2idx[name]

    def _compute_net_hpwl(self, net: swapNet) -> float:
        """Compute the half-perimeter wire length (HPWL) of a net.
        It returns the computed HPWL."""
        xs = [self.points[p].x for p in net.points]
        ys = [self.points[p].y for p in net.points]
        net.hpwl = (max(xs) - min(xs) + max(ys) - min(ys)) * net.weight
        return net.hpwl

    def compute_total_hpwl(self) -> float:
        """Compute the total half-perimeter wire length (HPWL) of the netlist.
        It returns the computed total HPWL."""
        self.hpwl = sum(self._compute_net_hpwl(net) for net in self.nets)  # Reset HPWL
        return self.hpwl

    def _split_modules(self) -> None:
        """Split all movable modules that are too large."""
        num_movable = len(self.movable)
        target_area = self.get_target_area()
        for i in range(num_movable):
            idx = self.movable[i]
            m = self.idx2module(idx)
            if m.area() <= target_area:
                continue

            assert not m.is_hard, "Only soft modules can be split"
            assert (
                len(m.rectangles) == 1
            ), "Only one rectangle per soft module is supported"
            r = m.rectangles[0]
            nrows, ncols = _best_split(
                r.shape.w, r.shape.h, target_area, aspect_ratio=0.5
            )
            if nrows > 1 or ncols > 1:
                if self._verbose:
                    print(
                        f"Splitting module {m.name} (area {r.area:.1f}) "
                        f"into {nrows}x{ncols} sub-blocks "
                        f"with net weight {self._split_net_factor:.1f}"
                    )
                self._split_module(idx, nrows, ncols, net_weight=self._split_net_factor)

    def _split_module(
        self, idx: int, nrows: int, ncols: int, net_weight: float = 1.0
    ) -> None:
        """Split a module into a matrix of smaller modules."""
        # Get the rectangle of the module to be split
        m = self._netlist.get_module(self.idx2name(idx))
        assert not m.is_hard, "Only soft modules can be split"
        assert len(m.rectangles) == 1, "Only one rectangle per soft module is supported"
        
        r = m.rectangles[0]
        center = r.center
        width, height = r.shape.w, r.shape.h

        # Compute the new dimensions
        new_width = width / ncols
        new_height = height / nrows
        # center of the top left submodule in the split
        top_left_center = Point(center.x - width / 2 + new_width / 2, 
                                center.y + height / 2 - new_height / 2)
        
        # find the submodule whose center minimises HPWL of external nets
        i_central, j_central = \
            self._find_central_submodule(idx, nrows, ncols)
        
        # Create new modules
        for i in range(nrows):
            for j in range(ncols):
                if i == i_central and j == j_central:
                    continue  # Skip the original module
                
                new_module = swapPoint(
                    x=top_left_center.x + j * new_width, y=top_left_center.y - i * new_height
                )
                new_idx = len(self.points)
                new_name = f"{m.name}_split_{new_idx}"
                self._name2idx[new_name] = new_idx
                self._idx2name.append(new_name)
                self.points.append(new_module)
                self.movable.append(new_idx)
                self._num_subblocks += 1
                # add internal nets
                net_idx = len(self.nets)
                if self._star_model:
                    # Star model for the nets between center and sub-blocks
                    self.nets.append(swapNet(weight=net_weight, points=[idx, new_idx], internal=True))
                    self.points[idx].nets.append(net_idx)
                    self.points[new_idx].nets.append(net_idx)
                else: # grid model
                    # add net towards submodule (i,j-1)
                    if j > 0:
                        net_idx = len(self.nets)
                        neigh_idx = new_idx - 1
                        if i == i_central and j - 1 == j_central:
                            neigh_idx = idx

                        self.nets.append(swapNet(weight=net_weight, points=[neigh_idx, new_idx], internal=True))
                        self.points[neigh_idx].nets.append(net_idx)
                        self.points[new_idx].nets.append(net_idx)
                    # add net towards submodule (i-1,j)
                    if i > 0:
                        net_idx = len(self.nets)
                        neigh_idx = new_idx - ncols
                        
                        # special care with the (i_central, j_central) submodule, since it keeps the original index
                        if i - 1 == i_central and j == j_central: 
                            # case (i-1,j) = (i_central, j_central)
                            neigh_idx = idx

                        elif i - 1 == i_central and j < j_central or \
                             i == i_central and j > j_central: 
                            # case (i_central, j_central) is between (i-1,j) and (i,j) in indexing order
                            # since its index was skipped, we are off by one
                            neigh_idx += 1
                        
                        self.nets.append(swapNet(weight=net_weight, points=[neigh_idx, new_idx], internal=True))
                        self.points[neigh_idx].nets.append(net_idx)
                        self.points[new_idx].nets.append(net_idx)

                    # add net towards submodule (i+1,j) if it is the central one
                    if i + 1 == i_central and j == j_central:
                        net_idx = len(self.nets)
                        neigh_idx = idx
                        self.nets.append(swapNet(weight=net_weight, points=[neigh_idx, new_idx], internal=True))
                        self.points[neigh_idx].nets.append(net_idx)
                        self.points[new_idx].nets.append(net_idx)

                    # add net towards submodule (i,j+1) if it is the central one
                    if i == i_central and j + 1 == j_central:
                        net_idx = len(self.nets)
                        neigh_idx = idx
                        self.nets.append(swapNet(weight=net_weight, points=[neigh_idx, new_idx], internal=True))
                        self.points[neigh_idx].nets.append(net_idx)
                        self.points[new_idx].nets.append(net_idx)
                    

        # Update the module's center to be the central
        module = self.points[idx]
        module.x = top_left_center.x + j_central * new_width
        module.y = top_left_center.y - i_central * new_height

    def _find_central_submodule(
        self, idx: int, nrows: int, ncols: int
    ) -> tuple[int, int]:
        """
        Finds the central submodule of a split module,
        which is the submodule that minimizes hpwl
        """
        # naive approach, calculate HPWL for each possibility and return best

        m = self._netlist.get_module(self.idx2name(idx))       
        r = m.rectangles[0]
        center = r.center
        width, height = r.shape.w, r.shape.h

        new_width = width / ncols
        new_height = height / nrows
        # center of the top left submodule in the split
        top_left_center = Point(center.x - width / 2 + new_width / 2, 
                                center.y + height / 2 - new_height / 2)

        best_hpwl = float('inf')
        best_ij = (0, 0)
        best_center = top_left_center

        for i in range(nrows):
            for j in range(ncols):
                center_ij = Point(
                    top_left_center.x + j * new_width,
                    top_left_center.y - i * new_height
                    )
                self.points[idx].x = center_ij.x
                self.points[idx].y = center_ij.y

                external_hpwl = sum([
                    self._compute_net_hpwl(self.nets[n]) for n in self.points[idx].nets
                ])

                if external_hpwl < best_hpwl:
                    best_ij = (i, j)
                    best_center = center_ij

                elif external_hpwl == best_hpwl:
                    # break ties with L1 distance to geometric center
                    if abs(center.x - center_ij.x) + abs(center.y - center_ij.y) < \
                       abs(center.x - best_center.x) + abs(center.y - best_center.y):
                        best_ij = (i,j)
                        best_center = center_ij

        # restore original center
        self.points[idx].x = center.x
        self.points[idx].y = center.y

        return best_ij

    def remove_subblocks(self) -> None:
        """Remove all fake subblocks created by splits.
        Only the name mappings are removed."""
        if self._num_subblocks == 0:
            return

        if self._verbose:
            print(f"Removing {self._num_subblocks} fake subblocks created by splits.")
        for _ in range(self._num_subblocks):
            # Remove from idx2name mappings
            name = self._idx2name.pop()
            del self._name2idx[name]

        del self._points[-self._num_subblocks :]
        del self._movable[-self._num_subblocks :]
        del self._nets[-self._num_subblocks :]

    def get_target_area(self) -> float:
        """Returns the largest area a module can have before needing splitting"""
        movable_areas: list[float] = sorted([self._areas[i] for i in self.movable])
        movable_area: float = sum(movable_areas)

        acc: float = 0.0
        i : int = 0
        while acc < movable_area * (1 - self._split_threshold):
            acc += movable_areas[i]
            i += 1

        return movable_areas[i - 1]



def _best_split(
    width: float, height: float, target_area: float, aspect_ratio: float = 0.5
) -> tuple[int, int]:
    """Compute the best odd split (nrows, ncols) for a module of given
    width, height, and target area. The solution must satisfy the aspect ratio constraint.
    It returns the best (nrows, ncols) split."""
    area = width * height
    best_cost = float("inf")
    best_split = None
    max_slices = int(area / target_area) + 2
    for nrows in range(1, max_slices, 1):
        for ncols in range(1, max_slices, 1):
            new_width = width / ncols
            new_height = height / nrows
            block_area = new_width * new_height
            ratio = min(new_width / new_height, new_height / new_width)
            if ratio < aspect_ratio:
                continue
            cost = abs(target_area - block_area)
            if cost < best_cost:
                best_cost, best_split = cost, (nrows, ncols)

    if best_split is not None:
        return best_split
    # Just in case, try with a more relaxed aspect ratio
    return _best_split(width, height, target_area, aspect_ratio * 0.95)


if __name__ == "__main__":
    w = input()
    h = input()
    a = input()
    print(_best_split(float(w), float(h), float(a)))
