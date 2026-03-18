#!/usr/bin/env python3
"""
Quadratic Placement (QP) standalone implementation
Input: netlist.yaml (e.g., ami33.yaml) and die.yaml
Output: placement result in YAML format 
"""

import numpy as np
import yaml
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import bicgstab
from typing import Dict, List, Tuple
import math
import time
import argparse
import sys
import os

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Module:
    """Represents a cell or io_pin"""
    def __init__(self, name: str, area: float = 0, aspect_ratio: List[float] = None,
                 center: List[float] = None, io_pin: bool = False, fixed: bool = False,
                 hard: bool = False, rectangles: List[List[float]] = None, 
                 original_attrs: dict = None):
        self.name = name
        self.area = area
        self.aspect_ratio = aspect_ratio or [1.0, 1.0]
        self.center = np.array(center or [0.0, 0.0], dtype=np.float64)
        self.io_pin = io_pin
        self.fixed = fixed
        self.hard = hard
        self.rectangles = rectangles  # Store original rectangles data
        self.original_attrs = original_attrs or {}  # Store original YAML attributes
        
        # Calculate width/height from rectangles (for hard modules) or area
        if rectangles and len(rectangles) > 0:
            # Check if rectangles is a flat list (io_pin format) or nested list (module format)
            if io_pin and isinstance(rectangles[0], (int, float)):
                # IO pin format: flat list [x, y, 0, 0] (only center coordinates)
                if len(rectangles) >= 2:
                    self.center = np.array([rectangles[0], rectangles[1]], dtype=np.float64)
                    self.width = 0.0
                    self.height = 0.0
                else:
                    raise ValueError(f"Module {name}: IO pin rectangles must have at least [x, y], got: {rectangles}")
            else:
                # Hard/fixed module: nested list format [[x, y, w, h], ...]
                # For trunk (first rectangle), extract width and height
                trunk = rectangles[0]
                
                # Handle different formats: trunk might be a dict, list, or nested structure
                if isinstance(trunk, dict):
                    # If trunk is a dict with 'trunk' key (old format)
                    if 'trunk' in trunk:
                        trunk_data = trunk['trunk']
                    else:
                        raise ValueError(f"Module {name}: Expected 'trunk' key in rectangles dict, got: {trunk}")
                elif isinstance(trunk, (list, tuple)) and len(trunk) >= 4:
                    # Standard format: [x, y, w, h]
                    trunk_data = trunk
                else:
                    raise TypeError(f"Module {name}: Invalid trunk format. Expected list/tuple with 4 elements [x, y, w, h], got: {type(trunk).__name__} = {trunk}")
                
                self.center = np.array([trunk_data[0], trunk_data[1]], dtype=np.float64)
                self.width = trunk_data[2]
                self.height = trunk_data[3]
                # Calculate actual area from rectangles
                if isinstance(rectangles[0], dict):
                    # If dict format, need to extract all rectangles properly
                    self.area = trunk_data[2] * trunk_data[3]
                else:
                    self.area = sum(rect[2] * rect[3] for rect in rectangles)
        elif area > 0 and not io_pin:
            # Soft module: calculate width/height from area and aspect ratio
            ar = (aspect_ratio[0] + aspect_ratio[1]) / 2 if aspect_ratio else 1.0
            self.height = math.sqrt(area / ar)
            self.width = area / self.height
        else:
            # io_pins are very small
            self.width = 0.0
            self.height = 0.0
        
        self.pins = []  # List of net indices this module connects to
        self.idx = -1  # Will be set during initialization


class Net:
    """Represents a net connecting multiple modules"""
    def __init__(self, modules: List[str], weight: float = 1.0):
        self.modules = modules  # List of module names (without weight)
        self.weight = weight    # Net weight (default 1.0)
        self.module_indices = []  # Will be populated with Module objects


class PlacementDB:
    """Database holding all placement information"""
    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.movable_modules: List[Module] = []  # Movable modules (soft + hard)
        self.fixed_modules: List[Module] = []    # Fixed modules (position fixed, non-io_pin)
        self.io_pins: List[Module] = []          # IO pins / terminals
        self.nets: List[Net] = []
        
        self.die_width = 0
        self.die_height = 0
        self.core_ll = np.array([0.0, 0.0])
        self.core_ur = np.array([0.0, 0.0])
    
    def load_yaml(self, netlist_file: str, die_file: str):
        """Load netlist and die from YAML files"""
        # Load netlist
        with open(netlist_file, 'r', encoding='utf-8') as f:
            netlist_data = yaml.safe_load(f)
        
        # Load die
        with open(die_file, 'r', encoding='utf-8') as f:
            die_data = yaml.safe_load(f)
        
        self.die_width = die_data['width']
        self.die_height = die_data['height']
        self.core_ll = np.array([0.0, 0.0])
        self.core_ur = np.array([self.die_width, self.die_height], dtype=np.float64)
        
        # Parse modules
        modules_data = netlist_data.get('Modules', {})
        for name, attrs in modules_data.items():
            # Determine module type:
            # - io_pin/terminal: io_pin=true or terminal=true, position fixed, no area
            # - fixed: fixed=true (non-io_pin), position AND size fixed
            # - hard: hard=true, position movable, size fixed
            # - soft: default, position and size both optimizable
            
            is_io_pin = attrs.get('io_pin', False) or attrs.get('terminal', False)
            is_fixed = attrs.get('fixed', False) and not is_io_pin
            is_hard = attrs.get('hard', False) and not is_fixed and not is_io_pin
            
            rectangles = attrs.get('rectangles')
            
            # Store original attributes for output
            original_attrs = {k: v for k, v in attrs.items() 
                            if k not in ['rectangles', 'center']}  # rectangles and center will be updated
            
            module = Module(
                name=name,
                area=attrs.get('area', 0),
                aspect_ratio=attrs.get('aspect_ratio'),
                center=attrs.get('center', [0, 0]),
                io_pin=is_io_pin,
                fixed=is_fixed,
                hard=is_hard,
                rectangles=rectangles,
                original_attrs=original_attrs
            )
            self.modules[name] = module
        
        # Separate modules into categories:
        # 1. io_pins: terminals with fixed position, no area
        # 2. fixed_modules: regular modules with fixed position and size
        # 3. movable_modules: soft (area+aspect_ratio) + hard (fixed dimensions)
        movable_idx = 0
        fixed_idx = 0  # Index for fixed modules (used as anchors)
        
        for module in self.modules.values():
            if module.io_pin:
                # IO pins / terminals: fixed position, no area
                module.idx = fixed_idx
                self.io_pins.append(module)
                fixed_idx += 1
            elif module.fixed:
                # Fixed modules (non-io_pin): fixed position and size
                # These act as anchors in QP but are NOT io_pins
                module.idx = fixed_idx
                self.fixed_modules.append(module)
                fixed_idx += 1
            else:
                # Movable modules: soft or hard
                # - Soft: area and aspect_ratio defined, dimensions optimizable
                # - Hard: rectangles defined with hard=true, dimensions fixed
                module.idx = movable_idx
                self.movable_modules.append(module)
                movable_idx += 1
        
        # Parse nets
        nets_data = netlist_data.get('Nets', [])
        for net_entry in nets_data:
            if not isinstance(net_entry, (list, tuple)) or len(net_entry) < 2:
                continue
            
            # Check if last element is a weight (numeric, not a module name)
            net_modules = list(net_entry)
            weight = 1.0
            
            if len(net_modules) >= 2 and isinstance(net_modules[-1], (int, float)) and not isinstance(net_modules[-1], str):
                weight = float(net_modules[-1])
                net_modules = net_modules[:-1]  # Remove weight from module list
            
            # Create net with weight
            net = Net(net_modules, weight=weight)
            
            # Map module names to Module objects
            for mod_name in net_modules:
                mod_name_str = str(mod_name)
                if mod_name_str in self.modules:
                    module = self.modules[mod_name_str]
                    net.module_indices.append(module)
                    module.pins.append(len(self.nets))
            
            if len(net.module_indices) > 1:  # Only add nets with 2+ pins
                self.nets.append(net)
        
        # Count module types
        n_hard = sum(1 for m in self.movable_modules if m.hard)
        n_soft = len(self.movable_modules) - n_hard
        
        # Count nets with custom weights
        n_weighted = sum(1 for net in self.nets if abs(net.weight - 1.0) > 1e-9)
        
        print(f"Loaded {len(self.movable_modules)} movable modules "
              f"({n_soft} soft, {n_hard} hard)")
        print(f"Fixed modules: {len(self.fixed_modules)}, IO pins: {len(self.io_pins)}")
        print(f"Total nets: {len(self.nets)} ({n_weighted} with custom weights)")
        print(f"Die size: {self.die_width} x {self.die_height}")
    
    def save_yaml(self, output_file: str):
        """
        Save placement result to YAML format matching input format.
        Preserves original attributes (adj_cluster, boundary, mib, etc.)
        """
        modules_data = {}
        
        # Process all modules
        all_modules = (list(self.movable_modules) + 
                       list(self.fixed_modules) + 
                       list(self.io_pins))
        
        for module in all_modules:
            module_dict = {}
            
            # Start with original attributes (preserves adj_cluster, boundary, mib, etc.)
            if module.original_attrs:
                module_dict.update(module.original_attrs)
            
            # IO pins / terminals: only terminal + fixed flags, nested rectangles
            if module.io_pin:
                module_dict = {
                    'terminal': True,
                    'fixed': True,
                    'rectangles': [[
                        float(module.center[0]),
                        float(module.center[1]),
                        0,
                        0,
                    ]],
                }
            
            # Fixed modules (non-io_pin): position and size fixed
            elif module.fixed:
                if 'fixed' not in module_dict:
                    module_dict['fixed'] = True
                # Output rectangles with original position and dimensions
                if module.rectangles and len(module.rectangles) > 0:
                    module_dict['rectangles'] = [
                        [float(v) for v in rect] for rect in module.rectangles
                    ]
                else:
                    # Create rectangle from current state
                    module_dict['rectangles'] = [[
                        float(module.center[0]),
                        float(module.center[1]),
                        float(module.width),
                        float(module.height)
                    ]]
            
            # Hard modules: position movable, dimensions fixed
            elif module.hard:
                if 'hard' not in module_dict:
                    module_dict['hard'] = True
                # Output rectangles with updated center position, original dimensions
                if module.rectangles and len(module.rectangles) > 0:
                    updated_rects = []
                    trunk = module.rectangles[0]
                    # Update trunk with new center position
                    updated_rects.append([
                        float(module.center[0]),
                        float(module.center[1]),
                        float(trunk[2]),  # Keep original width
                        float(trunk[3])   # Keep original height
                    ])
                    # Keep other rectangles if present (branches)
                    for rect in module.rectangles[1:]:
                        updated_rects.append([float(v) for v in rect])
                    module_dict['rectangles'] = updated_rects
                else:
                    # Single rectangle for hard module
                    module_dict['rectangles'] = [[
                        float(module.center[0]),
                        float(module.center[1]),
                        float(module.width),
                        float(module.height)
                    ]]
            
            # Soft modules: position and dimensions optimizable
            else:
                # Update area if present
                if module.area > 0:
                    module_dict['area'] = float(module.area)
                
                # Add rectangles with optimized position and calculated dimensions
                rect_x = float(module.center[0])
                rect_y = float(module.center[1])
                rect_w = float(module.width)
                rect_h = float(module.height)
                module_dict['rectangles'] = [[rect_x, rect_y, rect_w, rect_h]]
            
            modules_data[module.name] = module_dict
        
        # Save nets (with weights)
        nets_data = []
        for net in self.nets:
            # Build net entry: [module1, module2, ..., weight]
            net_entry = [m.name for m in net.module_indices]
            # Always append weight to preserve the format
            net_entry.append(float(net.weight))
            nets_data.append(net_entry)
        
        output_data = {
            'Modules': modules_data,
            'Nets': nets_data
        }
        
        # Use custom YAML dumper to match input format (4-space indent)
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, 
                     default_flow_style=False, 
                     sort_keys=False, 
                     allow_unicode=True,
                     indent=4)
        
        print(f"\nSaved placement to {output_file}")
    
    def calc_hpwl(self) -> float:
        """Calculate weighted Half-Perimeter Wire Length"""
        hpwl = 0.0
        for net in self.nets:
            if len(net.module_indices) < 2:
                continue
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            for module in net.module_indices:
                min_x = min(min_x, module.center[0])
                max_x = max(max_x, module.center[0])
                min_y = min(min_y, module.center[1])
                max_y = max(max_y, module.center[1])
            # Apply net weight to HPWL
            hpwl += net.weight * ((max_x - min_x) + (max_y - min_y))
        return hpwl
    
    def clip_to_core(self, pos: np.ndarray, module: Module) -> np.ndarray:
        """Clip module center to valid core region"""
        # For hard modules, width and height are fixed
        # For soft modules, use calculated dimensions
        hw = module.width / 2
        hh = module.height / 2
        x = np.clip(pos[0], self.core_ll[0] + hw, self.core_ur[0] - hw)
        y = np.clip(pos[1], self.core_ll[1] + hh, self.core_ur[1] - hh)
        return np.array([x, y])


class QuadraticPlacer:
    """Initial quadratic placement solver"""
    def __init__(self, db: PlacementDB):
        self.db = db
    
    def solve(self, max_iters: int = 20, tolerance: float = 1e-6):
        """Solve quadratic placement Ax=b for x and y"""
        print("\n=== Quadratic Placement ===")
        qp_start_time = time.time()
        
        n_movable = len(self.db.movable_modules)
        
        if n_movable == 0:
            print("Warning: No movable modules to place!")
            return 0.0
        
        # Move all to center initially
        center = (self.db.core_ll + self.db.core_ur) / 2
        for module in self.db.movable_modules:
            module.center = center.copy()
        
        # Current solution vectors (clipped positions)
        x_sol = np.array([m.center[0] for m in self.db.movable_modules])
        y_sol = np.array([m.center[1] for m in self.db.movable_modules])
        
        initial_hpwl = self.db.calc_hpwl()
        print(f"Initial HPWL: {initial_hpwl:.2f}")
        
        # Track for stability detection
        prev_hpwl = initial_hpwl
        stable_count = 0
        use_default_tol = False  # Only warn once
        
        for iteration in range(max_iters):
            # Build matrices
            A_x, b_x = self._build_matrices('x')
            A_y, b_y = self._build_matrices('y')
            
            # Solve using BiCGSTAB
            try:
                x_new, info_x = bicgstab(A_x, b_x, x0=x_sol, tol=tolerance, maxiter=100)
                y_new, info_y = bicgstab(A_y, b_y, x0=y_sol, tol=tolerance, maxiter=100)
            except (TypeError, ValueError):
                # Fallback for older scipy versions
                x_new, info_x = bicgstab(A_x, b_x, x0=x_sol, maxiter=100)
                y_new, info_y = bicgstab(A_y, b_y, x0=y_sol, maxiter=100)
                if not use_default_tol:
                    print("Note: Using default solver tolerance")
                    use_default_tol = True
            
            # Clip positions to core boundaries
            for i, module in enumerate(self.db.movable_modules):
                new_pos = np.array([x_new[i], y_new[i]])
                module.center = self.db.clip_to_core(new_pos, module)
            
            # Get clipped positions for error calculation
            x_clipped = np.array([m.center[0] for m in self.db.movable_modules])
            y_clipped = np.array([m.center[1] for m in self.db.movable_modules])
            
            # Calculate error using CLIPPED positions (actual position change)
            x_err = np.linalg.norm(x_clipped - x_sol)
            y_err = np.linalg.norm(y_clipped - y_sol)
            error = max(x_err, y_err)
            hpwl = self.db.calc_hpwl()
            
            print(f"Iter {iteration:3d}: Error={error:.6e}, HPWL={hpwl:.2f}")
            
            # Update solution vectors with clipped positions
            x_sol = x_clipped
            y_sol = y_clipped
            
            # Convergence check 1: position change is small
            if error < tolerance and iteration > 4:
                print(f"Converged at iteration {iteration} (position change < {tolerance})")
                break
            
            # Convergence check 2: HPWL and positions are stable
            if abs(hpwl - prev_hpwl) < 1e-6 and error < 1e-6:
                stable_count += 1
                if stable_count >= 3:
                    print(f"Converged at iteration {iteration} (stable for 3 iterations)")
                    break
            else:
                stable_count = 0
            
            prev_hpwl = hpwl
        
        qp_end_time = time.time()
        qp_runtime = qp_end_time - qp_start_time
        final_hpwl = self.db.calc_hpwl()
        print(f"QP finished. Final HPWL: {final_hpwl:.2f}")
        print(f"QP CPU Time: {qp_runtime:.3f} seconds\n")
        
        return qp_runtime
    
    def _build_matrices(self, coord: str) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build sparse matrix A and vector b for coordinate 'x' or 'y'
        
        Module classification for QP:
        - Movable: soft and hard modules (position can be optimized)
        - Anchor: io_pins and fixed modules (position fixed, act as anchors)
        """
        n_movable = len(self.db.movable_modules)
        A = lil_matrix((n_movable, n_movable))
        b = np.zeros(n_movable)
        
        coord_idx = 0 if coord == 'x' else 1
        
        for net in self.db.nets:
            pin_count = len(net.module_indices)
            if pin_count < 2:
                continue
            
            # Use net weight × edge weight (following ePlace)
            # net.weight: user-defined importance of this net
            # 1/(pin_count-1): clique model edge weight
            weight = net.weight / (pin_count - 1)
            
            for i in range(pin_count):
                pin1 = net.module_indices[i]
                # Movable = not io_pin and not fixed
                # Hard modules ARE movable (just with fixed dimensions)
                is_pin1_movable = not (pin1.io_pin or pin1.fixed)
                
                for j in range(i + 1, pin_count):
                    pin2 = net.module_indices[j]
                    is_pin2_movable = not (pin2.io_pin or pin2.fixed)
                    
                    # Both movable (soft or hard)
                    if is_pin1_movable and is_pin2_movable:
                        idx1 = pin1.idx
                        idx2 = pin2.idx
                        A[idx1, idx1] += weight
                        A[idx2, idx2] += weight
                        A[idx1, idx2] -= weight
                        A[idx2, idx1] -= weight
                    
                    # pin1 is anchor (io_pin or fixed), pin2 movable
                    elif not is_pin1_movable and is_pin2_movable:
                        idx2 = pin2.idx
                        A[idx2, idx2] += weight
                        b[idx2] += weight * pin1.center[coord_idx]
                    
                    # pin1 movable, pin2 is anchor (io_pin or fixed)
                    elif is_pin1_movable and not is_pin2_movable:
                        idx1 = pin1.idx
                        A[idx1, idx1] += weight
                        b[idx1] += weight * pin2.center[coord_idx]
        
        return A.tocsr(), b


def visualize_placement(db: PlacementDB, output_image: str, title: str = "QP Placement") -> bool:
    """
    Visualize the placement result with different colors for module types.
    
    Colors:
    - Soft modules: blue
    - Hard modules: green
    - Fixed modules: orange
    - IO pins: red dots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw die boundary
        ax.add_patch(patches.Rectangle(
            (0, 0), db.die_width, db.die_height,
            fill=False, edgecolor='darkblue', linewidth=3, linestyle='--'
        ))
        
        # Color definitions
        colors = {
            'soft': 'skyblue',
            'hard': 'lightgreen',
            'fixed': 'orange',
            'io_pin': 'red'
        }
        
        # Draw modules
        # 1. Soft modules (blue)
        for module in db.movable_modules:
            if module.hard:
                continue
            x_left = module.center[0] - module.width / 2
            y_bottom = module.center[1] - module.height / 2
            rect = patches.Rectangle(
                (x_left, y_bottom), module.width, module.height,
                facecolor=colors['soft'], edgecolor='black', alpha=0.7, linewidth=1
            )
            ax.add_patch(rect)
            if module.width > 5 and module.height > 5:
                ax.text(module.center[0], module.center[1], module.name,
                       ha='center', va='center', fontsize=6, weight='bold')
        
        # 2. Hard modules (green)
        for module in db.movable_modules:
            if not module.hard:
                continue
            x_left = module.center[0] - module.width / 2
            y_bottom = module.center[1] - module.height / 2
            rect = patches.Rectangle(
                (x_left, y_bottom), module.width, module.height,
                facecolor=colors['hard'], edgecolor='black', alpha=0.7, linewidth=1
            )
            ax.add_patch(rect)
            if module.width > 5 and module.height > 5:
                ax.text(module.center[0], module.center[1], module.name,
                       ha='center', va='center', fontsize=6, weight='bold')
        
        # 3. Fixed modules (orange)
        for module in db.fixed_modules:
            x_left = module.center[0] - module.width / 2
            y_bottom = module.center[1] - module.height / 2
            rect = patches.Rectangle(
                (x_left, y_bottom), module.width, module.height,
                facecolor=colors['fixed'], edgecolor='darkred', alpha=0.7, linewidth=2
            )
            ax.add_patch(rect)
            if module.width > 5 and module.height > 5:
                ax.text(module.center[0], module.center[1], module.name,
                       ha='center', va='center', fontsize=6, weight='bold')
        
        # 4. IO pins (red dots)
        for module in db.io_pins:
            ax.plot(module.center[0], module.center[1], 'ro', markersize=5,
                   markeredgecolor='darkred', markeredgewidth=1)
        
        # Create legend
        legend_handles = [
            patches.Patch(facecolor=colors['soft'], edgecolor='black', label=f"Soft ({len([m for m in db.movable_modules if not m.hard])})"),
            patches.Patch(facecolor=colors['hard'], edgecolor='black', label=f"Hard ({len([m for m in db.movable_modules if m.hard])})"),
            patches.Patch(facecolor=colors['fixed'], edgecolor='darkred', label=f"Fixed ({len(db.fixed_modules)})"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markeredgecolor='darkred', markersize=8, label=f"IO Pins ({len(db.io_pins)})")
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
        
        # Calculate HPWL
        hpwl = db.calc_hpwl()
        
        ax.set_xlim(-0.05 * db.die_width, 1.05 * db.die_width)
        ax.set_ylim(-0.05 * db.die_height, 1.05 * db.die_height)
        ax.set_aspect('equal')
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title(f'{title}\nDie: {db.die_width:.1f} × {db.die_height:.1f}, HPWL: {hpwl:.2f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_image}")
        return True
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        return False


def verify_fixed_modules(db: PlacementDB, original_netlist_file: str) -> Tuple[bool, List[str]]:
    """
    Verify that fixed modules have not been moved.
    
    Returns:
        (all_ok, moved_modules): 
        - all_ok: True if all fixed modules are at original positions
        - moved_modules: List of module names that were moved (should be empty)
    """
    # Load original netlist
    with open(original_netlist_file, 'r', encoding='utf-8') as f:
        original_data = yaml.safe_load(f)
    
    original_modules = original_data.get('Modules', {})
    moved_modules = []
    tolerance = 1e-6
    
    print("\n=== Fixed Module Verification ===")
    
    for module in db.fixed_modules:
        original_attrs = original_modules.get(module.name, {})
        
        # Get original position
        original_center = None
        if 'center' in original_attrs:
            original_center = original_attrs['center']
        elif 'rectangles' in original_attrs and original_attrs['rectangles']:
            rect = original_attrs['rectangles'][0]
            if len(rect) >= 2:
                original_center = [rect[0], rect[1]]
        
        if original_center is None:
            print(f"  Warning: Cannot find original position for fixed module {module.name}")
            continue
        
        # Compare positions
        dx = abs(module.center[0] - original_center[0])
        dy = abs(module.center[1] - original_center[1])
        
        if dx > tolerance or dy > tolerance:
            moved_modules.append(module.name)
            print(f"  ❌ MOVED: {module.name}")
            print(f"     Original: ({original_center[0]:.4f}, {original_center[1]:.4f})")
            print(f"     Current:  ({module.center[0]:.4f}, {module.center[1]:.4f})")
            print(f"     Delta:    ({dx:.6f}, {dy:.6f})")
        else:
            print(f"  ✓ OK: {module.name} at ({module.center[0]:.4f}, {module.center[1]:.4f})")
    
    # Also verify IO pins (terminals)
    io_pin_moved = []
    for module in db.io_pins:
        original_attrs = original_modules.get(module.name, {})
        
        original_center = None
        if 'center' in original_attrs:
            original_center = original_attrs['center']
        elif 'rectangles' in original_attrs and original_attrs['rectangles']:
            rect = original_attrs['rectangles'][0]
            if len(rect) >= 2:
                original_center = [rect[0], rect[1]]
        
        if original_center is None:
            continue
        
        dx = abs(module.center[0] - original_center[0])
        dy = abs(module.center[1] - original_center[1])
        
        if dx > tolerance or dy > tolerance:
            io_pin_moved.append(module.name)
    
    if io_pin_moved:
        print(f"\n  ⚠️  Warning: {len(io_pin_moved)} IO pins were moved (should not happen)")
        for name in io_pin_moved[:5]:  # Show first 5
            print(f"     - {name}")
        if len(io_pin_moved) > 5:
            print(f"     ... and {len(io_pin_moved) - 5} more")
    
    all_ok = len(moved_modules) == 0 and len(io_pin_moved) == 0
    
    print(f"\n{'='*40}")
    if all_ok:
        print(f"✓ All fixed modules verified: positions unchanged")
    else:
        print(f"❌ Verification failed:")
        print(f"   - Fixed modules moved: {len(moved_modules)}")
        print(f"   - IO pins moved: {len(io_pin_moved)}")
    print(f"{'='*40}\n")
    
    return all_ok, moved_modules + io_pin_moved


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quadratic Placement (QP) standalone implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python quadratic_placement.py ami33.yaml ami33_die.yaml output.yaml
  
  # With custom iterations
  python quadratic_placement.py ami33.yaml ami33_die.yaml output.yaml --max-iters 30
  
  # With custom tolerance
  python quadratic_placement.py ami33.yaml ami33_die.yaml output.yaml --tolerance 1e-8
        """)
    
    parser.add_argument('netlist', type=str, help='Input netlist YAML file (e.g., ami33.yaml)')
    parser.add_argument('die', type=str, help='Input die YAML file (e.g., ami33_die.yaml)')
    parser.add_argument('output', type=str, help='Output placement YAML file')
    parser.add_argument('--max-iters', type=int, default=200,
                       help='Maximum iterations for QP solver (default: 200)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--output-image', type=str, default=None,
                       help='Output visualization image file (e.g., output.png)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify fixed modules were not moved')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.netlist):
        print(f"Error: Netlist file not found: {args.netlist}")
        sys.exit(1)
    
    if not os.path.exists(args.die):
        print(f"Error: Die file not found: {args.die}")
        sys.exit(1)
    
    # Create database and load data
    print(f"Loading netlist: {args.netlist}")
    print(f"Loading die: {args.die}")
    db = PlacementDB()
    db.load_yaml(args.netlist, args.die)
    
    # Run quadratic placement
    qp = QuadraticPlacer(db)
    qp_time = qp.solve(max_iters=args.max_iters, tolerance=args.tolerance)
    
    # Save result
    db.save_yaml(args.output)
    
    # Generate visualization if requested
    if args.output_image:
        visualize_placement(db, args.output_image, title="QP Placement Result")
    else:
        # Default: use output filename with .png extension
        default_image = os.path.splitext(args.output)[0] + '.png'
        visualize_placement(db, default_image, title="QP Placement Result")
    
    # Verify fixed modules if requested
    if args.verify:
        all_ok, moved = verify_fixed_modules(db, args.netlist)
        if not all_ok:
            print("Warning: Some fixed modules were moved!")
            sys.exit(1)
    
    # Print summary
    final_hpwl = db.calc_hpwl()
    n_hard = sum(1 for m in db.movable_modules if m.hard)
    n_soft = len(db.movable_modules) - n_hard
    
    print(f"\n{'='*50}")
    print(f"{'PLACEMENT SUMMARY':^50}")
    print(f"{'='*50}")
    print(f"Movable Modules:   {len(db.movable_modules):>15d}")
    print(f"  - Soft:          {n_soft:>15d} (area/aspect_ratio)")
    print(f"  - Hard:          {n_hard:>15d} (fixed dimensions)")
    print(f"Fixed Modules:     {len(db.fixed_modules):>15d} (position fixed)")
    print(f"IO Pins:           {len(db.io_pins):>15d} (terminals)")
    print(f"Final HPWL:        {final_hpwl:>15.2f}")
    print(f"QP CPU Time:       {qp_time:>15.3f} seconds")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()

