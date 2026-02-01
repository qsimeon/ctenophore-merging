"""Core computational model for ctenophore merging after dissection.

This module provides the fundamental classes and functions for simulating
the biological process of ctenophore fragments merging back together after
dissection. It includes models for tissue dynamics, cell behavior, and
merging mechanics.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class MergeState(Enum):
    """Enumeration of possible merging states for ctenophore fragments."""
    SEPARATED = "separated"
    APPROACHING = "approaching"
    CONTACT = "contact"
    FUSING = "fusing"
    MERGED = "merged"


@dataclass
class CtenophoreFragment:
    """Represents a single ctenophore fragment after dissection.
    
    Attributes:
        fragment_id: Unique identifier for the fragment
        position: 3D position coordinates (x, y, z)
        velocity: 3D velocity vector
        mass: Mass of the fragment (arbitrary units)
        surface_area: Surface area available for merging
        cell_count: Number of cells in the fragment
        vitality: Health/viability score (0.0 to 1.0)
        adhesion_strength: Strength of adhesive properties (0.0 to 1.0)
        state: Current merging state
    """
    fragment_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    surface_area: float = 1.0
    cell_count: int = 1000
    vitality: float = 1.0
    adhesion_strength: float = 0.8
    state: MergeState = MergeState.SEPARATED
    
    def __post_init__(self):
        """Ensure position and velocity are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)


class CtenophoreMergeModel:
    """Main computational model for simulating ctenophore fragment merging.
    
    This class handles the physics and biology of fragment interactions,
    including attraction forces, collision detection, and fusion dynamics.
    """
    
    def __init__(self,
                 attraction_coefficient: float = 0.5,
                 contact_distance: float = 0.1,
                 fusion_threshold: float = 0.7,
                 time_step: float = 0.01,
                 viscosity: float = 0.1):
        """Initialize the merge model with physical parameters.
        
        Args:
            attraction_coefficient: Strength of chemical attraction between fragments
            contact_distance: Distance threshold for contact detection
            fusion_threshold: Minimum adhesion product for fusion to occur
            time_step: Simulation time step (seconds)
            viscosity: Environmental viscosity coefficient
        """
        self.attraction_coefficient = attraction_coefficient
        self.contact_distance = contact_distance
        self.fusion_threshold = fusion_threshold
        self.time_step = time_step
        self.viscosity = viscosity
        self.fragments: List[CtenophoreFragment] = []
        self.time_elapsed: float = 0.0
        self.merge_history: List[Dict[str, Any]] = []
    
    def add_fragment(self, fragment: CtenophoreFragment) -> None:
        """Add a fragment to the simulation.
        
        Args:
            fragment: CtenophoreFragment instance to add
        """
        self.fragments.append(fragment)
    
    def calculate_attraction_force(self,
                                   frag1: CtenophoreFragment,
                                   frag2: CtenophoreFragment) -> np.ndarray:
        """Calculate chemical attraction force between two fragments.
        
        The force is proportional to the product of adhesion strengths and
        inversely proportional to distance squared.
        
        Args:
            frag1: First fragment
            frag2: Second fragment
            
        Returns:
            3D force vector acting on frag1 due to frag2
        """
        displacement = frag2.position - frag1.position
        distance = np.linalg.norm(displacement)
        
        if distance < 1e-6:
            return np.zeros(3)
        
        direction = displacement / distance
        
        # Chemical attraction force (modified inverse square law)
        force_magnitude = (self.attraction_coefficient * 
                          frag1.adhesion_strength * 
                          frag2.adhesion_strength * 
                          frag1.vitality * 
                          frag2.vitality / 
                          (distance ** 2 + 0.01))  # Regularization term
        
        return force_magnitude * direction
    
    def calculate_drag_force(self, fragment: CtenophoreFragment) -> np.ndarray:
        """Calculate viscous drag force on a fragment.
        
        Args:
            fragment: Fragment experiencing drag
            
        Returns:
            3D drag force vector
        """
        return -self.viscosity * fragment.surface_area * fragment.velocity
    
    def update_fragment_state(self, fragment: CtenophoreFragment) -> None:
        """Update the merging state of a fragment based on proximity to others.
        
        Args:
            fragment: Fragment to update
        """
        if fragment.state == MergeState.MERGED:
            return
        
        min_distance = float('inf')
        for other in self.fragments:
            if other.fragment_id != fragment.fragment_id and other.state != MergeState.MERGED:
                distance = np.linalg.norm(fragment.position - other.position)
                min_distance = min(min_distance, distance)
        
        if min_distance <= self.contact_distance:
            if fragment.state == MergeState.CONTACT:
                fragment.state = MergeState.FUSING
            else:
                fragment.state = MergeState.CONTACT
        elif min_distance <= self.contact_distance * 3:
            fragment.state = MergeState.APPROACHING
        else:
            fragment.state = MergeState.SEPARATED
    
    def attempt_merge(self, frag1: CtenophoreFragment, frag2: CtenophoreFragment) -> Optional[CtenophoreFragment]:
        """Attempt to merge two fragments if conditions are met.
        
        Args:
            frag1: First fragment
            frag2: Second fragment
            
        Returns:
            New merged fragment if successful, None otherwise
        """
        distance = np.linalg.norm(frag1.position - frag2.position)
        
        if distance > self.contact_distance:
            return None
        
        # Check if adhesion is strong enough
        adhesion_product = frag1.adhesion_strength * frag2.adhesion_strength
        if adhesion_product < self.fusion_threshold:
            return None
        
        # Check if both fragments are in fusing state
        if frag1.state != MergeState.FUSING or frag2.state != MergeState.FUSING:
            return None
        
        # Create merged fragment
        total_mass = frag1.mass + frag2.mass
        new_position = (frag1.position * frag1.mass + frag2.position * frag2.mass) / total_mass
        new_velocity = (frag1.velocity * frag1.mass + frag2.velocity * frag2.mass) / total_mass
        
        merged = CtenophoreFragment(
            fragment_id=max(frag1.fragment_id, frag2.fragment_id) * 1000 + min(frag1.fragment_id, frag2.fragment_id),
            position=new_position,
            velocity=new_velocity,
            mass=total_mass,
            surface_area=frag1.surface_area + frag2.surface_area * 0.8,  # Some overlap
            cell_count=frag1.cell_count + frag2.cell_count,
            vitality=min(1.0, (frag1.vitality + frag2.vitality) / 2 + 0.1),  # Slight boost
            adhesion_strength=(frag1.adhesion_strength + frag2.adhesion_strength) / 2,
            state=MergeState.MERGED
        )
        
        # Record merge event
        self.merge_history.append({
            'time': self.time_elapsed,
            'fragment1_id': frag1.fragment_id,
            'fragment2_id': frag2.fragment_id,
            'merged_id': merged.fragment_id,
            'position': merged.position.copy()
        })
        
        return merged
    
    def step(self) -> None:
        """Advance the simulation by one time step.
        
        Updates positions, velocities, and checks for merging events.
        """
        # Update forces and velocities
        forces = {frag.fragment_id: np.zeros(3) for frag in self.fragments}
        
        for i, frag1 in enumerate(self.fragments):
            if frag1.state == MergeState.MERGED:
                continue
            
            # Calculate attraction forces from all other fragments
            for j, frag2 in enumerate(self.fragments):
                if i != j and frag2.state != MergeState.MERGED:
                    forces[frag1.fragment_id] += self.calculate_attraction_force(frag1, frag2)
            
            # Add drag force
            forces[frag1.fragment_id] += self.calculate_drag_force(frag1)
        
        # Update velocities and positions
        for frag in self.fragments:
            if frag.state != MergeState.MERGED:
                acceleration = forces[frag.fragment_id] / frag.mass
                frag.velocity += acceleration * self.time_step
                frag.position += frag.velocity * self.time_step
                
                # Update state based on proximity
                self.update_fragment_state(frag)
        
        # Check for merging
        merged_fragments = []
        fragments_to_remove = set()
        
        for i, frag1 in enumerate(self.fragments):
            if frag1.fragment_id in fragments_to_remove:
                continue
            for j, frag2 in enumerate(self.fragments[i+1:], start=i+1):
                if frag2.fragment_id in fragments_to_remove:
                    continue
                
                merged = self.attempt_merge(frag1, frag2)
                if merged is not None:
                    merged_fragments.append(merged)
                    fragments_to_remove.add(frag1.fragment_id)
                    fragments_to_remove.add(frag2.fragment_id)
                    break
        
        # Remove merged fragments and add new ones
        self.fragments = [f for f in self.fragments if f.fragment_id not in fragments_to_remove]
        self.fragments.extend(merged_fragments)
        
        # Decay vitality slightly over time
        for frag in self.fragments:
            frag.vitality *= 0.9999
        
        self.time_elapsed += self.time_step
    
    def run_simulation(self, duration: float, callback: Optional[callable] = None) -> None:
        """Run the simulation for a specified duration.
        
        Args:
            duration: Total simulation time (seconds)
            callback: Optional function called after each step with model state
        """
        steps = int(duration / self.time_step)
        
        for _ in range(steps):
            self.step()
            if callback is not None:
                callback(self)
    
    def get_fragment_count(self) -> int:
        """Get the current number of active fragments.
        
        Returns:
            Number of fragments not in MERGED state
        """
        return sum(1 for f in self.fragments if f.state != MergeState.MERGED)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get a snapshot of the current system state.
        
        Returns:
            Dictionary containing system metrics and fragment data
        """
        active_fragments = [f for f in self.fragments if f.state != MergeState.MERGED]
        
        return {
            'time': self.time_elapsed,
            'fragment_count': len(active_fragments),
            'total_mass': sum(f.mass for f in active_fragments),
            'total_cells': sum(f.cell_count for f in active_fragments),
            'average_vitality': np.mean([f.vitality for f in active_fragments]) if active_fragments else 0.0,
            'merge_events': len(self.merge_history),
            'fragments': [
                {
                    'id': f.fragment_id,
                    'position': f.position.tolist(),
                    'velocity': f.velocity.tolist(),
                    'mass': f.mass,
                    'state': f.state.value
                }
                for f in active_fragments
            ]
        }
