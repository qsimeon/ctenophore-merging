"""Utility functions for ctenophore merging simulation.

This module provides helper functions for data processing, visualization
preparation, analysis, and I/O operations for the ctenophore merge model.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
from pathlib import Path


def generate_initial_fragments(num_fragments: int,
                              spatial_extent: float = 10.0,
                              mass_range: Tuple[float, float] = (0.5, 2.0),
                              random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate initial fragment configurations for simulation.
    
    Args:
        num_fragments: Number of fragments to generate
        spatial_extent: Size of the spatial region for initial positions
        mass_range: Tuple of (min_mass, max_mass) for fragment masses
        random_seed: Optional seed for reproducibility
        
    Returns:
        List of dictionaries containing fragment initialization parameters
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    fragments = []
    for i in range(num_fragments):
        position = np.random.uniform(-spatial_extent/2, spatial_extent/2, size=3)
        mass = np.random.uniform(*mass_range)
        
        fragments.append({
            'fragment_id': i,
            'position': position,
            'mass': mass,
            'surface_area': mass ** (2/3),  # Approximate scaling
            'cell_count': int(mass * 1000),
            'vitality': np.random.uniform(0.8, 1.0),
            'adhesion_strength': np.random.uniform(0.6, 0.9)
        })
    
    return fragments


def calculate_center_of_mass(fragments: List[Any]) -> np.ndarray:
    """Calculate the center of mass of all fragments.
    
    Args:
        fragments: List of CtenophoreFragment objects
        
    Returns:
        3D position vector of center of mass
    """
    if not fragments:
        return np.zeros(3)
    
    total_mass = sum(f.mass for f in fragments)
    if total_mass == 0:
        return np.zeros(3)
    
    com = sum(f.position * f.mass for f in fragments) / total_mass
    return com


def calculate_system_energy(fragments: List[Any],
                           attraction_coefficient: float = 0.5) -> Dict[str, float]:
    """Calculate kinetic and potential energy of the system.
    
    Args:
        fragments: List of CtenophoreFragment objects
        attraction_coefficient: Attraction force coefficient from model
        
    Returns:
        Dictionary with 'kinetic', 'potential', and 'total' energy values
    """
    kinetic_energy = 0.0
    potential_energy = 0.0
    
    # Calculate kinetic energy
    for frag in fragments:
        velocity_squared = np.dot(frag.velocity, frag.velocity)
        kinetic_energy += 0.5 * frag.mass * velocity_squared
    
    # Calculate potential energy (pairwise interactions)
    for i, frag1 in enumerate(fragments):
        for frag2 in fragments[i+1:]:
            distance = np.linalg.norm(frag1.position - frag2.position)
            if distance > 1e-6:
                # Potential energy from attraction (negative for attractive force)
                potential_energy -= (attraction_coefficient * 
                                    frag1.adhesion_strength * 
                                    frag2.adhesion_strength / 
                                    distance)
    
    return {
        'kinetic': kinetic_energy,
        'potential': potential_energy,
        'total': kinetic_energy + potential_energy
    }


def calculate_fragment_distances(fragments: List[Any]) -> np.ndarray:
    """Calculate pairwise distances between all fragments.
    
    Args:
        fragments: List of CtenophoreFragment objects
        
    Returns:
        2D numpy array of distances (symmetric matrix)
    """
    n = len(fragments)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(fragments[i].position - fragments[j].position)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def analyze_merge_timeline(merge_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the timeline of merge events.
    
    Args:
        merge_history: List of merge event dictionaries from model
        
    Returns:
        Dictionary containing merge statistics and timeline analysis
    """
    if not merge_history:
        return {
            'total_merges': 0,
            'first_merge_time': None,
            'last_merge_time': None,
            'average_merge_interval': None,
            'merge_rate': 0.0
        }
    
    times = [event['time'] for event in merge_history]
    
    analysis = {
        'total_merges': len(merge_history),
        'first_merge_time': min(times),
        'last_merge_time': max(times),
        'merge_times': times
    }
    
    if len(times) > 1:
        intervals = np.diff(sorted(times))
        analysis['average_merge_interval'] = float(np.mean(intervals))
        analysis['merge_interval_std'] = float(np.std(intervals))
    else:
        analysis['average_merge_interval'] = None
        analysis['merge_interval_std'] = None
    
    # Calculate merge rate (merges per unit time)
    if analysis['last_merge_time'] > 0:
        analysis['merge_rate'] = len(merge_history) / analysis['last_merge_time']
    else:
        analysis['merge_rate'] = 0.0
    
    return analysis


def export_simulation_data(model: Any,
                          output_path: str,
                          include_history: bool = True) -> None:
    """Export simulation data to JSON file.
    
    Args:
        model: CtenophoreMergeModel instance
        output_path: Path to output JSON file
        include_history: Whether to include full merge history
    """
    data = {
        'parameters': {
            'attraction_coefficient': model.attraction_coefficient,
            'contact_distance': model.contact_distance,
            'fusion_threshold': model.fusion_threshold,
            'time_step': model.time_step,
            'viscosity': model.viscosity
        },
        'final_state': model.get_system_state(),
        'merge_statistics': analyze_merge_timeline(model.merge_history)
    }
    
    if include_history:
        data['merge_history'] = model.merge_history
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_simulation_data(input_path: str) -> Dict[str, Any]:
    """Load simulation data from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary containing simulation data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_merge_efficiency(initial_fragments: int,
                              final_fragments: int,
                              time_elapsed: float) -> Dict[str, float]:
    """Calculate efficiency metrics for the merging process.
    
    Args:
        initial_fragments: Number of fragments at start
        final_fragments: Number of fragments at end
        time_elapsed: Total simulation time
        
    Returns:
        Dictionary with efficiency metrics
    """
    if initial_fragments <= 1:
        return {
            'merge_percentage': 0.0,
            'merge_rate': 0.0,
            'efficiency_score': 0.0
        }
    
    merges_occurred = initial_fragments - final_fragments
    max_possible_merges = initial_fragments - 1
    
    merge_percentage = (merges_occurred / max_possible_merges) * 100
    merge_rate = merges_occurred / time_elapsed if time_elapsed > 0 else 0.0
    
    # Efficiency score: balance between merge percentage and speed
    efficiency_score = merge_percentage * (1.0 / (1.0 + time_elapsed))
    
    return {
        'merge_percentage': merge_percentage,
        'merge_rate': merge_rate,
        'efficiency_score': efficiency_score,
        'merges_occurred': merges_occurred,
        'max_possible_merges': max_possible_merges
    }


def prepare_trajectory_data(state_history: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Prepare trajectory data for visualization from state history.
    
    Args:
        state_history: List of system state dictionaries over time
        
    Returns:
        Dictionary mapping fragment IDs to trajectory arrays (time x 3)
    """
    trajectories = {}
    
    for state in state_history:
        for fragment in state.get('fragments', []):
            frag_id = fragment['id']
            position = np.array(fragment['position'])
            
            if frag_id not in trajectories:
                trajectories[frag_id] = []
            
            trajectories[frag_id].append(position)
    
    # Convert lists to numpy arrays
    for frag_id in trajectories:
        trajectories[frag_id] = np.array(trajectories[frag_id])
    
    return trajectories


def calculate_spatial_statistics(fragments: List[Any]) -> Dict[str, Any]:
    """Calculate spatial distribution statistics of fragments.
    
    Args:
        fragments: List of CtenophoreFragment objects
        
    Returns:
        Dictionary with spatial statistics
    """
    if not fragments:
        return {
            'mean_position': np.zeros(3),
            'std_position': np.zeros(3),
            'spatial_extent': 0.0,
            'mean_distance_from_center': 0.0
        }
    
    positions = np.array([f.position for f in fragments])
    center = calculate_center_of_mass(fragments)
    
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    
    # Calculate spatial extent (max distance between any two fragments)
    max_extent = 0.0
    for i, frag1 in enumerate(fragments):
        for frag2 in fragments[i+1:]:
            dist = np.linalg.norm(frag1.position - frag2.position)
            max_extent = max(max_extent, dist)
    
    # Mean distance from center of mass
    distances_from_center = [np.linalg.norm(f.position - center) for f in fragments]
    mean_dist_from_center = np.mean(distances_from_center)
    
    return {
        'mean_position': mean_pos,
        'std_position': std_pos,
        'spatial_extent': max_extent,
        'mean_distance_from_center': mean_dist_from_center,
        'center_of_mass': center
    }


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector (or zero vector if input is zero)
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm


def _calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians [0, π]
    """
    v1_norm = _normalize_vector(v1)
    v2_norm = _normalize_vector(v2)
    
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot_product)
