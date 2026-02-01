#!/usr/bin/env python3
"""
Computational Model of Ctenophore Merging After Dissection - Demo Script

This demo script demonstrates the complete functionality of the ctenophore
merging simulation model. It showcases:
1. Fragment generation and initialization
2. Simulation execution with different parameters
3. Analysis of merge events and system dynamics
4. Visualization of results
5. Data export and import capabilities

Ctenophores (comb jellies) have remarkable regenerative abilities. When dissected
into fragments, these pieces can merge back together to form functional organisms.
This model simulates the physical and biological processes involved in this merging.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json

# Import from the lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from core import MergeState, CtenophoreFragment, CtenophoreMergeModel
from utils import (
    generate_initial_fragments,
    calculate_center_of_mass,
    calculate_system_energy,
    calculate_fragment_distances,
    analyze_merge_timeline,
    export_simulation_data,
    load_simulation_data,
    calculate_merge_efficiency,
    prepare_trajectory_data,
    calculate_spatial_statistics
)


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def visualize_simulation_results(model: CtenophoreMergeModel, 
                                 state_history: List[Dict[str, Any]],
                                 merge_history: List[Dict[str, Any]]) -> None:
    """
    Create comprehensive visualizations of simulation results.
    
    Args:
        model: The simulation model
        state_history: History of system states
        merge_history: History of merge events
    """
    print_section_header("Generating Visualizations")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Fragment trajectories in 3D space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    trajectory_data = prepare_trajectory_data(state_history)
    
    for fragment_id, trajectory in trajectory_data.items():
        if len(trajectory) > 0:
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    alpha=0.6, linewidth=1.5, label=f'Frag {fragment_id}')
    
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_zlabel('Z Position (mm)')
    ax1.set_title('Fragment Trajectories in 3D Space')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    
    # 2. Number of fragments over time
    ax2 = fig.add_subplot(2, 3, 2)
    time_steps = [state['time'] for state in state_history]
    fragment_counts = [len(state['fragments']) for state in state_history]
    
    ax2.plot(time_steps, fragment_counts, 'b-', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Number of Fragments')
    ax2.set_title('Fragment Count Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Mark merge events
    for merge_event in merge_history:
        ax2.axvline(x=merge_event['time'], color='r', linestyle='--', alpha=0.5)
    
    # 3. System energy over time
    ax3 = fig.add_subplot(2, 3, 3)
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    
    for state in state_history:
        fragments = state['fragments']
        # Reconstruct fragment objects for energy calculation
        frag_objects = []
        for frag_data in fragments:
            frag = CtenophoreFragment(
                fragment_id=frag_data['id'],
                position=np.array(frag_data['position']),
                velocity=np.array(frag_data['velocity']),
                mass=frag_data['mass']
            )
            frag_objects.append(frag)
        
        energy = calculate_system_energy(frag_objects, model.attraction_coefficient)
        kinetic_energies.append(energy['kinetic'])
        potential_energies.append(energy['potential'])
        total_energies.append(energy['total'])
    
    ax3.plot(time_steps, kinetic_energies, 'r-', label='Kinetic', linewidth=2)
    ax3.plot(time_steps, potential_energies, 'b-', label='Potential', linewidth=2)
    ax3.plot(time_steps, total_energies, 'g-', label='Total', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Energy (arbitrary units)')
    ax3.set_title('System Energy Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Fragment distances heatmap (initial vs final)
    ax4 = fig.add_subplot(2, 3, 4)
    
    if len(state_history) > 0:
        initial_state = state_history[0]
        initial_frags = []
        for frag_data in initial_state['fragments']:
            frag = CtenophoreFragment(
                fragment_id=frag_data['id'],
                position=np.array(frag_data['position']),
                velocity=np.array(frag_data['velocity']),
                mass=frag_data['mass']
            )
            initial_frags.append(frag)
        
        initial_distances = calculate_fragment_distances(initial_frags)
        im = ax4.imshow(initial_distances, cmap='viridis', aspect='auto')
        ax4.set_xlabel('Fragment Index')
        ax4.set_ylabel('Fragment Index')
        ax4.set_title('Initial Pairwise Distances (mm)')
        plt.colorbar(im, ax=ax4)
    
    # 5. Merge event timeline
    ax5 = fig.add_subplot(2, 3, 5)
    
    if merge_history:
        merge_times = [event['time'] for event in merge_history]
        merge_indices = list(range(len(merge_times)))
        
        ax5.scatter(merge_times, merge_indices, c='red', s=100, alpha=0.6, marker='o')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Merge Event Index')
        ax5.set_title(f'Merge Events Timeline ({len(merge_history)} events)')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No merge events occurred', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Merge Events Timeline')
    
    # 6. Center of mass trajectory
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    com_trajectory = []
    for state in state_history:
        fragments = []
        for frag_data in state['fragments']:
            frag = CtenophoreFragment(
                fragment_id=frag_data['id'],
                position=np.array(frag_data['position']),
                velocity=np.array(frag_data['velocity']),
                mass=frag_data['mass']
            )
            fragments.append(frag)
        
        com = calculate_center_of_mass(fragments)
        com_trajectory.append(com)
    
    com_trajectory = np.array(com_trajectory)
    ax6.plot(com_trajectory[:, 0], com_trajectory[:, 1], com_trajectory[:, 2], 
            'r-', linewidth=2, label='Center of Mass')
    ax6.scatter(com_trajectory[0, 0], com_trajectory[0, 1], com_trajectory[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax6.scatter(com_trajectory[-1, 0], com_trajectory[-1, 1], com_trajectory[-1, 2], 
               c='r', s=100, marker='s', label='End')
    ax6.set_xlabel('X Position (mm)')
    ax6.set_ylabel('Y Position (mm)')
    ax6.set_zlabel('Z Position (mm)')
    ax6.set_title('System Center of Mass Trajectory')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('ctenophore_merge_simulation.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to 'ctenophore_merge_simulation.png'")
    
    # Show the plot
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except:
        print("  (Display not available, but file saved)")


def run_basic_simulation() -> tuple:
    """
    Run a basic simulation with default parameters.
    
    Returns:
        Tuple of (model, state_history, merge_history)
    """
    print_section_header("Basic Simulation: 5 Fragments")
    
    # Generate initial fragments
    print("\n1. Generating initial fragments...")
    initial_configs = generate_initial_fragments(
        num_fragments=5,
        spatial_extent=15.0,
        mass_range=(0.8, 1.5),
        random_seed=42
    )
    print(f"   ✓ Generated {len(initial_configs)} fragments")
    
    # Create model
    print("\n2. Initializing simulation model...")
    model = CtenophoreMergeModel(
        attraction_coefficient=0.5,
        drag_coefficient=0.1,
        merge_threshold=1.0,
        time_step=0.01
    )
    
    # Add fragments to model
    for config in initial_configs:
        model.add_fragment(
            position=config['position'],
            velocity=config['velocity'],
            mass=config['mass']
        )
    print(f"   ✓ Added {model.get_fragment_count()} fragments to model")
    
    # Display initial spatial statistics
    print("\n3. Initial spatial statistics:")
    initial_stats = calculate_spatial_statistics(model.fragments)
    print(f"   - Mean position: {initial_stats['mean_position']}")
    print(f"   - Spatial spread (std): {initial_stats['spatial_spread']:.3f} mm")
    print(f"   - Max distance from center: {initial_stats['max_distance_from_center']:.3f} mm")
    
    # Run simulation
    print("\n4. Running simulation...")
    print("   (This may take a moment...)")
    state_history, merge_history = model.run_simulation(
        max_time=50.0,
        record_interval=0.5
    )
    print(f"   ✓ Simulation complete!")
    print(f"   - Total time steps: {len(state_history)}")
    print(f"   - Merge events: {len(merge_history)}")
    print(f"   - Final fragment count: {model.get_fragment_count()}")
    
    return model, state_history, merge_history


def run_advanced_simulation() -> tuple:
    """
    Run an advanced simulation with more fragments and custom parameters.
    
    Returns:
        Tuple of (model, state_history, merge_history)
    """
    print_section_header("Advanced Simulation: 8 Fragments with Custom Parameters")
    
    # Generate initial fragments
    print("\n1. Generating initial fragments...")
    initial_configs = generate_initial_fragments(
        num_fragments=8,
        spatial_extent=20.0,
        mass_range=(0.5, 2.0),
        random_seed=123
    )
    print(f"   ✓ Generated {len(initial_configs)} fragments")
    
    # Create model with custom parameters
    print("\n2. Initializing simulation model with custom parameters...")
    model = CtenophoreMergeModel(
        attraction_coefficient=0.8,  # Stronger attraction
        drag_coefficient=0.15,        # More drag
        merge_threshold=0.8,          # Easier merging
        time_step=0.01
    )
    
    # Add fragments
    for config in initial_configs:
        model.add_fragment(
            position=config['position'],
            velocity=config['velocity'],
            mass=config['mass']
        )
    print(f"   ✓ Added {model.get_fragment_count()} fragments to model")
    
    # Run simulation
    print("\n3. Running simulation...")
    state_history, merge_history = model.run_simulation(
        max_time=60.0,
        record_interval=0.5
    )
    print(f"   ✓ Simulation complete!")
    print(f"   - Final fragment count: {model.get_fragment_count()}")
    
    return model, state_history, merge_history


def analyze_simulation_results(model: CtenophoreMergeModel,
                               state_history: List[Dict[str, Any]],
                               merge_history: List[Dict[str, Any]],
                               initial_fragment_count: int) -> None:
    """
    Perform comprehensive analysis of simulation results.
    
    Args:
        model: The simulation model
        state_history: History of system states
        merge_history: History of merge events
        initial_fragment_count: Number of fragments at start
    """
    print_section_header("Simulation Analysis")
    
    # 1. Merge timeline analysis
    print("\n1. Merge Timeline Analysis:")
    if merge_history:
        timeline_analysis = analyze_merge_timeline(merge_history)
        print(f"   - Total merge events: {timeline_analysis['total_merges']}")
        print(f"   - First merge at: {timeline_analysis['first_merge_time']:.2f} s")
        print(f"   - Last merge at: {timeline_analysis['last_merge_time']:.2f} s")
        print(f"   - Average time between merges: {timeline_analysis['average_merge_interval']:.2f} s")
        print(f"   - Merge rate: {timeline_analysis['merge_rate']:.4f} merges/s")
    else:
        print("   - No merge events occurred")
    
    # 2. Merge efficiency
    print("\n2. Merge Efficiency Metrics:")
    final_time = state_history[-1]['time'] if state_history else 0
    efficiency = calculate_merge_efficiency(
        initial_fragments=initial_fragment_count,
        final_fragments=model.get_fragment_count(),
        time_elapsed=final_time
    )
    print(f"   - Merge completion: {efficiency['merge_completion']:.1f}%")
    print(f"   - Fragments merged: {efficiency['fragments_merged']}")
    print(f"   - Merge rate: {efficiency['merge_rate']:.4f} merges/s")
    
    # 3. Final spatial statistics
    print("\n3. Final Spatial Configuration:")
    if model.fragments:
        final_stats = calculate_spatial_statistics(model.fragments)
        print(f"   - Mean position: {final_stats['mean_position']}")
        print(f"   - Spatial spread: {final_stats['spatial_spread']:.3f} mm")
        print(f"   - Max distance from center: {final_stats['max_distance_from_center']:.3f} mm")
        print(f"   - Bounding box volume: {final_stats['bounding_box_volume']:.3f} mm³")
    
    # 4. Energy analysis
    print("\n4. Energy Analysis:")
    if state_history:
        initial_state = state_history[0]
        final_state = state_history[-1]
        
        # Calculate initial energy
        initial_frags = []
        for frag_data in initial_state['fragments']:
            frag = CtenophoreFragment(
                fragment_id=frag_data['id'],
                position=np.array(frag_data['position']),
                velocity=np.array(frag_data['velocity']),
                mass=frag_data['mass']
            )
            initial_frags.append(frag)
        
        initial_energy = calculate_system_energy(initial_frags, model.attraction_coefficient)
        
        # Calculate final energy
        final_frags = []
        for frag_data in final_state['fragments']:
            frag = CtenophoreFragment(
                fragment_id=frag_data['id'],
                position=np.array(frag_data['position']),
                velocity=np.array(frag_data['velocity']),
                mass=frag_data['mass']
            )
            final_frags.append(frag)
        
        final_energy = calculate_system_energy(final_frags, model.attraction_coefficient)
        
        print(f"   Initial Energy:")
        print(f"     - Kinetic: {initial_energy['kinetic']:.4f}")
        print(f"     - Potential: {initial_energy['potential']:.4f}")
        print(f"     - Total: {initial_energy['total']:.4f}")
        print(f"   Final Energy:")
        print(f"     - Kinetic: {final_energy['kinetic']:.4f}")
        print(f"     - Potential: {final_energy['potential']:.4f}")
        print(f"     - Total: {final_energy['total']:.4f}")
        print(f"   Energy dissipated: {initial_energy['total'] - final_energy['total']:.4f}")


def demonstrate_data_export_import(model: CtenophoreMergeModel) -> None:
    """
    Demonstrate data export and import functionality.
    
    Args:
        model: The simulation model to export
    """
    print_section_header("Data Export/Import Demonstration")
    
    # Export data
    print("\n1. Exporting simulation data...")
    output_file = "ctenophore_simulation_data.json"
    export_simulation_data(model, output_file, include_history=True)
    print(f"   ✓ Data exported to '{output_file}'")
    
    # Load data back
    print("\n2. Loading simulation data...")
    loaded_data = load_simulation_data(output_file)
    print(f"   ✓ Data loaded successfully")
    print(f"   - Fragments in loaded data: {len(loaded_data['fragments'])}")
    print(f"   - State history entries: {len(loaded_data['state_history'])}")
    print(f"   - Merge events: {len(loaded_data['merge_history'])}")
    
    # Display some loaded data
    print("\n3. Sample of loaded data:")
    if loaded_data['fragments']:
        first_frag = loaded_data['fragments'][0]
        print(f"   First fragment:")
        print(f"     - ID: {first_frag['id']}")
        print(f"     - Position: {first_frag['position']}")
        print(f"     - Mass: {first_frag['mass']:.3f}")
        print(f"     - State: {first_frag['state']}")


def compare_parameter_effects() -> None:
    """
    Compare the effects of different simulation parameters.
    """
    print_section_header("Parameter Comparison Study")
    
    print("\nComparing different attraction coefficients...")
    print("(Running 3 simulations with varying parameters)\n")
    
    attraction_values = [0.3, 0.6, 0.9]
    results = []
    
    for i, attraction in enumerate(attraction_values, 1):
        print(f"Simulation {i}/3: attraction_coefficient = {attraction}")
        
        # Generate fragments
        configs = generate_initial_fragments(
            num_fragments=6,
            spatial_extent=15.0,
            random_seed=42  # Same seed for fair comparison
        )
        
        # Create and run model
        model = CtenophoreMergeModel(
            attraction_coefficient=attraction,
            drag_coefficient=0.1,
            merge_threshold=1.0,
            time_step=0.01
        )
        
        for config in configs:
            model.add_fragment(
                position=config['position'],
                velocity=config['velocity'],
                mass=config['mass']
            )
        
        state_history, merge_history = model.run_simulation(
            max_time=40.0,
            record_interval=1.0
        )
        
        # Calculate efficiency
        efficiency = calculate_merge_efficiency(
            initial_fragments=6,
            final_fragments=model.get_fragment_count(),
            time_elapsed=state_history[-1]['time']
        )
        
        results.append({
            'attraction': attraction,
            'final_fragments': model.get_fragment_count(),
            'merge_events': len(merge_history),
            'completion': efficiency['merge_completion']
        })
        
        print(f"  → Final fragments: {model.get_fragment_count()}, "
              f"Merges: {len(merge_history)}, "
              f"Completion: {efficiency['merge_completion']:.1f}%\n")
    
    # Summary
    print("\nComparison Summary:")
    print("-" * 60)
    print(f"{'Attraction':<12} {'Final Frags':<15} {'Merges':<10} {'Completion':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['attraction']:<12.1f} {result['final_fragments']:<15} "
              f"{result['merge_events']:<10} {result['completion']:<12.1f}%")
    print("-" * 60)


def main():
    """
    Main demonstration function that showcases all features of the
    ctenophore merging simulation model.
    """
    print("\n" + "=" * 70)
    print("  CTENOPHORE FRAGMENT MERGING SIMULATION")
    print("  Computational Model Demo")
    print("=" * 70)
    print("\nThis demo demonstrates a computational model of ctenophore")
    print("(comb jelly) fragment merging after dissection. The simulation")
    print("models physical forces, fragment dynamics, and merging behavior.")
    
    try:
        # Run basic simulation
        model1, state_history1, merge_history1 = run_basic_simulation()
        
        # Analyze results
        analyze_simulation_results(model1, state_history1, merge_history1, 5)
        
        # Visualize results
        visualize_simulation_results(model1, state_history1, merge_history1)
        
        # Demonstrate data export/import
        demonstrate_data_export_import(model1)
        
        # Run advanced simulation
        model2, state_history2, merge_history2 = run_advanced_simulation()
        analyze_simulation_results(model2, state_history2, merge_history2, 8)
        
        # Parameter comparison
        compare_parameter_effects()
        
        # Final summary
        print_section_header("Demo Complete!")
        print("\n✓ All simulations completed successfully")
        print("✓ Visualizations generated")
        print("✓ Data exported and verified")
        print("\nGenerated files:")
        print("  - ctenophore_merge_simulation.png (visualization)")
        print("  - ctenophore_simulation_data.json (simulation data)")
        print("\nThe simulation model successfully demonstrates:")
        print("  • Fragment generation and initialization")
        print("  • Physical force calculations (attraction and drag)")
        print("  • Fragment merging dynamics")
        print("  • Comprehensive analysis and visualization")
        print("  • Data persistence and retrieval")
        print("  • Parameter sensitivity analysis")
        
        print("\n" + "=" * 70)
        print("  Thank you for exploring the ctenophore merging model!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
