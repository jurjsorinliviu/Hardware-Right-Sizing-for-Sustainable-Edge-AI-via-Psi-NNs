"""
Tier 1 Enhancements Master Runner
==================================

Runs all Tier 1 manuscript improvements:
1. Statistical validation (10 seeds × 5 problems)
2. Heat equation experiment
3. Wave equation experiment
4. Architecture sensitivity ([20,20,20,20,20], [100,100,100])
5. Long-term convergence (10,000 epochs)

Estimated total runtime: ~14 hours

Author: Sorin Liviu Jurj
Date: 2025-11-15
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from statistical_validation import run_experiment_multiple_times, compare_regimes_statistical
from three_regime_burgers_experiment import run_three_regime_burgers
from three_regime_laplace_experiment import run_three_regime_laplace
from three_regime_memristor_experiment import run_three_regime_memristor
from three_regime_heat_experiment import run_three_regime_heat
from three_regime_wave_experiment import run_three_regime_wave


def create_experiment_wrapper(experiment_fn, **kwargs):
    """Create a wrapper function for statistical validation"""
    def wrapper(seed=42):
        results = experiment_fn(seed=seed, **kwargs)
        # Handle different result structures
        if 'training' in results['continuous']:
            # Burgers format: nested under 'training'
            return {
                'final_loss': results['continuous']['training']['final_loss'],
                'passive_loss': results['passive']['training']['final_loss'],
                'active_loss': results['active']['training']['final_loss']
            }
        else:
            # Laplace/Memristor format: direct keys
            return {
                'final_loss': results['continuous']['final_loss'],
                'passive_loss': results['passive']['final_loss'],
                'active_loss': results['active']['final_loss']
            }
    return wrapper


def serialize_for_json(obj):
    """Recursively convert numpy types and nested dicts to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


def run_tier1_enhancements():
    """
    Run all Tier 1 enhancements sequentially
    """
    print("=" * 80)
    print("TIER 1 ENHANCEMENTS - COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print()
    print("This will run:")
    print("  1. Statistical validation (10 seeds × 5 problems)")
    print("  2. Heat equation experiment")
    print("  3. Wave equation experiment")
    print("  4. Architecture sensitivity tests")
    print("  5. Long-term convergence tests (10,000 epochs)")
    print()
    print("Estimated total runtime: ~14 hours")
    print("=" * 80)
    print()
    
    input("Press Enter to start (or Ctrl+C to cancel)...")
    
    start_time_total = time.time()
    results_summary = {}
    
    # =========================================================================
    # PHASE 1: STATISTICAL VALIDATION (10 seeds × 5 problems)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: STATISTICAL VALIDATION")
    print("=" * 80)
    print("Running each experiment 10 times with different seeds")
    print("Estimated time: ~6 hours")
    print()
    
    num_seeds = 10
    base_results_dir = "chapter4/results/statistical_validation"
    
    # 1.1 Burgers PDE
    print("\n" + "-" * 80)
    print("1.1 BURGERS PDE - Statistical Validation")
    print("-" * 80)
    
    burgers_wrapper = create_experiment_wrapper(
        run_three_regime_burgers,
        epochs=3000,
        save_dir=f"{base_results_dir}/burgers"
    )
    
    burgers_stats = run_experiment_multiple_times(
        burgers_wrapper,
        num_runs=num_seeds,
        experiment_name="burgers_statistical",
        results_dir=f"{base_results_dir}/burgers"
    )
    
    results_summary['burgers_statistical'] = burgers_stats
    
    # 1.2 Laplace Equation
    print("\n" + "-" * 80)
    print("1.2 LAPLACE EQUATION - Statistical Validation")
    print("-" * 80)
    
    laplace_wrapper = create_experiment_wrapper(
        run_three_regime_laplace,
        epochs=3000,
        save_dir=f"{base_results_dir}/laplace"
    )
    
    laplace_stats = run_experiment_multiple_times(
        laplace_wrapper,
        num_runs=num_seeds,
        experiment_name="laplace_statistical",
        results_dir=f"{base_results_dir}/laplace"
    )
    
    results_summary['laplace_statistical'] = laplace_stats
    
    # 1.3 Memristor Device
    print("\n" + "-" * 80)
    print("1.3 MEMRISTOR DEVICE - Statistical Validation")
    print("-" * 80)
    
    memristor_wrapper = create_experiment_wrapper(
        run_three_regime_memristor,
        epochs=3000,
        save_dir=f"{base_results_dir}/memristor"
    )
    
    memristor_stats = run_experiment_multiple_times(
        memristor_wrapper,
        num_runs=num_seeds,
        experiment_name="memristor_statistical",
        results_dir=f"{base_results_dir}/memristor"
    )
    
    results_summary['memristor_statistical'] = memristor_stats
    
    # 1.4 Heat Equation
    print("\n" + "-" * 80)
    print("1.4 HEAT EQUATION - Statistical Validation")
    print("-" * 80)
    
    heat_wrapper = create_experiment_wrapper(
        run_three_regime_heat,
        epochs=3000,
        save_dir=f"{base_results_dir}/heat"
    )
    
    heat_stats = run_experiment_multiple_times(
        heat_wrapper,
        num_runs=num_seeds,
        experiment_name="heat_statistical",
        results_dir=f"{base_results_dir}/heat"
    )
    
    results_summary['heat_statistical'] = heat_stats
    
    # 1.5 Wave Equation
    print("\n" + "-" * 80)
    print("1.5 WAVE EQUATION - Statistical Validation")
    print("-" * 80)
    
    wave_wrapper = create_experiment_wrapper(
        run_three_regime_wave,
        epochs=3000,
        save_dir=f"{base_results_dir}/wave"
    )
    
    wave_stats = run_experiment_multiple_times(
        wave_wrapper,
        num_runs=num_seeds,
        experiment_name="wave_statistical",
        results_dir=f"{base_results_dir}/wave"
    )
    
    results_summary['wave_statistical'] = wave_stats
    
    # =========================================================================
    # PHASE 2: ARCHITECTURE SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: ARCHITECTURE SENSITIVITY")
    print("=" * 80)
    print("Testing [20,20,20,20,20] (deeper) and [100,100,100] (wider)")
    print("Estimated time: ~3 hours")
    print()
    
    # 2.1 Deeper network [20,20,20,20,20]
    print("\n" + "-" * 80)
    print("2.1 BURGERS - Deeper Network [20,20,20,20,20]")
    print("-" * 80)
    
    burgers_deep = run_three_regime_burgers(
        epochs=3000,
        hidden_sizes=[20, 20, 20, 20, 20],
        save_dir="chapter4/results/architecture_sensitivity/burgers_deep"
    )
    
    results_summary['burgers_deep'] = {
        'continuous': burgers_deep['continuous']['training']['final_loss'],
        'passive': burgers_deep['passive']['training']['final_loss'],
        'active': burgers_deep['active']['training']['final_loss']
    }
    
    # 2.2 Wider network [100,100,100]
    print("\n" + "-" * 80)
    print("2.2 BURGERS - Wider Network [100,100,100]")
    print("-" * 80)
    
    burgers_wide = run_three_regime_burgers(
        epochs=3000,
        hidden_sizes=[100, 100, 100],
        save_dir="chapter4/results/architecture_sensitivity/burgers_wide"
    )
    
    results_summary['burgers_wide'] = {
        'continuous': burgers_wide['continuous']['training']['final_loss'],
        'passive': burgers_wide['passive']['training']['final_loss'],
        'active': burgers_wide['active']['training']['final_loss']
    }
    
    # =========================================================================
    # PHASE 3: LONG-TERM CONVERGENCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: LONG-TERM CONVERGENCE")
    print("=" * 80)
    print("Testing 10,000 epochs to see long-term behavior")
    print("Estimated time: ~3 hours")
    print()
    
    # 3.1 Burgers 10k epochs
    print("\n" + "-" * 80)
    print("3.1 BURGERS - 10,000 Epochs")
    print("-" * 80)
    
    burgers_10k = run_three_regime_burgers(
        epochs=10000,
        save_dir="chapter4/results/long_term_convergence/burgers_10k"
    )
    
    results_summary['burgers_10k'] = {
        'continuous': burgers_10k['continuous']['training']['final_loss'],
        'passive': burgers_10k['passive']['training']['final_loss'],
        'active': burgers_10k['active']['training']['final_loss']
    }
    
    # 3.2 Laplace 10k epochs
    print("\n" + "-" * 80)
    print("3.2 LAPLACE - 10,000 Epochs")
    print("-" * 80)
    
    laplace_10k = run_three_regime_laplace(
        epochs=10000,
        save_dir="chapter4/results/long_term_convergence/laplace_10k"
    )
    
    results_summary['laplace_10k'] = {
        'continuous': laplace_10k['continuous']['final_loss'],
        'passive': laplace_10k['passive']['final_loss'],
        'active': laplace_10k['active']['final_loss']
    }
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time_total
    
    print("\n" + "=" * 80)
    print("TIER 1 ENHANCEMENTS - COMPLETE!")
    print("=" * 80)
    print(f"\nTotal execution time: {total_time/3600:.2f} hours")
    print()
    
    # Save comprehensive summary
    summary_path = "chapter4/results/tier1_enhancements_summary.json"
    with open(summary_path, 'w') as f:
        # Convert to JSON-serializable format using recursive serializer
        json_summary = serialize_for_json(results_summary)
        json.dump(json_summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\n1. STATISTICAL VALIDATION (with 95% CI)")
    for problem in ['burgers', 'laplace', 'memristor', 'heat', 'wave']:
        key = f'{problem}_statistical'
        if key in results_summary:
            stats = results_summary[key]
            if 'final_loss' in stats:
                mean = stats['final_loss']['mean']
                ci_lower = stats['final_loss']['ci_95']['lower']
                ci_upper = stats['final_loss']['ci_95']['upper']
                print(f"  {problem.capitalize():12s}: {mean:.6f} ± [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    print("\n2. ARCHITECTURE SENSITIVITY")
    if 'burgers_deep' in results_summary:
        deep = results_summary['burgers_deep']
        print(f"  Deeper [20,20,20,20,20]: Passive {deep['passive']:.6f} vs Continuous {deep['continuous']:.6f}")
    if 'burgers_wide' in results_summary:
        wide = results_summary['burgers_wide']
        print(f"  Wider [100,100,100]:     Passive {wide['passive']:.6f} vs Continuous {wide['continuous']:.6f}")
    
    print("\n3. LONG-TERM CONVERGENCE (10,000 epochs)")
    for problem in ['burgers', 'laplace']:
        key = f'{problem}_10k'
        if key in results_summary:
            result = results_summary[key]
            passive_deg = ((result['passive'] / result['continuous']) - 1) * 100
            print(f"  {problem.capitalize():12s}: Passive {passive_deg:+.2f}% vs Continuous")
    
    print("\n" + "=" * 80)
    print("All Tier 1 enhancements complete!")
    print("Results ready for manuscript inclusion with:")
    print("  ✓ Error bars and confidence intervals")
    print("  ✓ Statistical significance tests")
    print("  ✓ 5 diverse test problems (Burgers, Laplace, Memristor, Heat, Wave)")
    print("  ✓ Architecture robustness validation")
    print("  ✓ Long-term convergence analysis")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    try:
        results = run_tier1_enhancements()
        print("\n✓ SUCCESS: All Tier 1 enhancements completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ INTERRUPTED: Tier 1 enhancements cancelled by user")
    except Exception as e:
        print(f"\n\n✗ ERROR: {str(e)}")
        raise