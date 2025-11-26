"""
Comprehensive Solar Constraint Analysis Runner
==============================================

Runs both:
1. Realistic Solar Model Validation (Equations 50-51)
2. Duty Cycle Sweep (30%, 50%, 70%)

This provides complete validation of the methodology under
varying energy constraints.

Author: Sorin Liviu Jurj
Date: 2025-11-15
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import experiment modules
from experiments.realistic_solar_validation import run_realistic_solar_experiment
from experiments.duty_cycle_sweep import run_duty_cycle_sweep


def run_comprehensive_analysis():
    """
    Run all comprehensive solar constraint experiments
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "COMPREHENSIVE SOLAR CONSTRAINT ANALYSIS")
    print("=" * 80)
    print()
    print("This will run two comprehensive experiments:")
    print("  1. Realistic Solar Model Validation")
    print("     - Full Equations 50-51 implementation")
    print("     - Sinusoidal diurnal cycle + Weather Markov chain")
    print("     - Variable duty cycle (naturally varies with weather)")
    print()
    print("  2. Duty Cycle Sweep")
    print("     - Test duty cycles: 30%, 50%, 70%")
    print("     - Find optimal energy-accuracy trade-off")
    print("     - Demonstrate robustness to varying energy availability")
    print()
    print("Estimated total runtime: ~2-3 hours")
    print("=" * 80)
    print()
    
    overall_start = time.time()
    
    # ==================================================================
    # EXPERIMENT 1: Realistic Solar Model Validation
    # ==================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1/2: REALISTIC SOLAR MODEL VALIDATION")
    print("=" * 80)
    
    exp1_start = time.time()
    
    try:
        results_realistic = run_realistic_solar_experiment(
            results_dir='chapter4/results/realistic_solar_burgers'
        )
        exp1_time = time.time() - exp1_start
        print(f"\n✓ Experiment 1 completed successfully in {exp1_time:.1f}s ({exp1_time/60:.1f} min)")
    except Exception as e:
        print(f"\n✗ Experiment 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ==================================================================
    # EXPERIMENT 2: Duty Cycle Sweep
    # ==================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2/2: DUTY CYCLE SWEEP")
    print("=" * 80)
    
    exp2_start = time.time()
    
    try:
        results_sweep = run_duty_cycle_sweep(
            results_dir='chapter4/results/duty_cycle_sweep'
        )
        exp2_time = time.time() - exp2_start
        print(f"\n✓ Experiment 2 completed successfully in {exp2_time:.1f}s ({exp2_time/60:.1f} min)")
    except Exception as e:
        print(f"\n✗ Experiment 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ==================================================================
    # OVERALL SUMMARY
    # ==================================================================
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print(" " * 20 + "COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Total execution time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print()
    print("Results saved to:")
    print("  1. chapter4/results/realistic_solar_burgers/")
    print("  2. chapter4/results/duty_cycle_sweep/")
    print()
    print("Key Findings:")
    print("-" * 80)
    
    # Realistic Solar Results
    print("\n1. REALISTIC SOLAR MODEL:")
    if results_realistic:
        baseline_loss = results_realistic['continuous']['final_loss']
        passive_loss = results_realistic['passive_realistic']['final_loss']
        active_loss = results_realistic['active_realistic']['final_loss']
        passive_dc = results_realistic['passive_realistic']['duty_cycle']
        
        print(f"   Baseline (Continuous):  Loss = {baseline_loss:.6f}")
        print(f"   Passive (Realistic):    Loss = {passive_loss:.6f} ({(passive_loss/baseline_loss-1)*100:+.2f}%)")
        print(f"   Active (Realistic):     Loss = {active_loss:.6f} ({(active_loss/baseline_loss-1)*100:+.2f}%)")
        print(f"   Actual Duty Cycle:      {passive_dc*100:.2f}%")
        print(f"   Conclusion: {'Passive IMPROVES' if passive_loss < baseline_loss else 'Passive degrades slightly'}")
    
    # Duty Cycle Sweep Results
    print("\n2. DUTY CYCLE SWEEP:")
    if results_sweep:
        baseline_loss = results_sweep['continuous']['final_loss']
        
        for dc_pct in [30, 50, 70]:
            dc_key = f'duty_cycle_{dc_pct}'
            if dc_key in results_sweep:
                passive_loss = results_sweep[dc_key]['passive']['final_loss']
                active_loss = results_sweep[dc_key]['active']['final_loss']
                
                print(f"   {dc_pct}% Duty Cycle:")
                print(f"     Passive: {passive_loss:.6f} ({(passive_loss/baseline_loss-1)*100:+.2f}%)")
                print(f"     Active:  {active_loss:.6f} ({(active_loss/baseline_loss-1)*100:+.2f}%)")
        
        # Find optimal duty cycle
        best_dc = None
        best_loss = float('inf')
        for dc_pct in [30, 50, 70]:
            dc_key = f'duty_cycle_{dc_pct}'
            if dc_key in results_sweep:
                passive_loss = results_sweep[dc_key]['passive']['final_loss']
                if passive_loss < best_loss:
                    best_loss = passive_loss
                    best_dc = dc_pct
        
        if best_dc:
            energy_savings = 100 - best_dc
            print(f"\n   Optimal Configuration:")
            print(f"     Duty Cycle: {best_dc}%")
            print(f"     Energy Savings: {energy_savings}%")
            print(f"     Performance: {(best_loss/baseline_loss-1)*100:+.2f}% vs baseline")
    
    print("\n" + "=" * 80)
    print("All results, plots, and analysis saved successfully!")
    print("=" * 80)
    print()
    
    return {
        'realistic_solar': results_realistic,
        'duty_cycle_sweep': results_sweep,
        'total_time': overall_time
    }


if __name__ == '__main__':
    results = run_comprehensive_analysis()
    
    if results:
        print("\n✓ SUCCESS: All comprehensive solar constraint experiments completed!")
    else:
        print("\n✗ FAILURE: Some experiments failed. Check error messages above.")
        sys.exit(1)