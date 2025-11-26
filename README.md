# Sustainable Edge AI: Discovering Minimal Hardware Requirements Through Physics Structure-Informed Learning Under Renewable Energy Constraints

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Author**: Sorin Liviu Jurj   
> **Status**: Under Review

## 📋 Overview

This repository contains the complete implementation and experimental validation for our framework that couples **physics structure-informed learning** with **renewable energy constraints** to address three critical challenges in Edge AI deployment:

1. **Hardware Over-specification**: Eliminates the "*Bill-of-Materials Bomb*" causing up to 31× cost penalties
2. **Carbon Footprint**: Achieves 14× reduction in lifecycle emissions through solar-constrained training
3. **Deployment Optimization**: Automatically extracts hardware requirements before prototyping

### Key Results

| Metric                 | Improvement                                     |
| ---------------------- | ----------------------------------------------- |
| **Cost Savings**       | 96.8% (from $249 Jetson Orin Nano → $8 STM32H7) |
| **Carbon Reduction**   | 14× lifecycle emissions (238 kg → 17.6 kg CO₂)  |
| **Training Energy**    | 50% reduction via solar constraints             |
| **Performance Impact** | <6% degradation for suitable problem classes    |

## 🎯 Key Contributions

### 1. Hardware Specification Extraction

Automatically computes from discovered neural network structures:

- **TOPS Requirements**: Minimum computational throughput
- **Memory Footprint**: RAM/ROM requirements
- **Power Budget**: Average and peak consumption
- **Platform Recommendation**: TinyML, Mid-Range, or High-Performance tiers

### 2. Solar-Constrained Training Protocol

Novel training approach using 50% duty cycle renewable energy:

- **Checkpoint Mechanism**: Preserves Adam optimizer momentum across power interruptions
- **Passive Regime (κ=0)**: Universally outperforms adaptive regularization
- **Problem Classification**: Predicts method applicability before deployment

### 3. Three-Class Problem Taxonomy

| Class                    | Mathematical Characteristics                                 | Degradation | Examples                                |
| ------------------------ | ------------------------------------------------------------ | ----------- | --------------------------------------- |
| **Class A (Optimal)**    | Elliptic/parabolic PDEs with diffusive damping               | <20%        | Burgers (+1.9%), Laplace (+5.8%)        |
| **Class B (Unsuitable)** | First-order hyperbolic transport; nonlinear reaction-diffusion | >100%       | Advection (+7200%), Allen-Cahn (+1660%) |
| **Class C (Marginal)**   | Second-order hyperbolic with oscillatory behavior            | 20-200%     | Wave (+163%), Heat (+172%)              |

### 4. Statistical Validation

- **60+ experiments** across 6 problem classes
- **10 independent random seeds** per configuration
- **95% confidence intervals** via bootstrap resampling
- **Paired t-tests** with p < 0.001 significance threshold

## 🏗️ Repository Structure

```
├── requirements.txt                # Python dependencies
├── run_all_experiments.py          # Master experiment runner
├── sustainable_edge_ai.py      # Main implementation
├── generate_figure1_timeline.py
├── generate_figure4_realistic_solar.py
├── run_all_chapter4_experiments.py
│
├── experiments/                # Individual problem experiments
│   ├── burgers_solar_experiment.py
│   ├── duty_cycle_sweep.py
│   ├── kappa_sweep_experiment.py
│   ├── realistic_solar_validation.py
│   ├── statistical_validation.py
│   ├── three_regime_advection_experiment.py
│   └── results/                    # Experimental outputs
│       ├── figure1_three_regime_timeline.png
│       ├── figure4_realistic_solar_comparison.png
│       ├── unified_results.csv
│       ├── unified_comparison_table.tex
│       ├── three_regime_burgers/
│       ├── three_regime_laplace/
│       ├── three_regime_memristor/
│       ├── statistical_validation/
│       ├── architecture_sensitivity/
│       ├── long_term_convergence/
│       └── realistic_solar_burgers/
│
├── PSI-HDL-implementation/          # Base Ψ-HDL framework
│   ├── Code/
│   │   ├── structure_extractor.py  # Hierarchical clustering
│   │   ├── verilog_generator.py    # HDL code generation
│   │   └── vteam_baseline.py       # Memristor baseline
│   └── Psi-NN-main/                # Original Ψ-NN framework
│       ├── Module/
│       │   ├── PsiNN_burgers.py
│       │   ├── PsiNN_laplace.py
│       │   ├── PsiNN_poisson.py
│       │   └── Training.py
│       └── Config/                 # Experiment configurations
│
├── results/                         # Legacy demonstration results
│   ├── burgers/
│   ├── memristor/
│   └── multi_model/
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU (for full experiments)
nvidia-smi
```

### Installation

```bash
# Clone repository
git clone https://github.com/jurjsorinliviu/Sustainable-Edge-AI.git
cd Sustainable-Edge-AI

# Install dependencies
pip install -r requirements.txt
pip install -r PSI-HDL-implementation/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Running Experiments

#### 1. Quick Demonstration (Chapter 4 Extensions)

```bash
# Run all Chapter 4 experiments
cd chapter4
python run_all_chapter4_experiments.py

# Results saved to chapter4/results/
```

#### 2. Individual Problem Classes

```bash
# Three-regime comparison (Burgers PDE)
python experiments/burgers_solar_experiment.py

# Statistical validation with 10 seeds
python experiments/statistical_validation.py

# Kappa sweep analysis (κ = 0.0 to 2.0)
python experiments/kappa_sweep_experiment.py

# Realistic weather-dependent solar patterns
python experiments/realistic_solar_validation.py
```

#### 3. Generate Paper Figures

```bash
# Figure 1: Three-regime training timeline
python generate_figure1_timeline.py

# Figure 4: Realistic solar comparison
python generate_figure4_realistic_solar.py
```

## 📊 Core Modules

### 1. Solar-Constrained Training

```python
from sustainable_edge_ai import SolarConstrainedTrainer

# Initialize trainer
trainer = SolarConstrainedTrainer(model, config={
    'duty_cycle': 0.5,           # 50% solar availability
    'active_period': 10,         # Train for 10 steps
    'idle_period': 10,           # Idle for 10 steps
    'checkpoint_frequency': 100, # Save every 100 steps
    'adaptive_regularization': False  # Use passive regime (κ=0)
})

# Training loop
for epoch in range(num_epochs):
    loss = trainer.train_step(loss_fn=compute_loss, optimizer=optimizer)
    if trainer.should_checkpoint():
        trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
```

### 2. Hardware Specification Extraction

```python
from sustainable_edge_ai import HardwareSpecificationExtractor
from structure_extractor import StructureExtractor

# Extract network structure
struct_extractor = StructureExtractor(model, model_type="PsiNN_burgers")
hw_extractor = HardwareSpecificationExtractor(model, struct_extractor)

# Compute specifications
specs = {
    'operations': hw_extractor.compute_operations(),
    'tops': hw_extractor.compute_tops_requirement(target_fps=30.0),
    'memory_kb': hw_extractor.compute_memory_requirements() / 1024,
    'power_mw': hw_extractor.estimate_power_consumption() * 1000
}

print(f"TOPS Required: {specs['tops']:.6f}")
print(f"Memory: {specs['memory_kb']:.2f} KB")
print(f"Power: {specs['power_mw']:.2f} mW")
```

### 3. Platform Recommendation

```python
from sustainable_edge_ai import EdgeAIPlatformRecommender

# Initialize recommender
recommender = EdgeAIPlatformRecommender()

# Get platform recommendations
platforms = recommender.recommend_platform(
    requirements=specs,
    constraints={'max_cost_usd': 100, 'max_power_mw': 10000}
)

# Display results
for i, platform in enumerate(platforms[:3], 1):
    print(f"{i}. {platform['name']}: "
          f"${platform['cost']:.2f}, "
          f"{platform['utilization']*100:.1f}% utilization, "
          f"Fit: {platform['fit_category']}")
```

### 4. Carbon Footprint Analysis

```python
from sustainable_edge_ai import CarbonFootprintAnalyzer

# Initialize analyzer
analyzer = CarbonFootprintAnalyzer()

# Compute lifecycle emissions
carbon = analyzer.compute_lifecycle_carbon(
    platform=platforms[0],
    deployment_years=5.0,
    training_regime='solar',  # vs 'grid'
    duty_cycle=0.5
)

print(f"Training Carbon: {carbon['training_kg_co2']:.3f} kg CO₂")
print(f"Deployment Carbon: {carbon['deployment_kg_co2']:.1f} kg CO₂")
print(f"Total Lifecycle: {carbon['total_kg_co2']:.1f} kg CO₂")
```

## 🔬 Experimental Results

### Problem Classification Summary

| Problem       | PDE Type                | Solar Compatible | Degradation | Cohen's d |
| ------------- | ----------------------- | ---------------- | ----------- | --------- |
| **Burgers**   | Parabolic (nonlinear)   | ✅ Yes            | +1.9%       | 0.67      |
| **Laplace**   | Elliptic (steady-state) | ✅ Yes            | +5.8%       | 3.5       |
| **Memristor** | ODE (device physics)    | ❌ No             | +236%       | -         |
| **Heat**      | Parabolic               | ❌ No             | +172.5%     | -         |
| **Wave**      | Hyperbolic (2nd-order)  | ⚠️ Marginal       | +163.2%     | 8.2       |
| **Advection** | Hyperbolic (1st-order)  | ❌ No             | +7200%      | 8.5       |

### Passive vs Active Regime (κ-Sweep Analysis)

Wave equation results at learning rate 1×10⁻²:

| κ Value           | Degradation | Status             |
| ----------------- | ----------- | ------------------ |
| **0.0** (Passive) | **-15.25%** | ✅ Best performance |
| 0.25              | -14.54%     | ✅ Good             |
| 0.5               | -9.21%      | ✅ Acceptable       |
| 0.75              | +86.47%     | ❌ Failed           |
| 1.0               | +106.97%    | ❌ Failed           |
| 2.0               | +140.22%    | ❌ Failed           |

**Key Finding**: Passive regime (κ=0) universally outperforms all active variants.

### Platform Recommendation Example (Burgers PDE)

| Platform        | TOPS  | Cost  | Power  | Utilization | Fit            | Score  |
| --------------- | ----- | ----- | ------ | ----------- | -------------- | ------ |
| **STM32H7**     | 0.082 | $8    | 400 mW | 0.027%      | Optimal        | 214    |
| Nordic nRF52840 | 0.026 | $3.50 | 15 mW  | 0.085%      | Over-specified | 252    |
| TI AM62A        | 2.0   | $35   | 2 W    | <0.001%     | Over-specified | 17,500 |

### Carbon Footprint Comparison (5-year lifecycle)

| Scenario            | Training | Deployment | Total       | Reduction    |
| ------------------- | -------- | ---------- | ----------- | ------------ |
| **Solar + STM32H7** | 0.036 kg | 17.6 kg    | **17.6 kg** | **14× less** |
| Grid + Jetson Orin  | 0.356 kg | 208 kg     | **238 kg**  | Baseline     |

At scale (10,000 devices): **2,204 tons CO₂ reduction** = 479 cars removed for 1 year

## 📈 Reproducing Results

### Statistical Validation Protocol

All experiments use rigorous statistical validation:

```python
# Example: Burgers PDE with 10 seeds
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
results = {'continuous': [], 'passive': [], 'active': []}

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train all three regimes
    results['continuous'].append(train_continuous(seed))
    results['passive'].append(train_solar_passive(seed))
    results['active'].append(train_solar_active(seed))

# Compute statistics
from scipy import stats
mean_continuous = np.mean(results['continuous'])
mean_passive = np.mean(results['passive'])
degradation = 100 * (mean_passive - mean_continuous) / mean_continuous

# Paired t-test
t_stat, p_value = stats.ttest_rel(results['continuous'], results['passive'])
cohens_d = (mean_passive - mean_continuous) / np.std(results['continuous'])

print(f"Degradation: {degradation:.1f}%")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.2f}")
```

### Expected Runtime

| Experiment             | Seeds       | Epochs  | GPU Time   | Wall Clock |
| ---------------------- | ----------- | ------- | ---------- | ---------- |
| Single problem         | 1           | 6000    | ~3 hours   | ~3 hours   |
| Statistical validation | 10          | 6000    | ~30 hours  | ~30 hours  |
| Full Chapter 4         | 60+ configs | Various | ~180 hours | ~7 days    |

**Note**: Solar-constrained training extends wall-clock time by 2× due to 50% duty cycle.

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@article{jurj2025sustainable,
  author = {Sorin Liviu Jurj},
  title = {Sustainable Edge AI: Discovering Minimal Hardware Requirements 
           Through Physics Structure-Informed Learning Under Renewable 
           Energy Constraints},
  journal = {Under Review},
  year = {2025},
  url = {https://github.com/jurjsorinliviu/Sustainable-Edge-AI}
}
```

## 📚 Related Publications

1. **Ψ-HDL Framework**: [PSI-HDL GitHub](https://github.com/jurjsorinliviu/PSI-HDL)
2. **Original Ψ-NN**: [Psi-NN GitHub](https://github.com/ZitiLiu/Psi-NN)

## 🔍 Key Findings Summary

### ✅ What Works

1. **Elliptic and Parabolic PDEs** (Class A):
   - Burgers equation: +1.9% degradation
   - Laplace equation: +5.8% degradation
   - Smooth optimization landscapes enable successful intermittent training

2. **Passive Regime (κ=0)**:
   - Universally outperforms adaptive regularization
   - Simpler deployment (no hyperparameters)
   - Checkpoint momentum preservation provides sufficient regularization

3. **Hardware Specification Extraction**:
   - Accurate TOPS, memory, and power predictions
   - Enables platform selection before prototyping
   - 96.8% cost reduction demonstrated

### ❌ What Doesn't Work

1. **Hyperbolic Transport Problems** (Class B):
   - Advection equation: +7200% degradation
   - Wavefront desynchronization fundamental issue
   - Requires continuous training or >80% duty cycle

2. **Nonlinear Reaction-Diffusion** (Class B):
   - Allen-Cahn equation: +1660% degradation
   - Phase separation dynamics disrupted by interruptions
   - Metastable state trapping during checkpoints

3. **Hysteretic Device Physics** (Class B):
   - Memristor: +236% degradation
   - Memory effects incompatible with power interruptions
   - Requires continuous state tracking

### ⚠️ Requires Careful Tuning (Class C)

- Wave equation: +163% degradation at standard learning rate
- Improves to -15% with 10× higher learning rate but high variance
- Second-order oscillatory dynamics sensitive to hyperparameters

## 🛠️ Hardware Platforms Database

The framework supports automatic recommendation from 6 validated platforms:

| Tier          | Platform         | TOPS  | Memory  | Power  | Cost  | Technology       |
| ------------- | ---------------- | ----- | ------- | ------ | ----- | ---------------- |
| **TinyML**    | Nordic nRF52840  | 0.026 | 256 KB  | 15 mW  | $3.50 | Cortex-M4        |
| **TinyML**    | STM32H7          | 0.082 | 1024 KB | 400 mW | $8    | Cortex-M7        |
| **Mid-Range** | TI AM62A         | 2.0   | 2 MB    | 2 W    | $35   | Cortex-A53+CNN   |
| **Mid-Range** | TI TDA4VM        | 8.0   | 8 MB    | 4.5 W  | $80   | Cortex-A72+DSP   |
| **High-Perf** | Hailo-8          | 26.0  | 4 MB    | 5 W    | $150  | Neural Processor |
| **High-Perf** | Jetson Orin Nano | 40.0  | 8 MB    | 10 W   | $249  | Ampere GPU       |

## 🌍 Environmental Impact

### Single Device (5-year lifecycle)

- **Traditional Approach**: Grid training + Jetson Orin Nano = **238 kg CO₂**
- **Our Framework**: Solar training + STM32H7 = **17.6 kg CO₂**
- **Reduction**: 220.4 kg CO₂ (92.6% less)

### At Scale

| Deployment       | Traditional  | Our Framework | Reduction    | Equivalent                  |
| ---------------- | ------------ | ------------- | ------------ | --------------------------- |
| 1,000 devices    | 238 tons     | 17.6 tons     | 220 tons     | 48 cars removed             |
| 10,000 devices   | 2,380 tons   | 176 tons      | 2,204 tons   | 479 cars / 614 acres forest |
| 1M devices (IoT) | 238,000 tons | 17,600 tons   | 220,400 tons | 47,900 cars / 61,400 acres  |

**Note**: Equivalencies based on EPA averages (4.6 tons CO₂/car/year, 0.36 tons CO₂/acre/year forest sequestration)

## 🤝 Contributing

We welcome contributions! Areas of interest:

- [ ] Additional problem class validation (Poisson, biharmonic, Helmholtz)
- [ ] Extended platform database (Qualcomm, Google Coral, Intel Movidius)
- [ ] Real hardware deployment validation
- [ ] Multi-physics coupled problems (thermoelasticity, MHD)
- [ ] Theoretical convergence guarantees
- [ ] Wind + solar hybrid renewable strategies

## 📞 Contact

**Sorin Liviu Jurj**  
Politehnica University of Timisoara, Romania  
Email: jurjsorinliviu@yahoo.de  
GitHub: [@jurjsorinliviu](https://github.com/jurjsorinliviu)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Ψ-NN framework by Liu et al. (Nature Communications, 2025)
- PSI-HDL framework development

---

**Last Updated**: November 2025  
**Paper Status**: Under Review  
**Code Version**: v1.0.0
