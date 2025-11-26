"""
Chapter 4 Experiment: Burgers Equation with Solar-Constrained Training

This experiment extends the existing PsiNN_burgers.py to demonstrate:
1. Solar-constrained training (50% duty cycle)
2. Hardware specification extraction
3. Platform recommendation for Edge AI deployment
4. Carbon footprint analysis

Uses actual Ψ-NN architecture from PSI-HDL-implementation/Code/PsiNN_burgers.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import json

# Add paths
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "PSI-HDL-implementation" / "Code"))
sys.path.insert(0, str(BASE_DIR / "PSI-HDL-implementation" / "Psi-NN-main" / "Module"))
sys.path.insert(0, str(BASE_DIR / "chapter4"))

# Import actual Ψ-NN architecture
import PsiNN_burgers
from structure_extractor import StructureExtractor

# Import Chapter 4 extensions
from sustainable_edge_ai import (
    SolarConstrainedTrainer,
    HardwareSpecificationExtractor,
    EdgeAIPlatformRecommender,
    CarbonFootprintAnalyzer,
    create_experiment_config
)


class BurgersSolarExperiment:
    """Burgers equation with solar-constrained training"""
    
    def __init__(self, node_num: int = 16, use_solar: bool = True):
        """
        Initialize experiment
        
        Args:
            node_num: Number of neurons per layer
            use_solar: Enable solar-constrained training
        """
        self.node_num = node_num
        self.use_solar = use_solar
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create actual Ψ-NN Burgers architecture
        self.model = PsiNN_burgers.Net(node_num=node_num, output_num=1)
        self.model.to(self.device)
        
        print(f"✓ Using actual Ψ-NN Burgers architecture")
        print(f"  Node num: {node_num}")
        print(f"  Solar training: {use_solar}")
        print(f"  Device: {self.device}")
        
        # Configuration
        self.config = create_experiment_config('burgers', solar_training=use_solar)
        
        # Results storage
        self.results = {
            'training_mode': 'solar' if use_solar else 'grid',
            'losses': [],
            'times': [],
            'solar_stats': None,
            'hardware_specs': None,
            'platform_recommendation': None,
            'carbon_footprint': None
        }
    
    def generate_data(self, n_points: int = 2000):
        """Generate Burgers equation training data"""
        print(f"\n[DATA] Generating Burgers equation data ({n_points} points)...")
        
        # Domain: x ∈ [-1, 1], t ∈ [0, 1]
        np.random.seed(42)
        x = np.random.uniform(-1, 1, (n_points, 1))
        t = np.random.uniform(0, 1, (n_points, 1))
        
        # Analytical solution: u(x,t) = -sin(πx) * exp(-ν*π²*t)
        nu = 0.01 / np.pi  # Viscosity
        u = -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(np.hstack([t, x])).to(self.device)
        self.X_train.requires_grad = True
        self.u_train = torch.FloatTensor(u).to(self.device)
        
        print(f"  ✓ Generated {n_points} training points")
        print(f"  Domain: t ∈ [0, 1], x ∈ [-1, 1]")
        print(f"  Viscosity ν = {nu:.6f}")
        
        return self.X_train, self.u_train
    
    def physics_loss(self, X, u_pred):
        """
        Compute Burgers equation physics loss
        PDE: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
        """
        nu = 0.01 / np.pi
        
        # Compute derivatives with respect to X (which has gradients enabled)
        # u_pred is already computed from X, so we can compute gradients
        grad_u = torch.autograd.grad(
            outputs=u_pred, inputs=X,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = grad_u[:, 0:1]  # ∂u/∂t
        u_x = grad_u[:, 1:2]  # ∂u/∂x
        
        # Second derivative ∂²u/∂x²
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        
        # PDE residual
        pde_residual = u_t + u_pred * u_x - nu * u_xx
        
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self):
        """Compute boundary condition loss"""
        # Initial condition: u(x, 0) = -sin(πx)
        n_bc = 100
        x_ic = torch.linspace(-1, 1, n_bc).reshape(-1, 1).to(self.device)
        t_ic = torch.zeros_like(x_ic).to(self.device)
        X_ic = torch.cat([t_ic, x_ic], dim=1)
        X_ic.requires_grad = True
        
        u_ic_pred = self.model(X_ic)
        u_ic_true = -torch.sin(np.pi * x_ic)
        
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
        
        # Boundary conditions: u(-1, t) = u(1, t) = 0
        t_bc = torch.linspace(0, 1, n_bc).reshape(-1, 1).to(self.device)
        x_left = -torch.ones_like(t_bc).to(self.device)
        x_right = torch.ones_like(t_bc).to(self.device)
        
        X_left = torch.cat([t_bc, x_left], dim=1)
        X_right = torch.cat([t_bc, x_right], dim=1)
        X_left.requires_grad = True
        X_right.requires_grad = True
        
        u_left = self.model(X_left)
        u_right = self.model(X_right)
        
        loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)
        
        return loss_ic + loss_bc
    
    def compute_loss(self, reg_weight: float = 1e-3):
        """Compute total loss with regularization"""
        # Forward pass
        u_pred = self.model(self.X_train)
        
        # Data loss
        loss_data = torch.mean((u_pred - self.u_train) ** 2)
        
        # Physics loss
        loss_physics = self.physics_loss(self.X_train, u_pred)
        
        # Boundary loss
        loss_boundary = self.boundary_loss()
        
        # Regularization (for Ψ-NN compression)
        loss_reg = torch.tensor(0.0).to(self.device)
        for param in self.model.parameters():
            loss_reg += torch.norm(param, p=2)
        loss_reg *= reg_weight
        
        # Total loss
        loss_total = loss_data + loss_physics + loss_boundary + loss_reg
        
        return loss_total
    
    def train(self, epochs: int = 5000):
        """Train model with optional solar constraints"""
        print(f"\n[TRAIN] Training Ψ-NN Burgers model...")
        print(f"  Mode: {'Solar-constrained' if self.use_solar else 'Grid-powered'}")
        print(f"  Epochs: {epochs}")
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Initialize solar trainer if needed
        if self.use_solar:
            solar_trainer = SolarConstrainedTrainer(self.model, self.config)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            if self.use_solar:
                # Solar-constrained training
                loss_value = solar_trainer.train_step(
                    loss_fn=self.compute_loss,
                    optimizer=optimizer
                )
                if loss_value is not None:
                    self.results['losses'].append(loss_value)
            else:
                # Standard grid training
                optimizer.zero_grad()
                loss = self.compute_loss(reg_weight=self.config['reg_weight'])
                loss.backward()
                optimizer.step()
                self.results['losses'].append(loss.item())
            
            # Logging
            if (epoch + 1) % 500 == 0:
                current_loss = self.results['losses'][-1] if self.results['losses'] else 0
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1:5d}/{epochs}: Loss = {current_loss:.6e}, Time = {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        self.results['times'].append(total_time)
        
        # Get solar training statistics
        if self.use_solar:
            self.results['solar_stats'] = solar_trainer.get_training_stats()
            print(f"\n  Solar Training Statistics:")
            print(f"    Total steps: {self.results['solar_stats']['total_steps']}")
            print(f"    Active steps: {self.results['solar_stats']['active_steps']}")
            print(f"    Actual duty cycle: {self.results['solar_stats']['actual_duty_cycle']:.2%}")
        
        print(f"\n  ✓ Training complete: {total_time:.1f}s")
        print(f"  Final loss: {self.results['losses'][-1]:.6e}")
    
    def extract_hardware_specs(self):
        """Extract hardware specifications from trained model"""
        print(f"\n[HARDWARE] Extracting specifications...")
        
        # Use existing structure extractor
        struct_extractor = StructureExtractor(self.model, model_type="PsiNN_burgers")
        structure = struct_extractor.extract()
        
        # Extract hardware specs
        hw_extractor = HardwareSpecificationExtractor(self.model, struct_extractor)
        
        # Compute specs
        ops = hw_extractor.compute_operations()
        tops = hw_extractor.compute_tops_requirement(target_fps=30.0)
        memory = hw_extractor.compute_memory_requirements()
        power = hw_extractor.estimate_power_consumption()
        
        self.results['hardware_specs'] = {
            'operations': ops,
            'tops_requirement': tops,
            'memory': memory,
            'power': power
        }
        
        print(f"  ✓ Operations: {ops['total_operations']:,}")
        print(f"  ✓ TOPS required: {tops:.6f}")
        print(f"  ✓ Memory: {memory['total_memory_kb']:.2f} KB")
        print(f"  ✓ Power: {power['total_power_mw']:.2f} mW")
        
        return self.results['hardware_specs']
    
    def recommend_platform(self):
        """Recommend Edge AI platform"""
        print(f"\n[PLATFORM] Recommending Edge AI hardware...")
        
        hw_specs = self.results['hardware_specs']
        
        recommender = EdgeAIPlatformRecommender()
        recommendations = recommender.recommend_platform(
            requirements={
                'tops': hw_specs['tops_requirement'],
                'memory_kb': hw_specs['memory']['total_memory_kb'],
                'power_mw': hw_specs['power']['total_power_mw']
            },
            constraints={
                'max_cost_usd': 100,
                'max_power_mw': 10000
            }
        )
        
        self.results['platform_recommendation'] = recommendations[:3]  # Top 3
        
        print(f"  ✓ Top recommendations:")
        for i, platform in enumerate(self.results['platform_recommendation'], 1):
            print(f"    {i}. {platform['name']}: ${platform['cost_usd']}, "
                  f"{platform['power_mw']}mW, Score={platform['score']:.3f}")
        
        return self.results['platform_recommendation']
    
    def analyze_carbon_footprint(self):
        """Analyze lifecycle carbon footprint"""
        print(f"\n[CARBON] Computing lifecycle emissions...")
        
        analyzer = CarbonFootprintAnalyzer()
        
        if self.results['platform_recommendation']:
            platform = self.results['platform_recommendation'][0]  # Best platform
            
            carbon = analyzer.compute_lifecycle_carbon(
                platform=platform,
                deployment_years=3.0,
                duty_cycle=0.5 if self.use_solar else 1.0
            )
            
            self.results['carbon_footprint'] = carbon
            
            print(f"  Platform: {platform['name']}")
            print(f"  Embodied carbon: {carbon['embodied_carbon_kg']:.2f} kg CO2-eq")
            if self.use_solar:
                print(f"  Solar operational: {carbon['solar_operational_kg']:.2f} kg CO2-eq")
                print(f"  Total (solar): {carbon['solar_total_kg']:.2f} kg CO2-eq")
                print(f"  ✓ Carbon saved vs grid: {carbon['carbon_saved_kg']:.2f} kg CO2-eq "
                      f"({carbon['carbon_reduction_percent']:.1f}%)")
            else:
                print(f"  Grid operational: {carbon['grid_operational_kg']:.2f} kg CO2-eq")
                print(f"  Total (grid): {carbon['grid_total_kg']:.2f} kg CO2-eq")
        
        return self.results['carbon_footprint']
    
    def save_results(self, output_dir: Path):
        """Save experiment results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_file = output_dir / f"burgers_{self.results['training_mode']}_results.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {
                'training_mode': self.results['training_mode'],
                'node_num': self.node_num,
                'final_loss': float(self.results['losses'][-1]) if self.results['losses'] else None,
                'training_time': self.results['times'][0] if self.results['times'] else None,
                'solar_stats': self.results['solar_stats'],
                'hardware_specs': self.results['hardware_specs'],
                'platform_recommendation': self.results['platform_recommendation'],
                'carbon_footprint': self.results['carbon_footprint']
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n[SAVE] Results saved to: {results_file}")
        
        # Plot training curve
        self.plot_training_curve(output_dir)
        
        return results_file
    
    def plot_training_curve(self, output_dir: Path):
        """Plot and save training curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['losses'], linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Burgers Equation Training ({self.results["training_mode"].title()})', 
                  fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = output_dir / f"burgers_{self.results['training_mode']}_training.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Training plot saved: {plot_file}")


def run_experiment(use_solar: bool = True, node_num: int = 16):
    """Run complete Burgers solar experiment"""
    print("="*70)
    print("CHAPTER 4: BURGERS EQUATION WITH Ψ-NN ARCHITECTURE")
    print("="*70)
    
    # Create experiment
    experiment = BurgersSolarExperiment(node_num=node_num, use_solar=use_solar)
    
    # Generate data
    experiment.generate_data(n_points=2000)
    
    # Train model
    experiment.train(epochs=3000)
    
    # Extract hardware specifications
    experiment.extract_hardware_specs()
    
    # Recommend platform
    experiment.recommend_platform()
    
    # Analyze carbon footprint
    experiment.analyze_carbon_footprint()
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "chapter4_results" / "burgers"
    experiment.save_results(output_dir)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return experiment.results


if __name__ == "__main__":
    # Run both grid and solar experiments
    print("\n### Running GRID-powered training ###\n")
    results_grid = run_experiment(use_solar=False)
    
    print("\n\n### Running SOLAR-powered training ###\n")
    results_solar = run_experiment(use_solar=True)