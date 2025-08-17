from typing import Optional, Tuple, Any, Dict, List, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
import json

from mspEN.modules.types import VisualizeFnType, TestConfig, VAEType, DiscrType
from mspEN.msp import MSP

class mspVAETester:
    def __init__(self,
        config: TestConfig,
        model_dir: Path,
        vae_model: VAEType,
        msp: Optional[MSP] = None,
        discriminators: dict[str, DiscrType] = {},
        visualize_fn: Optional[VisualizeFnType] = None,
        device: Optional[torch.device] = None
    ):
        
        # ==================================
        # Initialise internals
        # ==================================

        # Pull configs
        self.model_dir = model_dir
        self.config = config
        self.visualize_fn = visualize_fn

        # Setup models
        self.vae_model = vae_model
        self.msp = msp
        self.discriminators = discriminators

        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_model.to(self.device)
        for name, discr in self.discriminators.items():
            discr.to(self.device)
        
        # Testing state
        self.test_metrics: dict[str, list[float]] = {}
        self.latent_codes: list[torch.Tensor] = []
        self.reconstructions: list[torch.Tensor] = []
        self.original_data: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        
        # Initialize metrics store
        self.metric_keys = ['Total_Loss', 'Recon_Loss', 'VAE_Loss', 'MSE', 'LPIPS', 'SSIM']
        if self.msp is not None: 
            self.metric_keys.append('MSP_Loss')
        for name in self.discriminators.keys():
            self.metric_keys.append(f'Discriminator_{name}_Score')

        # Initialize metrics tracking dictionaries
        for key in self.metric_keys:
            self.test_metrics[key] = []
        
        # ==================================
        # Log status message
        # ==================================

        print(f"MSP VAE Tester initialized:")
        print(f"  Device: {self.device}")
        print(f"  VAE parameters: {sum(p.numel() for p in self.vae_model.parameters()):,}")
        if self.discriminators:
            for name, disc in self.discriminators.items():
                print(f"  Discriminator '{name}' parameters: {sum(p.numel() for p in disc.parameters()):,}")
        print(f"  MSP enabled: {self.msp is not None}")
        print(f"  Output directory: {self.model_dir}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint for testing"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # ==================================
        # Load model parameters
        # ==================================

        self.vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        
        # Load discriminator states
        if self.discriminators and 'discriminator_state_dicts' in checkpoint:
            for name, discr in self.discriminators.items():
                if name in checkpoint['discriminator_state_dicts']:
                    discr.load_state_dict(checkpoint['discriminator_state_dicts'][name])
        
        # Load MSP state
        if self.msp and 'msp_state_dict' in checkpoint:
            self.msp.load_state_dict(checkpoint['msp_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        return checkpoint
    
    def compute_reconstruction_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
        """Compute various reconstruction quality metrics"""
        metrics = {}
        
        # MSE
        mse = torch.mean((original - reconstructed) ** 2).item()
        metrics['MSE'] = mse
        
        # PSNR
        if mse > 0:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
            metrics['PSNR'] = psnr
        else:
            metrics['PSNR'] = float('inf')
        
        # L1 Loss
        l1_loss = torch.mean(torch.abs(original - reconstructed)).item()
        metrics['L1'] = l1_loss
         
        # Cosine similarity (for latent analysis)
        original_flat = original.view(original.size(0), -1)
        reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(original_flat, reconstructed_flat, dim=1).mean().item()
        metrics['Cosine_Similarity'] = cos_sim
        
        return metrics
    
    def test_reconstruction_quality(self, test_loader: DataLoader):
        """Test reconstruction quality on the test set"""
        self.vae_model.eval()
        for _, discr in self.discriminators.items():
            discr.eval()
        
        # =========================
        # Setup metrics tracking
        # =========================
        batch_metrics: dict[str, list[float]] = {k: [] for k in self.metric_keys}
        all_recon_metrics: dict[str, list[float]] = defaultdict(list)
        
        # Clear previous results
        self.latent_codes.clear()
        self.reconstructions.clear()
        self.original_data.clear()
        self.labels.clear()
        
        # =====================
        # Test loop
        # =====================
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(test_loader, desc='Testing Reconstruction Quality')):
                data = data.to(self.device)
                label = label.to(self.device)
                batch_size = data.size(0)
                
                # =====================
                # Forward pass of VAE
                # =====================
                prod, z, mu, logvar = self.vae_model(data)
                
                # Store for later analysis
                self.latent_codes.append(z.cpu())
                self.reconstructions.append(prod.cpu())
                self.original_data.append(data.cpu())
                self.labels.append(label.cpu())
                
                # Calculate VAE losses
                vae_loss, l_rec, l_vae = self.vae_model.loss(prod, data, mu, logvar)
                
                # Get MSP loss
                l_msp = torch.zeros((), device=self.device)
                if self.msp is not None:
                    l_msp = self.msp.loss(data, label, z)
                
                # Get discriminator scores
                discr_scores = {}
                for name, discr in self.discriminators.items():
                    fake_score = discr(prod).mean()
                    discr_scores[name] = fake_score.item()
                    batch_metrics[f'Discriminator_{name}_Score'].append(fake_score.item())
                
                # Calculate reconstruction metrics
                recon_metrics = self.compute_reconstruction_metrics(data, prod)
                for metric_name, value in recon_metrics.items():
                    all_recon_metrics[metric_name].append(value)
                
                # Store main losses
                total_loss = vae_loss + l_msp
                batch_metrics['Total_Loss'].append(total_loss.item())
                batch_metrics['Recon_Loss'].append(l_rec.item() if isinstance(l_rec, torch.Tensor) else l_rec)
                batch_metrics['VAE_Loss'].append(l_vae.item() if isinstance(l_vae, torch.Tensor) else l_vae)
                
                if self.msp is not None:
                    batch_metrics['MSP_Loss'].append(l_msp.item())
                
                # Add reconstruction metrics to main metrics
                if 'MSE' in recon_metrics:
                    batch_metrics['MSE'].append(recon_metrics['MSE'])
        
        # =====================
        # Calculate statistics
        # =====================
        test_stats = {}
        for metric_name, values in batch_metrics.items():
            if values:  # Only process metrics that have values
                test_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Add additional reconstruction metrics
        for metric_name, values in all_recon_metrics.items():
            if metric_name not in test_stats:  # Don't overwrite existing metrics
                test_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Store metrics for later use
        for metric_name, stats in test_stats.items():
            self.test_metrics[metric_name] = [stats['mean']]  # Store mean for consistency with training code
        
        return test_stats
    
    def analyze_latent_space(self, num_samples: int = 1000):
        """Analyze the learned latent space"""
        if not self.latent_codes:
            print("No latent codes available. Run test_reconstruction_quality first.")
            return {}
        
        # Concatenate all latent codes
        all_z = torch.cat(self.latent_codes[:num_samples//len(self.latent_codes[0]) + 1])[:num_samples]
        all_labels = torch.cat(self.labels[:num_samples//len(self.labels[0]) + 1])[:num_samples]
        
        latent_stats = {}
        
        # Basic statistics
        latent_stats['latent_dim'] = all_z.shape[1]
        latent_stats['mean_activation'] = torch.mean(all_z, dim=0).tolist()
        latent_stats['std_activation'] = torch.std(all_z, dim=0).tolist()
        latent_stats['activation_range'] = {
            'min': torch.min(all_z, dim=0)[0].tolist(),
            'max': torch.max(all_z, dim=0)[0].tolist()
        }
        
        # Correlation analysis
        correlation_matrix = torch.corrcoef(all_z.T)
        latent_stats['avg_correlation'] = torch.mean(torch.abs(correlation_matrix)).item()
        latent_stats['max_correlation'] = torch.max(torch.abs(correlation_matrix - torch.eye(correlation_matrix.shape[0]))).item()
        
        # Effective dimensionality (participation ratio)
        var_per_dim = torch.var(all_z, dim=0)
        participation_ratio = (torch.sum(var_per_dim) ** 2) / torch.sum(var_per_dim ** 2)
        latent_stats['participation_ratio'] = participation_ratio.item()
        
        return latent_stats
    
    def interpolate_latent(self, num_steps: int = 10, num_pairs: int = 5):
        """Perform latent space interpolation between random pairs"""
        if not self.latent_codes:
            print("No latent codes available. Run test_reconstruction_quality first.")
            return None
        
        self.vae_model.eval()
        
        # Get random pairs of latent codes
        all_z = torch.cat(self.latent_codes)
        indices = torch.randperm(len(all_z))[:num_pairs*2].reshape(-1, 2)
        
        interpolations = []
        
        with torch.no_grad():
            for i in range(num_pairs):
                z1, z2 = all_z[indices[i]]
                z1, z2 = z1.to(self.device), z2.to(self.device)
                
                # Create interpolation
                alphas = torch.linspace(0, 1, num_steps)
                interp_sequence = []
                
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    
                    # Decode interpolated latent
                    if hasattr(self.vae_model, 'decode'):
                        x_interp = self.vae_model.decode(z_interp.unsqueeze(0))
                    else:
                        # Assume the model has a decoder attribute
                        x_interp = self.vae_model.decoder(z_interp.unsqueeze(0))
                    
                    interp_sequence.append(x_interp.squeeze(0).cpu())
                
                interpolations.append(torch.stack(interp_sequence))
        
        return torch.stack(interpolations)  # Shape: [num_pairs, num_steps, ...]
    
    def generate_samples(self, num_samples: int = 64):
        """Generate new samples from the learned distribution"""
        self.vae_model.eval()
        
        with torch.no_grad():
            # Sample from prior
            if hasattr(self.vae_model, 'latent_dim'):
                latent_dim = self.vae_model.latent_dim
            else:
                # Infer from existing latent codes
                latent_dim = self.latent_codes[0].shape[1] if self.latent_codes else 64
            
            z_samples = torch.randn(num_samples, latent_dim).to(self.device)
            
            # Decode samples
            if hasattr(self.vae_model, 'decode'):
                generated = self.vae_model.decode(z_samples)
            else:
                generated = self.vae_model.decoder(z_samples)
        
        return generated.cpu()
    
    def visualize_test_results(self, num_samples: int = 8):
        """Visualize test results including reconstructions and generations"""
        if self.visualize_fn is None:
            print("No visualization function provided. Skipping visualization.")
            return
        
        if not self.original_data:
            print("No test data available. Run test_reconstruction_quality first.")
            return
        
        # Visualize reconstructions
        original_batch = self.original_data[0][:num_samples]
        label_batch = self.labels[0][:num_samples]
        
        self.visualize_fn(
            self.vae_model, original_batch, label_batch,
            self.model_dir, 'test_reconstructions', self.device
        )
        
        # Visualize generated samples
        generated_samples = self.generate_samples(num_samples)
        fake_labels = torch.zeros(num_samples, dtype=torch.long)  # Placeholder labels
        
        self.visualize_fn(
            self.vae_model, generated_samples, fake_labels,
            self.model_dir, 'test_generations', self.device
        )
    
    def plot_test_metrics(self, test_stats: dict):
        """Plot test metrics and statistics"""
        if not test_stats:
            return
        
        # Filter metrics that have meaningful statistics
        plottable_metrics = {k: v for k, v in test_stats.items() 
                           if isinstance(v, dict) and 'mean' in v}
        
        if not plottable_metrics:
            return
        
        # Create subplots for different metric categories
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Loss metrics
        loss_metrics = {k: v for k, v in plottable_metrics.items() if 'Loss' in k}
        if loss_metrics:
            names = list(loss_metrics.keys())
            means = [loss_metrics[name]['mean'] for name in names]
            stds = [loss_metrics[name]['std'] for name in names]
            
            axes[0].bar(names, means, yerr=stds, capsize=5, alpha=0.7)
            axes[0].set_title('Loss Metrics')
            axes[0].set_ylabel('Loss Value')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Reconstruction quality metrics
        recon_metrics = {k: v for k, v in plottable_metrics.items() 
                        if k in ['MSE', 'PSNR', 'L1', 'Cosine_Similarity']}
        if recon_metrics:
            names = list(recon_metrics.keys())
            means = [recon_metrics[name]['mean'] for name in names]
            stds = [recon_metrics[name]['std'] for name in names]
            
            axes[1].bar(names, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
            axes[1].set_title('Reconstruction Quality Metrics')
            axes[1].set_ylabel('Metric Value')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Discriminator scores
        discr_metrics = {k: v for k, v in plottable_metrics.items() if 'Discriminator' in k}
        if discr_metrics:
            names = [k.replace('Discriminator_', '').replace('_Score', '') for k in discr_metrics.keys()]
            means = [list(discr_metrics.values())[i]['mean'] for i in range(len(names))]
            stds = [list(discr_metrics.values())[i]['std'] for i in range(len(names))]
            
            axes[2].bar(names, means, yerr=stds, capsize=5, alpha=0.7, color='green')
            axes[2].set_title('Discriminator Scores')
            axes[2].set_ylabel('Score')
            axes[2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Distribution of main metrics
        if 'Total_Loss' in plottable_metrics:
            # This would require the raw values, so we'll show a summary instead
            summary_data = []
            summary_labels = []
            for name in ['Total_Loss', 'Recon_Loss', 'VAE_Loss']:
                if name in plottable_metrics:
                    summary_data.append([
                        plottable_metrics[name]['min'],
                        plottable_metrics[name]['mean'],
                        plottable_metrics[name]['max']
                    ])
                    summary_labels.append(name)
            
            if summary_data:
                x = np.arange(len(summary_labels))
                width = 0.25
                
                mins = [d[0] for d in summary_data]
                means = [d[1] for d in summary_data]
                maxs = [d[2] for d in summary_data]
                
                axes[3].bar(x - width, mins, width, label='Min', alpha=0.7)
                axes[3].bar(x, means, width, label='Mean', alpha=0.7)
                axes[3].bar(x + width, maxs, width, label='Max', alpha=0.7)
                
                axes[3].set_title('Loss Statistics')
                axes[3].set_ylabel('Value')
                axes[3].set_xticks(x)
                axes[3].set_xticklabels(summary_labels, rotation=45)
                axes[3].legend()
        
        # Hide unused subplots
        for i in range(len(axes)):
            if i >= 4:
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'test_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_test_results(self, test_stats: dict, latent_stats: dict):
        """Save test results to JSON file"""
        results = {
            'test_metrics': test_stats,
            'latent_analysis': latent_stats,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
        }
        
        with open(self.model_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Test results saved to: {self.model_dir / 'test_results.json'}")
    
    def test(self, test_loader: DataLoader, checkpoint_path: Optional[Path] = None):
        """Main testing pipeline"""
        
        # ==================================
        # Load checkpoint if provided
        # ==================================
        
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        
        print("Starting comprehensive testing...")
        
        # ==================================
        # Test reconstruction quality
        # ==================================
        
        print("Testing reconstruction quality...")
        test_stats = self.test_reconstruction_quality(test_loader)
        
        # ==================================
        # Analyze latent space
        # ==================================
        
        print("Analyzing latent space...")
        latent_stats = self.analyze_latent_space()
        
        # ==================================
        # Visualizations
        # ==================================
        
        print("Generating visualizations...")
        self.visualize_test_results()
        
        # Plot interpolations if we have latent codes
        if self.latent_codes:
            interpolations = self.interpolate_latent()
            if interpolations is not None:
                # Save interpolation visualization logic would go here
                print("Latent interpolations computed")
        
        # ==================================
        # Save results
        # ==================================
        
        self.plot_test_metrics(test_stats)
        self.save_test_results(test_stats, latent_stats)
        
        # ==================================
        # Print summary
        # ==================================
        
        print(f'\nTesting completed!')
        print(f'  Total Loss: {test_stats.get("Total_Loss", {}).get("mean", "N/A"):.4f}')
        print(f'  Reconstruction MSE: {test_stats.get("MSE", {}).get("mean", "N/A"):.4f}')
        if 'PSNR' in test_stats:
            print(f'  PSNR: {test_stats["PSNR"]["mean"]:.2f} dB')
        if latent_stats:
            print(f'  Latent Dim: {latent_stats.get("latent_dim", "N/A")}')
            print(f'  Effective Dim (PR): {latent_stats.get("participation_ratio", "N/A"):.2f}')
        print(f'Results saved to: {self.model_dir}')
        
        return test_stats, latent_stats

# exports
__all__ = ['mspVAETester']