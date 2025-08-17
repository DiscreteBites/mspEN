import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPatchDiscriminator(nn.Module):
    """
    Spatial discriminator focusing on current spread patterns across neuron bundles
    
    Inspired by PatchGAN but designed for neurogram spatial structure:
    - Processes strips along the spatial dimension (neuron bundles)
    - Outputs patch-wise real/fake decisions for different spatial regions
    - Focuses on physiologically plausible current spread patterns
    """
    
    def __init__(self, input_channels=1, ndf=64):
        super().__init__()
        
        # Spatial-focused convolutions (reduce spatial faster than temporal)
        self.features = nn.Sequential(
            # Input: (batch, 1, 150, T)
            nn.Conv2d(input_channels, ndf, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 75, T)
            
            nn.Conv2d(ndf, ndf*2, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 37, T)
            
            nn.Conv2d(ndf*2, ndf*4, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 256, 18, T)
            
            nn.Conv2d(ndf*4, ndf*8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 512, 9, T)
            
            # Final patch prediction layer
            nn.Conv2d(ndf*8, 1, kernel_size=(3, 3), stride=1, padding=1),
            # → (batch, 1, 9, T) - patch predictions along spatial regions
        )
    
    def forward(self, x):
        # x: (batch, 1, 150, T)
        patches = self.features(x)  # (batch, 1, 9, T)
        return patches  # Each spatial patch gets a real/fake score
    
    def loss(self, pred, target, for_generator=True):
        """PatchGAN-style loss"""
        if for_generator:
            # Generator wants all patches to be classified as real
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target)


class TemporalPatchDiscriminator(nn.Module):
    """
    Temporal discriminator focusing on phoneme dynamics over time
    
    Designed for neurogram temporal structure:
    - Processes strips along the temporal dimension
    - Outputs patch-wise decisions for different time windows
    - Focuses on realistic phoneme onset/offset patterns
    """
    
    def __init__(self, input_channels=1, ndf=64):
        super().__init__()
        
        # Temporal-focused convolutions (reduce temporal faster than spatial)
        self.features = nn.Sequential(
            # Input: (batch, 1, 150, T)
            nn.Conv2d(input_channels, ndf, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 150, T/2)
            
            nn.Conv2d(ndf, ndf*2, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 150, T/4)
            
            nn.Conv2d(ndf*2, ndf*4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 256, 150, T/8)
            
            nn.Conv2d(ndf*4, ndf*8, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 512, 150, T/16)
            
            # Final patch prediction layer
            nn.Conv2d(ndf*8, 1, kernel_size=(3, 3), stride=1, padding=1),
            # → (batch, 1, 150, T/16) - patch predictions along temporal regions
        )
    
    def forward(self, x):
        # x: (batch, 1, 150, T)
        patches = self.features(x)  # (batch, 1, 150, T/16)
        return patches  # Each temporal patch gets a real/fake score
    
    def loss(self, pred, target, for_generator=True):
        """PatchGAN-style loss"""
        if for_generator:
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target)


class GlobalNeurogramDiscriminator(nn.Module):
    """
    Global discriminator for overall neurogram authenticity
    
    Similar to traditional PatchGAN but balanced spatial/temporal reduction:
    - Processes entire neurogram holistically
    - Outputs single real/fake decision for whole neurogram
    - Ensures global coherence and structure
    """
    
    def __init__(self, input_channels=1, ndf=64):
        super().__init__()
        
        # Balanced spatial/temporal reduction
        self.features = nn.Sequential(
            # Input: (batch, 1, 150, T)
            nn.Conv2d(input_channels, ndf, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 75, T/2)
            
            nn.Conv2d(ndf, ndf*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 37, T/4)
            
            nn.Conv2d(ndf*2, ndf*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 256, 18, T/8)
            
            nn.Conv2d(ndf*4, ndf*8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 512, 9, T/16)
            
            # Global average pooling + final decision
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf*8, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1, 150, T)
        decision = self.features(x)  # (batch, 1) - single real/fake score
        return decision
    
    def loss(self, pred, target, for_generator=True):
        """Standard binary classification loss"""
        if for_generator:
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target)


class MultiScaleNeurogramDiscriminator(nn.Module):
    """
    Multi-scale discriminator combining spatial, temporal, and global analysis
    
    Inspired by progressive GAN discriminators:
    - Processes neurogram at multiple scales simultaneously
    - Combines local and global authenticity decisions
    - More robust than single-scale approaches
    """
    
    def __init__(self, input_channels=1, ndf=32):
        super().__init__()
        
        # Fine scale (full resolution)
        self.fine_scale = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
        )
        
        # Medium scale (2x downsampled)
        self.medium_scale = nn.Sequential(
            nn.AvgPool2d(2),  # Downsample input
            nn.Conv2d(input_channels, ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
        )
        
        # Coarse scale (4x downsampled)
        self.coarse_scale = nn.Sequential(
            nn.AvgPool2d(4),  # Downsample input
            nn.Conv2d(input_channels, ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(ndf*6, ndf*4, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf*4, 1)
        )
    
    def forward(self, x):
        # Process at multiple scales
        fine_features = self.fine_scale(x)
        medium_features = self.medium_scale(x)
        coarse_features = self.coarse_scale(x)
        
        # Upsample to common size for fusion
        target_size = fine_features.shape[2:]
        medium_up = F.interpolate(medium_features, size=target_size, mode='bilinear', align_corners=False)
        coarse_up = F.interpolate(coarse_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Fuse multi-scale features
        combined = torch.cat([fine_features, medium_up, coarse_up], dim=1)
        decision = self.fusion(combined)
        
        return decision
    
    def loss(self, pred, target, for_generator=True):
        """Standard binary classification loss"""
        if for_generator:
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target)


# Factory function for easy experimentation
def create_neurogram_discriminator(discriminator_type='spatial', **kwargs):
    """
    Factory function to create different discriminator types
    
    Args:
        discriminator_type: 'spatial', 'temporal', 'global', 'multiscale'
        **kwargs: Additional arguments passed to discriminator constructor
    """
    discriminators = {
        'spatial': SpatialPatchDiscriminator,
        'temporal': TemporalPatchDiscriminator,
        'global': GlobalNeurogramDiscriminator,
        'multiscale': MultiScaleNeurogramDiscriminator
    }
    
    if discriminator_type not in discriminators:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
    
    return discriminators[discriminator_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test all discriminator types
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 150, 50)  # Neurogram batch
    
    discriminator_types = ['spatial', 'temporal', 'global', 'multiscale']
    
    for disc_type in discriminator_types:
        print(f"\nTesting {disc_type} discriminator:")
        
        # Create discriminator
        discriminator = create_neurogram_discriminator(disc_type)
        
        # Forward pass
        output = discriminator(input_tensor)
        
        # Test loss
        real_loss = discriminator.loss(output, torch.ones_like(output), for_generator=False)
        gen_loss = discriminator.loss(output, torch.ones_like(output), for_generator=True)
        
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Real loss: {real_loss.item():.4f}")
        print(f"  Generator loss: {gen_loss.item():.4f}")
        print(f"  Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Usage in your trainer:
"""
# Single discriminator
spatial_disc = create_neurogram_discriminator('spatial')
spatial_opt = torch.optim.Adam(spatial_disc.parameters(), lr=2e-4)

trainer = mspVAETrainer(
    config=config,
    vae_model=vae_model,
    vae_optimizer=vae_optimizer,
    discriminators={'spatial': (spatial_disc, spatial_opt)}
)

# Multiple discriminators
discriminators = {
    'spatial': (create_neurogram_discriminator('spatial'), 
                torch.optim.Adam(spatial_disc.parameters(), lr=2e-4)),
    'temporal': (create_neurogram_discriminator('temporal'),
                 torch.optim.Adam(temporal_disc.parameters(), lr=2e-4)),
    'global': (create_neurogram_discriminator('global'),
               torch.optim.Adam(global_disc.parameters(), lr=1e-4))
}

trainer = mspVAETrainer(
    config=config,
    vae_model=vae_model,
    vae_optimizer=vae_optimizer,
    discriminators=discriminators
)
"""