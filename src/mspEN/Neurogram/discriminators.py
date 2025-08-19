import torch
import torch.nn as nn
import torch.nn.functional as F

from mspEN.modules.types import DiscrType

class SpatialPatchDiscriminator(DiscrType):
    """
    Spatial discriminator focusing on current spread patterns across neuron bundles
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
    
    def loss(self, pred, target, train_discriminator=True):
        """Fixed PatchGAN-style loss"""
        # pred: (batch, 1, 9, T), target: same shape
        if not train_discriminator:
            # Generator training: wants discriminator to think all patches are real
            target = torch.ones_like(pred)
        # For discriminator training, target is passed in (real=1, fake=0)
        
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


class TemporalPatchDiscriminator(DiscrType):
    """
    Temporal discriminator focusing on phoneme dynamics over time
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
    
    def loss(self, pred, target, train_discriminator=True):
        """Fixed PatchGAN-style loss"""
        # pred: (batch, 1, 150, T/16), target: same shape
        if not train_discriminator:
            # Generator training: wants discriminator to think all patches are real
            target = torch.ones_like(pred)
        # For discriminator training, target is passed in (real=1, fake=0)
        
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


class GlobalNeurogramDiscriminator(DiscrType):
    """
    RECOMMENDED: Simplified global discriminator for overall neurogram authenticity
    """
    
    def __init__(self, input_channels=1, ndf=64):
        super().__init__()
        
        # Simplified architecture - balanced spatial/temporal reduction
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
            
            # Global average pooling + final decision
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf*4, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1, 150, T)
        decision = self.features(x)  # (batch, 1) - single real/fake score
        return decision
    
    def loss(self, pred, target, train_discriminator=True):
        """Fixed standard binary classification loss"""
        # pred: (batch, 1), target: (batch, 1)
        if not train_discriminator:
            # Generator training: wants discriminator to think samples are real
            target = torch.ones_like(pred)
        # For discriminator training, target is passed in (real=1, fake=0)
        
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


class BasicNeurogramDiscriminator(DiscrType):
    """
    RECOMMENDED: Simple, effective discriminator for neurograms
    
    Key improvements:
    - Fewer parameters, faster training
    - Balanced spatial/temporal processing  
    - Single output for simplicity
    - Proven architecture patterns
    """
    
    def __init__(self, input_channels=1, ndf=32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1: Initial feature extraction
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 32, 75, T/2)
            
            # Layer 2: More features, batch norm
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 37, T/4)
            
            # Layer 3: Higher level features
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 18, T/8)
            
            # Layer 4: Final feature compression
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 256, 9, T/16)
            
            # Global pooling and classification
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf*8, 1)
        )
    
    def forward(self, x):
        return self.features(x)
    
    def loss(self, pred, target, train_discriminator=True):
        """Simple, correct loss function"""
        if not train_discriminator:
            # Generator wants to fool discriminator (all real)
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


# Alternative: Spectral normalization version for training stability
class SpectralNormDiscriminator(DiscrType):
    """
    ALTERNATIVE: Discriminator with spectral normalization for stability
    """
    
    def __init__(self, input_channels=1, ndf=32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.utils.spectral_norm(
                nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, 1))
        )
    
    def forward(self, x):
        return self.features(x)
    
    def loss(self, pred, target, train_discriminator=True):
        if not train_discriminator:
            target = torch.ones_like(pred)
        
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')