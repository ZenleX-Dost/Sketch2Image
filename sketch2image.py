import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import vgg19
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.checkpoint import checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
from torch.cuda.amp import autocast, GradScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ImprovedGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        
        # Improved initial block with more channels
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Deeper downsampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # More residual blocks for better feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels*8) for _ in range(12)
        ])
        
        # Multiple attention blocks at different scales
        self.attention1 = AttentionBlock(base_channels*8)
        self.attention2 = AttentionBlock(base_channels*4)
        
        # Progressive upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Refined output block
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial and downsampling
        x1 = checkpoint(self.initial, x) #changehere
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        
        # Residual blocks with intermediate attention
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            if i == 5:  # Add attention in the middle
                x = self.attention1(x)
        
        # Progressive upsampling with skip connections and attention
        x = self.up1(x)
        x = self.attention2(x + x3)  # Skip connection
        x = self.up2(x)
        x = x + x2  # Skip connection
        x = self.up3(x)
        x = x + x1  # Skip connection
        
        return self.final(x)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_channels=4, base_channels=64):  # 4 channels: 3 for image + 1 for sketch
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final = nn.Conv2d(base_channels*8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, sketch, image):
        # Concatenate sketch and image along channel dimension
        x = torch.cat([sketch, image], dim=1)
        x = self.initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final(x)
        return x

class ImprovedVGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:35]).eval()
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, generated, target):
        if generated.device.type == 'cuda':
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            
        generated = (generated + 1) / 2  # Transform from [-1, 1] to [0, 1]
        target = (target + 1) / 2
        
        # Normalize with ImageNet stats
        generated = (generated - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        generated_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        
        return F.mse_loss(generated_features, target_features)

class SketchDataset(Dataset):
    def __init__(self, sketch_dir, image_dir, mode='train', image_size=256):
        self.sketch_dir = Path(sketch_dir)
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.image_size = image_size
        
        # Get all image pairs
        self.sketch_paths = sorted(list(self.sketch_dir.glob('*.png')))
        
        # Training augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch_path = self.sketch_paths[idx]
        image_path = self.image_dir / sketch_path.name
        
        # Load images
        sketch = np.array(Image.open(sketch_path).convert('L'))
        real_image = np.array(Image.open(image_path).convert('RGB'))
        
        # Apply transformations
        transformed = self.transform(image=real_image, mask=sketch)
        real_image = transformed['image'].float() / 127.5 - 1
        sketch = transformed['mask'].float().unsqueeze(0) / 127.5 - 1
        
        return sketch, real_image

class GradientPenalty:
    def __call__(self, discriminator, real_samples, fake_samples, sketches):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        d_interpolates = discriminator(sketches, interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class TrainingConfig:
    def __init__(self):
        self.image_size = 256
        self.batch_size = 16  # Increased batch size
        self.num_epochs = 200  # More epochs
        self.lr_g = 0.0002
        self.lr_d = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_l1 = 100
        self.lambda_perceptual = 10
        self.lambda_gp = 10
        self.save_interval = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 8  # Increased workers
        self.accumulation_steps = 2  # Gradient accumulation
        self.perceptual_loss = ImprovedVGGLoss().to(self.device)

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, device):
    """Load checkpoint and return the starting epoch."""
    try:
        logger.info(f"Attempting to load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Debug information
        logger.info(f"Checkpoint contents: {checkpoint.keys()}")
        
        # Load model states
        generator.load_state_dict(checkpoint['generator_state_dict'])
        logger.info("Generator state loaded successfully")
        
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        logger.info("Discriminator state loaded successfully")
        
        # Load optimizer states
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        logger.info("Generator optimizer state loaded successfully")
        
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        logger.info("Discriminator optimizer state loaded successfully")
        
        # Load scheduler states if they exist
        if scheduler_g is not None and 'scheduler_g_state_dict' in checkpoint:
            scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            logger.info("Generator scheduler state loaded successfully")
        
        if scheduler_d is not None and 'scheduler_d_state_dict' in checkpoint:
            scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
            logger.info("Discriminator scheduler state loaded successfully")
        
        epoch = checkpoint['epoch']
        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
        return epoch + 1
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        logger.error("Starting training from scratch")
        return 0

def compute_discriminator_loss(discriminator, real_images, fake_images, sketches, lambda_gp=10):
    # Compute discriminator outputs for real and fake images
    real_validity = discriminator(sketches, real_images)
    fake_validity = discriminator(sketches, fake_images)

    # Wasserstein loss
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

    # Gradient penalty for improved training stability
    gp = gradient_penalty(discriminator, real_images, fake_images, sketches)

    # Total loss with gradient penalty term
    d_loss += lambda_gp * gp
    return d_loss
def compute_generator_loss(generator, discriminator, fake_images, real_images, sketches, config):
    # Compute discriminator's prediction for fake images
    fake_validity = discriminator(sketches, fake_images)
    
    # Adversarial loss (Wasserstein)
    g_loss = -torch.mean(fake_validity)
    
    # L1 Loss between generated and real images
    l1_loss = F.l1_loss(fake_images, real_images)
    
    # Perceptual loss if available
    perceptual_loss = config.perceptual_loss(fake_images, real_images) if hasattr(config, 'perceptual_loss') else 0
    
    # Combine losses
    g_loss += config.lambda_l1 * l1_loss + config.lambda_perceptual * perceptual_loss
    return g_loss
def gradient_penalty(discriminator, real_images, fake_images, sketches, lambda_gp=10):
    batch_size = real_images.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    disc_interpolated = discriminator(sketches, interpolated)

    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def train(config, resume_from=None):
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = SketchDataset(
        sketch_dir="C:/Users/amine/Desktop/Sketch2Image/sketches",
        image_dir="C:/Users/amine/Desktop/Sketch2Image/images",     
        image_size=config.image_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize models
    generator = ImprovedGenerator().to(config.device)
    discriminator = ImprovedDiscriminator().to(config.device)
    
    # Parameter groups for generator with different learning rates
    g_params = [
        {'params': generator.initial.parameters(), 'lr': config.lr_g * 0.5},
        {'params': generator.down1.parameters(), 'lr': config.lr_g * 0.75},
        {'params': generator.down2.parameters(), 'lr': config.lr_g * 0.75},
        {'params': generator.res_blocks.parameters(), 'lr': config.lr_g},
        {'params': generator.up1.parameters(), 'lr': config.lr_g * 0.75},
        {'params': generator.up2.parameters(), 'lr': config.lr_g * 0.75},
        {'params': generator.final.parameters(), 'lr': config.lr_g * 0.5}
    ]
    
    # Optimizers
    optimizer_g = optim.AdamW(g_params, betas=(config.beta1, config.beta2))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.OneCycleLR(
        optimizer_g,
        max_lr=config.lr_g,
        epochs=config.num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.2
    )
    
    scheduler_d = optim.lr_scheduler.OneCycleLR(
        optimizer_d,
        max_lr=config.lr_d,
        epochs=config.num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.2
    )
    scaler = torch.amp.GradScaler('cuda')
    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_checkpoint(
            resume_from,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            config.device
        )
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        generator.train()
        discriminator.train()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch_idx, (sketches, real_images) in enumerate(progress_bar):
            sketches, real_images = sketches.to(config.device), real_images.to(config.device)
            
            with torch.amp.autocast('cuda'):
                # Train Discriminator
                optimizer_d.zero_grad()
                fake_images = generator(sketches)
                d_loss = compute_discriminator_loss(discriminator, real_images, fake_images.detach(), sketches)
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                
                # Train Generator (less frequent updates)
                if batch_idx % config.accumulation_steps == 0:
                    optimizer_g.zero_grad()
                    # Pass config to compute_generator_loss
                    g_loss = compute_generator_loss(
                        generator, 
                        discriminator, 
                        fake_images, 
                        real_images, 
                        sketches, 
                        config  # Added config parameter here
                    )
                    scaler.scale(g_loss).backward()
                    scaler.step(optimizer_g)
                    scheduler_g.step()
            
            # Update scaler for AMP
            scaler.update()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item()
            })
            
        
        # Save samples
        if (epoch + 1) % config.save_interval == 0:
            with torch.no_grad():
                generator.eval()
                sample_sketches = next(iter(dataloader))[0][:8].to(config.device)
                sample_generated = generator(sample_sketches)
                save_image(
                    sample_generated,
                    f'samples/epoch_{epoch+1}.png',
                    nrow=4,
                    normalize=True
                )
        
                # In your training loop, when saving checkpoints:
        checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict(),
        'scheduler_d_state_dict': scheduler_d.state_dict(),
    }

        # Save checkpoint
        torch.save(
            checkpoint,
            f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
        )
        torch.cuda.empty_cache()
    
    logger.info(f"Training completed after {config.num_epochs} epochs!")
    
    # Save final models
    torch.save(generator.state_dict(), 'checkpoints/final_generator.pt')
    torch.save(discriminator.state_dict(), 'checkpoints/final_discriminator.pt')
    
    return generator, discriminator
def main():
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize config
    config = TrainingConfig()
    
    # Check for latest checkpoint
    checkpoint_dir = 'checkpoints'
    latest_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            logger.info("Automatically resuming from latest checkpoint...")
    
    # Start training with checkpoint if available, otherwise from scratch
    if latest_checkpoint:
        logger.info("Resuming training...")
        generator, discriminator = train(config, resume_from=latest_checkpoint)
    else:
        logger.info("Starting training from scratch...")
        generator, discriminator = train(config)
        
    logger.info("Training completed!")
    
if __name__ == "__main__":
    main()