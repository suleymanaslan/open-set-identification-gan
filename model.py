import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

import utils
import networks


def init_model(model_dir):
    device = torch.device("cuda:0")
    
    generator_net = networks.Generator().to(device)
    discriminator_net = networks.Discriminator().to(device)
    
    generator_net.apply(networks.weights_init)
    discriminator_net.apply(networks.weights_init)
    
    utils.print_and_log(model_dir, generator_net)
    utils.print_and_log(model_dir, discriminator_net)
    
    learning_rate = 0.0002
    beta1 = 0.5
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    return discriminator_net, generator_net, discriminator_optimizer, generator_optimizer


def train(discriminator_net, generator_net, discriminator_optimizer, generator_optimizer, model_dir, dataloader, latent_size, num_epochs, batch_size):
    device = torch.device("cuda:0")
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(8, latent_size, 1, 1, device=device)
    
    real_label = torch.full((batch_size,), 1, dtype=torch.float32, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float32, device=device)
    
    generator_losses = []
    discriminator_losses = []
    generated_images = None
    train_iters = 0
    torch_ones = torch.tensor([1.0]).to(device)
    
    utils.print_and_log(model_dir, f"{datetime.now()} Starting Training")
    for epoch in range(num_epochs):
        for batch_index, (real_images, _) in enumerate(dataloader, 0):
            discriminator_net.zero_grad()
            
            real_inputs = real_images.to(device)
            
            output = discriminator_net(real_inputs).view(-1)
            
            discriminator_loss_real = criterion(output, real_label)
            
            discriminator_loss_real.backward()
            real_output = output.mean().item()
            
            noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
            
            generated_inputs = generator_net(noise)

            output = discriminator_net(generated_inputs.detach()).view(-1)

            discriminator_loss_fake = criterion(output, fake_label)

            discriminator_loss_fake.backward()
            fake_output1 = output.mean().item()

            discriminator_loss = discriminator_loss_real + discriminator_loss_fake

            discriminator_optimizer.step()

            generator_net.zero_grad()

            output = discriminator_net(generated_inputs).view(-1)

            generator_loss = criterion(output, real_label)

            generator_loss.backward()
            fake_output2 = output.mean().item()

            generator_optimizer.step()

            if batch_index % 50 == 0:
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch:02d}/{num_epochs}][{batch_index:04d}/{len(dataloader)}]\t"
                                    f"D_Loss:{discriminator_loss.item():.4f} G_Loss:{generator_loss.item():.4f} Real:{real_output:.4f} "
                                    f"Fake1:{fake_output1:.4f} Fake2:{fake_output2:.4f}")

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            if (train_iters % 2000 == 0) or ((epoch == num_epochs-1) and (batch_index == len(dataloader)-1)):
                with torch.no_grad():
                    generated_inputs = generator_net(fixed_noise).detach().cpu().numpy()
                    generated_images = generated_inputs if generated_images is None else np.concatenate((generated_images, generated_inputs), axis=0)

            train_iters += 1
        
    return generator_losses, discriminator_losses, generated_images
