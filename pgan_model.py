import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F

from pytorch_GAN_zoo.models.loss_criterions import base_loss_criterions
from pytorch_GAN_zoo.models.loss_criterions.gradient_losses import WGANGPGradientPenalty
from pytorch_GAN_zoo.models.utils.utils import finiteCheck

import utils
import pgan_networks


def init_model(model_dir):
    device = torch.device("cuda:0")
    
    learning_rate = 0.001
    
    discriminator_net = pgan_networks.PGANDiscriminator().to(device)
    generator_net = pgan_networks.PGANGenerator().to(device)
    
    optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator_net.parameters()), betas=[0, 0.99], lr=learning_rate)
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, generator_net.parameters()), betas=[0, 0.99], lr=learning_rate)
    
    optimizer_d.zero_grad()
    optimizer_g.zero_grad()
    
    utils.print_and_log(model_dir, generator_net)
    utils.print_and_log(model_dir, discriminator_net)
    
    return discriminator_net, generator_net, optimizer_d, optimizer_g


def train(discriminator_net, generator_net, optimizer_d, optimizer_g, model_dir):
    device = torch.device("cuda:0")
    
    loss_criterion = base_loss_criterions.WGANGP(device)
    
    epsilon_d = 0.001
    
    n_scales = 5
    model_alpha = 0.0
    alpha_update_cons = 0.0025
    epoch_per_scale = 64
    batch_size = 64
    learning_rate = 0.001
    
    show_scaled_img = False
    normalize_t = torch.Tensor([0.5]).to(device)
    scale_t = torch.Tensor([255]).to(device)
    
    fixed_latent = torch.randn(16, 512).to(device)
    
    generated_images = {0: [], 
                        1: [], 
                        2: [], 
                        3: [], 
                        4: []}
    
    utils.print_and_log(model_dir, f"{datetime.now()} Starting Training")
    
    for scale in range(0, n_scales):
        if scale > 0:
            model_alpha = 1.0
        
        dataloader = utils.get_dataloader(image_size=2**(scale+2), batch_size=batch_size)
        
        utils.print_and_log(model_dir, f"{datetime.now()} Starting scale:{scale}")
        
        if show_scaled_img:
            real_images, _ = next(iter(dataloader))
            plt.imshow((real_images[0].numpy().transpose(1, 2, 0)+1.0)*0.5)
            plt.show()
        
        for batch_step in range(1, (epoch_per_scale*20000//batch_size)+1):
            if batch_step % 25 == 0 and model_alpha > 0:
                model_alpha = model_alpha - alpha_update_cons
                model_alpha = 0.0 if model_alpha < 1e-5 else model_alpha
            
            real_images, _ = next(iter(dataloader))
            batch_images = real_images.to(device)
            
            if model_alpha > 0:
                low_res_real = F.avg_pool2d(batch_images, (2, 2))
                low_res_real = F.interpolate(low_res_real, scale_factor=2, mode='nearest')
                batch_images = model_alpha * low_res_real + (1-model_alpha) * batch_images
            
            discriminator_net.set_alpha(model_alpha)
            generator_net.set_alpha(model_alpha)
            
            optimizer_d.zero_grad()
            
            pred_real_d = discriminator_net(batch_images, False)
            
            loss_d = loss_criterion.getCriterion(pred_real_d, True)
            all_loss_d = loss_d
            
            input_latent = torch.randn(batch_size, 512).to(device)
            
            pred_fake_g = generator_net(input_latent).detach()
            pred_fake_d = discriminator_net(pred_fake_g, False)
            
            loss_d_fake = loss_criterion.getCriterion(pred_fake_d, False)
            
            all_loss_d += loss_d_fake
            
            loss_d_grad = WGANGPGradientPenalty(batch_images, pred_fake_g, discriminator_net, weight=10.0, backward=True)
            
            loss_epsilon = (pred_real_d[:, 0] ** 2).sum() * epsilon_d
            all_loss_d += loss_epsilon
            
            all_loss_d.backward(retain_graph=True)
            finiteCheck(discriminator_net.parameters())
            optimizer_d.step()
            
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            
            input_noise = torch.randn(batch_size, 512).to(device)
            
            pred_fake_g = generator_net(input_noise)
            
            pred_fake_d, phi_g_fake = discriminator_net(pred_fake_g, True)
            
            loss_g_fake = loss_criterion.getCriterion(pred_fake_d, True)
            loss_g_fake.backward(retain_graph=True)
            
            finiteCheck(generator_net.parameters())
            optimizer_g.step()
            
            if batch_step == 1 or batch_step % 100 == 0:
                utils.print_and_log(model_dir, f"{datetime.now()} [{scale}/{n_scales}][{batch_step:05d}/{epoch_per_scale*10000//batch_size}], Alpha:{model_alpha:.4f}, "
                                    f"Loss_G:{loss_g_fake.item():.4f}, Loss_DR:{loss_d.item():.4f}, "
                                    f"Loss_DF:{loss_d_fake.item():.4f}, Loss_DG:{loss_d_grad:.4f}, Loss_DE:{loss_epsilon.item():.4f}")
            
            if batch_step % 100 == 0:
                with torch.no_grad():
                    generated_inputs = generator_net(fixed_latent).detach()
                    generated_images[scale] += [generated_inputs.cpu().numpy().transpose(0, 2, 3, 1)]
        
        if scale < 4:
            discriminator_net.add_scale(depth_new_scale=512)
            generator_net.add_scale(depth_new_scale=512)
            
            discriminator_net.to(device)
            generator_net.to(device)
            
            optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator_net.parameters()), betas=[0, 0.99], lr=learning_rate)
            optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, generator_net.parameters()), betas=[0, 0.99], lr=learning_rate)
            
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            
            utils.print_and_log(model_dir, generator_net)
            utils.print_and_log(model_dir, discriminator_net)
            
    for i in range(n_scales):
        generated_images[i] = np.array(generated_images[i])
    
    return discriminator_net, generator_net, fixed_latent, generated_images
