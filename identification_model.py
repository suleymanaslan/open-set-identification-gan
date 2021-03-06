import copy
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import utils
import networks


def init_model(model_dir, nb_of_classes):
    device = torch.device("cuda:0")
    
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    resnext50_32x4d.fc = nn.Linear(resnext50_32x4d.fc.in_features, nb_of_classes)
    resnext50_32x4d = resnext50_32x4d.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(resnext50_32x4d.parameters(), lr=0.001)
    
    utils.print_and_log(model_dir, resnext50_32x4d)
    
    return resnext50_32x4d, criterion, optimizer


def init_gan_model(model_dir, nb_of_classes):
    device = torch.device("cuda:0")
    
    generator_net = networks.Generator().to(device)
    discriminator_net = networks.ClassifierDiscriminator(nb_of_classes=nb_of_classes).to(device)
    
    generator_net.apply(networks.weights_init)
    discriminator_net.apply(networks.weights_init)
    
    utils.print_and_log(model_dir, generator_net)
    utils.print_and_log(model_dir, discriminator_net)
    
    learning_rate = 0.0002
    beta1 = 0.5
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    criterion = nn.CrossEntropyLoss()
    
    return discriminator_net, generator_net, criterion, discriminator_optimizer, generator_optimizer


def train(model_dir, network, criterion, optimizer, data_dirs, dataloaders, unknown_id, num_epochs):
    device = torch.device("cuda:0")
    
    best_model = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    
    utils.print_and_log(model_dir, f"{datetime.now()} Starting Training")
    
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch+1}/{num_epochs} Training")
                network.train()
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch+1}/{num_epochs} Validation")
                network.eval()
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_iter = 0
            
            epoch_unknown_correct = 0
            epoch_unknown_total = 0
            epoch_known_correct = 0
            epoch_known_total = 0
            
            for batch_index, (inputs, labels) in enumerate(dataloaders[data_dirs[phase]]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = network(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    unknown_correct = torch.sum((preds == unknown_id) & (labels.data == unknown_id)).item()
                    unknown_total = torch.sum(labels.data == unknown_id).item()
                    known_correct = torch.sum(preds == labels.data).item() - unknown_correct
                    known_total = inputs.size(0) - unknown_total
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        if batch_index % (len(dataloaders[data_dirs[phase]]) // 10) == 0:
                            utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}][{batch_index:04d}/{len(dataloaders[data_dirs[phase]])}] "
                                                f"Loss:{loss.item():.4f} Acc:{torch.sum(preds == labels.data).item()/inputs.size(0):.4f} "
                                                f"Corrects(Known):{known_correct:02d}/{known_total:02d} "
                                                f"Corrects(Unknown):{unknown_correct:02d}/{unknown_total:02d}")
                    
                    epoch_loss += loss.item()
                    epoch_acc += torch.sum(preds == labels.data).item()/inputs.size(0)
                    epoch_iter += 1
                    
                    epoch_unknown_correct += unknown_correct
                    epoch_unknown_total += unknown_total
                    epoch_known_correct += known_correct
                    epoch_known_total += known_total
            
            epoch_loss = epoch_loss / epoch_iter
            epoch_acc = epoch_acc / epoch_iter
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
            
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}] Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
                val_losses.append(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(network.state_dict())
    
    utils.print_and_log(model_dir, f"{datetime.now()} Best Val Acc:{best_acc:4f}")
    network.load_state_dict(best_model)
    
    return network, best_acc, train_losses, val_losses


def train_batch_iterator(model_dir, network, criterion, optimizer, batch_iter, unknown_id, num_epochs):
    device = torch.device("cuda:0")
    
    best_model = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    
    scale_t = torch.Tensor([255]).to(device)
    normalize_mean_t = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    normalize_std_t = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    batch_size = batch_iter.batch_size
    train_epoch_step = int(round(batch_iter.size / batch_size))
    display_steps = np.linspace(1, train_epoch_step, 20, endpoint=True).astype(np.uint32)
    
    val_data = torch.load("data/preprocessed_data/val_data.pt")
    val_labels = torch.load("data/preprocessed_data/val_labels.pt")
    val_epoch_step = int(round(val_data.shape[0] / batch_size))
    
    utils.print_and_log(model_dir, f"{datetime.now()} Starting Training")
    
    for epoch_i in range(1, num_epochs+1):
        for phase in ["train", "val"]:
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch_i}/{num_epochs} Training")
                network.train()
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch_i}/{num_epochs} Validation")
                network.eval()
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_iter = 0
            
            epoch_unknown_correct = 0
            epoch_unknown_total = 0
            epoch_known_correct = 0
            epoch_known_total = 0
            
            for step_i in range(0, train_epoch_step if phase == "train" else val_epoch_step):
                if phase == "train":
                    batch_inputs, batch_labels = batch_iter.next_batch()
                    batch_inputs = ((batch_inputs.type(torch.FloatTensor).to(device) / scale_t) - normalize_mean_t) / normalize_std_t
                    batch_labels = batch_labels.type(torch.long).to(device)
                else:
                    batch_inputs = ((val_data[step_i*batch_size:step_i*batch_size+batch_size].type(torch.FloatTensor).to(device) / scale_t) - normalize_mean_t) / normalize_std_t
                    batch_labels = val_labels[step_i*batch_size:step_i*batch_size+batch_size].type(torch.long).to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = network(batch_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, batch_labels)
                    
                    unknown_correct = torch.sum((preds == unknown_id) & (batch_labels.data == unknown_id)).item()
                    unknown_total = torch.sum(batch_labels.data == unknown_id).item()
                    known_correct = torch.sum(preds == batch_labels.data).item() - unknown_correct
                    known_total = batch_inputs.size(0) - unknown_total
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        if (step_i+1) in display_steps:
                            utils.print_and_log(model_dir, f"{datetime.now()} Epoch:{epoch_i:02d}, Step:{step_i+1:04d}/{train_epoch_step}, "
                                                f"Iter:{(step_i+1)*batch_size:05d}/{train_epoch_step*batch_size}, "
                                                f"Loss:{loss.item():.4f} Acc:{torch.sum(preds == batch_labels.data).item()/batch_inputs.size(0):.4f} "
                                                f"Corrects(Known):{known_correct:02d}/{known_total:02d} "
                                                f"Corrects(Unknown):{unknown_correct:02d}/{unknown_total:02d}")
                    
                    epoch_loss += loss.item()
                    epoch_acc += torch.sum(preds == batch_labels.data).item()/batch_inputs.size(0)
                    epoch_iter += 1
                    
                    epoch_unknown_correct += unknown_correct
                    epoch_unknown_total += unknown_total
                    epoch_known_correct += known_correct
                    epoch_known_total += known_total
            
            epoch_loss = epoch_loss / epoch_iter
            epoch_acc = epoch_acc / epoch_iter
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch_i:02d}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
            
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch_i:02d}/{num_epochs}] Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
                val_losses.append(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(network.state_dict())
    
    utils.print_and_log(model_dir, f"{datetime.now()} Best Val Acc:{best_acc:4f}")
    network.load_state_dict(best_model)
    
    return network, best_acc, train_losses, val_losses


def train_gan(model_dir, discriminator_net, generator_net, criterion, discriminator_optimizer, generator_optimizer, data_dirs, dataloaders, unknown_id, latent_size, num_epochs, batch_size):
    device = torch.device("cuda:0")
    
    fixed_noise = torch.randn(8, latent_size, 1, 1, device=device)
    
    best_model = copy.deepcopy(discriminator_net.state_dict())
    best_model_g = copy.deepcopy(generator_net.state_dict())
    best_acc = 0.0
    
    train_losses = []
    train_losses_g = []
    val_losses = []
    
    utils.print_and_log(model_dir, f"{datetime.now()} Starting Training")
    
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch+1}/{num_epochs} Training")
                discriminator_net.train()
                generator_net.train()
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} Epoch {epoch+1}/{num_epochs} Validation")
                discriminator_net.eval()
                generator_net.eval()
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_iter = 0
            
            epoch_unknown_correct = 0
            epoch_unknown_total = 0
            epoch_known_correct = 0
            epoch_known_total = 0
            
            for batch_index, (inputs, labels) in enumerate(dataloaders[data_dirs[phase]]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == "train"):
                    discriminator_net.zero_grad()
                    outputs = discriminator_net(inputs).squeeze()
                    loss_d_real = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    unknown_correct = torch.sum((preds == unknown_id) & (labels.data == unknown_id)).item()
                    unknown_total = torch.sum(labels.data == unknown_id).item()
                    known_correct = torch.sum(preds == labels.data).item() - unknown_correct
                    known_total = inputs.size(0) - unknown_total
                    
                    batch_size = inputs.size(0)
                    noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
                    generated_inputs = generator_net(noise)
                    generated_labels = torch.full((batch_size,), unknown_id, dtype=torch.int64, device=device)
                    
                    outputs_fake = discriminator_net(generated_inputs.detach()).squeeze()
                    loss_d_fake = criterion(outputs_fake, generated_labels)
                    _, preds_fake = torch.max(outputs_fake, 1)
                    
                    if phase == "train":
                        loss_d_real.backward()
                        loss_d_fake.backward()
                        discriminator_optimizer.step()
                        
                        generator_net.zero_grad()
                        outputs_fake = discriminator_net(generated_inputs).squeeze()
                        loss_g = criterion(outputs_fake, generated_labels) * (-1) + 1
                        loss_g.backward()
                        generator_optimizer.step()
                        
                        train_losses.append(loss_d_real.item())
                        train_losses_g.append(loss_g.item())
                        if batch_index % (len(dataloaders[data_dirs[phase]]) // 10) == 0:
                            utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}][{batch_index:04d}/{len(dataloaders[data_dirs[phase]])}] "
                                                f"Loss:{loss_d_real.item():.4f} Acc:{torch.sum(preds == labels.data).item()/inputs.size(0):.4f} "
                                                f"Corrects(Known):{known_correct:02d}/{known_total:02d} "
                                                f"Corrects(Unknown):{unknown_correct:02d}/{unknown_total:02d}")
                    
                    epoch_loss += loss_d_real.item()
                    epoch_acc += torch.sum(preds == labels.data).item()/inputs.size(0)
                    epoch_iter += 1
                    
                    epoch_unknown_correct += unknown_correct
                    epoch_unknown_total += unknown_total
                    epoch_known_correct += known_correct
                    epoch_known_total += known_total
            
            epoch_loss = epoch_loss / epoch_iter
            epoch_acc = epoch_acc / epoch_iter
            if phase == "train":
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
            
            else:
                utils.print_and_log(model_dir, f"{datetime.now()} [{epoch+1:02d}/{num_epochs}] Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f} "
                                    f"Corrects(Known):{epoch_known_correct:04d}/{epoch_known_total:04d} Corrects(Unknown):{epoch_unknown_correct:04d}/{epoch_unknown_total:04d}")
                val_losses.append(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(discriminator_net.state_dict())
                    best_model_g = copy.deepcopy(generator_net.state_dict())
    
    utils.print_and_log(model_dir, f"{datetime.now()} Best Val Acc:{best_acc:4f}")
    discriminator_net.load_state_dict(best_model)
    generator_net.load_state_dict(best_model_g)
    
    return discriminator_net, generator_net, best_acc, train_losses, train_losses_g, val_losses
