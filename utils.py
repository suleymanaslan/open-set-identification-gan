import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets
import torchvision.transforms as transforms

from tqdm import tqdm


def copy_data():
    NB_OF_IDENTITIES = 50
    UNKNOWN_IDENTITIES = 5 # 10%
    NB_OF_IMAGES = 200
    VAL_IMAGES = 10 # 5%
    
    folder_path = 'vggface2_data/test/'
    copy_train_path = "data/train"
    copy_val_path = "data/val"
    
    if not os.path.exists(copy_train_path):
        os.makedirs(copy_train_path)
    if not os.path.exists(copy_val_path):
        os.makedirs(copy_val_path)
    
    data_path = os.path.join(os.getcwd(), folder_path)
    
    count = 0
    min_nb_images = np.inf
    
    print(f"Identities in training set:{len(os.listdir(data_path))}, "
          f"selecting {NB_OF_IDENTITIES} identities and {NB_OF_IMAGES} images for each identity")
    
    for folder in tqdm(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if len(os.listdir(folder_path)) < NB_OF_IMAGES:
            continue
        class_train_path = f"{copy_train_path}/{count}/" if count < NB_OF_IDENTITIES - UNKNOWN_IDENTITIES else f"{copy_train_path}/{NB_OF_IDENTITIES - UNKNOWN_IDENTITIES}/"
        class_val_path = f"{copy_val_path}/{count}/" if count < NB_OF_IDENTITIES - UNKNOWN_IDENTITIES else f"{copy_val_path}/{NB_OF_IDENTITIES - UNKNOWN_IDENTITIES}/"
        if not os.path.exists(class_train_path):
            os.makedirs(class_train_path)
        if not os.path.exists(class_val_path):
            os.makedirs(class_val_path)
        if min_nb_images > len(os.listdir(folder_path)):
            min_nb_images = len(os.listdir(folder_path))
        for filename in os.listdir(folder_path)[:NB_OF_IMAGES-VAL_IMAGES]:
            shutil.copy2(f"{folder_path}/{filename}", f"{class_train_path}/{count}_{filename}")
        for filename in os.listdir(folder_path)[NB_OF_IMAGES-VAL_IMAGES:NB_OF_IMAGES]:
            shutil.copy2(f"{folder_path}/{filename}", f"{class_val_path}/{count}_{filename}")
        count += 1
        if count >= NB_OF_IDENTITIES:
            break
    
    print(f"Min number of images for an identity:{min_nb_images}")


def print_and_log(model_dir, text):
    print(text)
    print(text, file=open(f'{model_dir}/log.txt', 'a'))


def init_training(batch_size, pgan=False):
    training_timestamp = str(int(time.time()))
    model_dir = f'trained_models/model_{training_timestamp}/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if pgan:
        shutil.copy2('./vggface2_pgan.ipynb', model_dir)
        shutil.copy2('./pgan_networks.py', model_dir)
        shutil.copy2('./pgan_model.py', model_dir)
    else:
        shutil.copy2('./vggface2_gan.ipynb', model_dir)
        shutil.copy2('./networks.py', model_dir)
        shutil.copy2('./model.py', model_dir)
    
    dataset = torchvision.datasets.ImageFolder(root="data/aligned_train", transform=transforms.Compose([transforms.Resize(64),
                                                                                                        transforms.RandomHorizontalFlip(),
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return model_dir, dataloader


def save_model(model_dir, discriminator_net, generator_net, generator_losses, discriminator_losses, generated_images):
    torch.save(discriminator_net.state_dict(), f"{model_dir}/net_discriminator.pth")
    torch.save(generator_net.state_dict(), f"{model_dir}/net_generator.pth")
    np.save(f"{model_dir}/losses_generator.npy", np.array(generator_losses))
    np.save(f"{model_dir}/losses_discriminator.npy", np.array(discriminator_losses))
    np.save(f"{model_dir}/generated_images.npy", generated_images)
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses,label="Generator")
    plt.plot(discriminator_losses,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_dir}/loss.png")
    plt.show()


def get_dataloader(image_size=64, batch_size=8):
    dataset = torchvision.datasets.ImageFolder(root="data/aligned_train", transform=transforms.Compose([transforms.Resize(image_size),
                                                                                                        transforms.RandomHorizontalFlip(),
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader


def save_pgan_model(model_dir, discriminator_net, generator_net, fixed_latent, generated_images, n_scales=5):
    img_sizes = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}
    
    torch.save(discriminator_net.state_dict(), f"{model_dir}/net_discriminator.pth")
    torch.save(generator_net.state_dict(), f"{model_dir}/net_generator.pth")
    torch.save(fixed_latent, f"{model_dir}/fixed_latent.pt")
    for i in range(n_scales):
        np.save(f"{model_dir}/generated_images_{img_sizes[i]}x{img_sizes[i]}.npy", generated_images[i])


def plot_generated_images(model_dir, generated_images, n_scales=5):
    img_sizes = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}
    
    for i in range(n_scales):
        eval_img = generated_images[i][-1][-1]
        plt.figure(figsize=(9*2, 4*2))
        plt.imshow((eval_img+1.0)*0.5)
