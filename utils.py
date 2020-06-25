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
    NB_OF_IDENTITIES = 460
    UNKNOWN_IDENTITIES = 46 # 10%
    NB_OF_IMAGES = 200
    VAL_IMAGES = 40 # 20%
    
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
        class_train_path = f"{copy_train_path}/{count:04d}/" if count < NB_OF_IDENTITIES - UNKNOWN_IDENTITIES else f"{copy_train_path}/{(NB_OF_IDENTITIES - UNKNOWN_IDENTITIES):04d}/"
        class_val_path = f"{copy_val_path}/{count:04d}/" if count < NB_OF_IDENTITIES - UNKNOWN_IDENTITIES else f"{copy_val_path}/{(NB_OF_IDENTITIES - UNKNOWN_IDENTITIES):04d}/"
        if not os.path.exists(class_train_path):
            os.makedirs(class_train_path)
        if not os.path.exists(class_val_path):
            os.makedirs(class_val_path)
        if min_nb_images > len(os.listdir(folder_path)):
            min_nb_images = len(os.listdir(folder_path))
        for filename in os.listdir(folder_path)[:NB_OF_IMAGES-VAL_IMAGES]:
            shutil.copy2(f"{folder_path}/{filename}", f"{class_train_path}/{count:04d}_{filename}")
        for filename in os.listdir(folder_path)[NB_OF_IMAGES-VAL_IMAGES:NB_OF_IMAGES]:
            shutil.copy2(f"{folder_path}/{filename}", f"{class_val_path}/{count:04d}_{filename}")
        count += 1
        if count >= NB_OF_IDENTITIES:
            break
    
    print(f"Min number of images for an identity:{min_nb_images}")


def print_and_log(model_dir, text):
    print(text)
    print(text, file=open(f'{model_dir}/log.txt', 'a'))


def init_training(batch_size, pgan=False):
    training_timestamp = str(int(time.time()))
    model_label = "pgan" if pgan else "gan"
    model_dir = f'trained_models/model_{model_label}_{training_timestamp}/'

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


def init_identification_training(batch_size):
    training_timestamp = str(int(time.time()))
    model_dir = f'trained_models/model_identification_resnext_{training_timestamp}/'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    shutil.copy2('./identification_resnext.ipynb', model_dir)
    
    data_dirs = {"train": "aligned_train", "val": "aligned_val"}
    
    data_transforms = {data_dirs["train"]: transforms.Compose([transforms.Resize(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       data_dirs["val"]: transforms.Compose([transforms.Resize(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      }
    
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join("data", x), data_transforms[x]) for x in [data_dirs["train"], data_dirs["val"]]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, drop_last=True) for x in [data_dirs["train"], data_dirs["val"]]}
    class_names = image_datasets[data_dirs["train"]].classes
    
    return model_dir, data_dirs, dataloaders, class_names


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


def save_identification_model(model_dir, network, best_acc, train_losses, val_losses):
    torch.save(network.state_dict(), f"{model_dir}/model_{best_acc:.4f}.pth")
    np.save(f"{model_dir}/train_losses.npy" , np.array(train_losses))
    np.save(f"{model_dir}/val_losses.npy" , np.array(val_losses))

    train_epoch_step = len(train_losses)//len(val_losses)

    train_plot_steps = np.arange(len(train_losses))+1
    val_plot_steps = (np.arange(len(val_losses))+1)*train_epoch_step
    plt.figure(figsize=(10,5))
    plt.title("Losses")
    plt.plot(train_plot_steps, train_losses, label='train_loss', linewidth=3)
    plt.plot(val_plot_steps, val_losses, label='val_loss', linewidth=3)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_dir}/loss.png")
    plt.show()


def plot_generated_images(model_dir, generated_images, n_scales=5):
    img_sizes = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}
    
    for i in range(n_scales):
        eval_img = generated_images[i][-1][-1]
        plt.figure(figsize=(9*2, 4*2))
        plt.imshow((eval_img+1.0)*0.5)


def evaluate_identification(model_dir, network, data_dirs, dataloaders, class_names):
    device = torch.device("cuda:0")
    
    network.eval()
    num_images = 3

    with torch.no_grad():
        val_inputs, val_labels = next(iter(dataloaders[data_dirs["val"]]))
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)

        outputs = network(val_inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(num_images):
            plt.figure(figsize=(8,8))
            plt.title(f"Actual:{class_names[val_labels[j]]}, Predicted:{class_names[preds[j]]}")
            plt.imshow((np.transpose(val_inputs[j].cpu().numpy(), (1, 2, 0))*np.array([0.229, 0.224, 0.225]))+np.array([0.485, 0.456, 0.406]))
            plt.savefig(f"{model_dir}/sample_predicted_{j}.png")
