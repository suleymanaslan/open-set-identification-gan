import os
import shutil
import numpy as np
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
