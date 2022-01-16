import os
import numpy as np
from PIL import Image
from utils import write_hdf5

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./hdf5/"

def get_datasets(mode):
    assert(mode in ['training', 'test'])

    # path
    original_dir = f'./datasets/{mode}/images/'
    groundTruth_dir = f'./datasets/{mode}/1st_manual/'
    borderMasks_dir = f'./datasets/{mode}/mask/'

    originalImgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    borderMasks = np.empty((Nimgs,height,width))

    for root, _, files in os.walk(original_dir): #list all files, directories in the path
        for index in range(len(files)):
            # x
            img = Image.open(os.path.join(root, files[index]))
            originalImgs[index] = np.asarray(img)
            # y
            img_index = files[index].split('_')[0]
            g_truth = Image.open(os.path.join(groundTruth_dir,f'{img_index}_manual1.gif'))
            groundTruth[index] = np.asarray(g_truth)
            # mask
            b_mask = Image.open(os.path.join(borderMasks_dir, f'{img_index}_{mode}_mask.gif'))
            borderMasks[index] = np.asarray(b_mask)
    
    # reshaping for my standard tensors
    originalImgs = np.transpose(originalImgs,(0,3,1,2))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    borderMasks = np.reshape(borderMasks,(Nimgs,1,height,width))

    # assert ground truth and border masks are correctly withih pixel value range 0-255 (black-white)
    assert(np.max(groundTruth)==255 and np.max(borderMasks)==255)
    assert(np.min(groundTruth)==0 and np.min(borderMasks)==0)

    assert(originalImgs.shape == (Nimgs,channels,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(borderMasks.shape == (Nimgs,1,height,width))

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # save datasets
    write_hdf5(originalImgs, dataset_path + f'DRIVE_dataset_originalImgs_{mode}.hdf5')
    write_hdf5(groundTruth, dataset_path + f'DRIVE_dataset_groundTruth_{mode}.hdf5')
    write_hdf5(borderMasks,dataset_path + f'DRIVE_dataset_borderMasks_{mode}.hdf5')

get_datasets(mode='training')
get_datasets(mode='test')