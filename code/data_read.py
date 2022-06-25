import os
import cv2
from math import floor 
import numpy as np

def read_greyscale_image(img_path):
    assert(os.path.exists(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return image / 255


# It is needed to be done because masks have 256x256 pixels while 
# images have 299x299 pixels. Also mask needs normalization in order
# to be applied correctly.
def process_mask(mask_image):
    pixels_to_add = floor((299 - 256) / 2)

    new_mask = cv2.copyMakeBorder(mask_image, pixels_to_add, pixels_to_add+1, 
                                pixels_to_add, pixels_to_add+1, cv2.BORDER_CONSTANT,
                                value=[0,0,0])

    return new_mask


# root_dir is the path to the directory of COVID-19
# Radiography Dataset.
#
# Add augmentation functions or add it in a separate function
def read_dataset(root_dir):
    unmasked_images = []
    masked_images   = []
    labels          = []

    assert(os.path.isdir(root_dir))

    data_dirs = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    for data_dir in data_dirs:
        images_dir = os.path.join(root_dir, data_dir, 'images')
        masks_dir  = os.path.join(root_dir, data_dir, 'masks')

        assert(os.path.isdir(images_dir))
        assert(os.path.isdir(masks_dir))

        image_files = os.listdir(images_dir)

        for img_file in image_files:
            img_path  = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)

            assert(os.path.exists(mask_path))

            unmasked_image = read_greyscale_image(img_path)

            mask = read_greyscale_image(mask_path)
            mask = process_mask(mask)

            masked_image = np.multiply(unmasked_image, mask)
            
            unmasked_images.append(cv2.resize(unmasked_image, (128, 128)))
            masked_images.append(cv2.resize(masked_image, (128, 128)))
            labels.append(data_dir)
        
        print('finished', data_dir)

    return unmasked_images, masked_images, labels



import matplotlib.pyplot as plt

instances, masks, labels = read_dataset('../data/COVID-19_Radiography_Dataset')
print(len(instances), len(masks), len(labels))

for i in range(15):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(masks[i])
    axarr[1].imshow(instances[i])

    plt.show()
