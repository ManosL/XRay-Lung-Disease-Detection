import argparse
from collections import Counter
from math import floor, ceil
import os, sys

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Lambda, Compose, ToTensor, Resize, AutoAugmentPolicy
from torchvision.transforms import RandomAutocontrast, RandomRotation

from torchinfo import summary

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from cnn_large import CNNmodelLarge
from resnet import ResNetType, ResNetXRayModel

from train import XRayModel
from xRaysDataset import xRaysDataset

from graph_utils import plot_train_val, plot_confusion_matrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_PATH           = '/home/manosl/Desktop/MSc Courses Projects/2nd Semester/Deep Learning/Project 1/data/COVID-19_Radiography_Dataset'
VAL_DATASET_PERCENTAGE = 0.1


# Make the prediction online in batches and not just get all of it
def dataset_extract_predictions_and_labels(model, test_dataset):
    preds  = []
    labels = []

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    for data in test_dataloader:
        curr_images, curr_labels = data[0].to(device), data[1].to(device)

        curr_preds = model.predict(curr_images)

        preds  += curr_preds.tolist()
        labels += curr_labels.tolist()

    return preds, labels



def check_positive_int(value):
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    
    return ivalue



def check_positive_float(value):
    fvalue = float(value)

    if fvalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    
    return fvalue



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid directory" % dir_path_str)

    return dir_path_str



def check_is_file(file_path):
    file_path_str = str(file_path)

    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid file" % file_path_str)

    return file_path


  
def input_resnet_params():
    pretrained       = None
    layers_to_freeze = []

    pretrained = input('\nDo you want to use pretrained weights for your ResNet?(strictly y/n): ')

    while pretrained not in ['y', 'n']:
        print('\n\nPlase type \'y\' or \'n\'')
        pretrained = input('\nDo you want to use pretrained weights for your ResNet?(strictly y/n): ')
    
    if pretrained == 'y':
        pretrained = True
    else:
        pretrained = False

    for i in range(4):
        to_freeze = input('\nDo you want to freeze ResNet\'s layer' + str(i+1) + '?(strictly y/n): ')

        while to_freeze not in ['y', 'n']:
            print('\n\nPlase type \'y\' or \'n\'')
            to_freeze = input('\nDo you want to freeze ResNet\'s layer' + str(i+1) + '?(strictly y/n): ')

        if to_freeze == 'y':
            layers_to_freeze.append(i)

    return pretrained, layers_to_freeze



def input_model_to_train():
    model_names = ['Handcrafted CNN(5 CNN Layers and 2 Linear Layer)',
                'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
                'ResNet152']

    model_choice = -1

    print('Choose a model to train on X-Rays Dataset:')
    
    for i, model_name in zip(range(len(model_names)), model_names):
        print('\t' + str(i + 1) + '. ' + model_name)

    model_choice = int(input('\nGive the number of the model you want to train: '))

    while model_choice < 1 or model_choice > len(model_names):
        print('\n\nYou gave invalid model number')

        print('\nChoose a model to train on X-Rays Dataset:')
    
        for i, model_name in zip(range(len(model_names)), model_names):
            print('\t' + str(i + 1) + '. ' + model_name)

        model_choice = int(input('\nGive the number of the model you want to train: '))
    
    # Extracting the model
    model = None

    if model_choice == 1:
        model = CNNmodelLarge()
    else:
        possible_resnets = [ResNetType.RESNET18, ResNetType.RESNET34, ResNetType.RESNET50,
                            ResNetType.RESNET101, ResNetType.RESNET152]
        
        resnet_type      = possible_resnets[model_choice - 2]

        pretrained, layers_to_freeze = input_resnet_params()

        model = ResNetXRayModel(resnet_type, pretrained, layers_to_freeze)

    return model


    
def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('description=COVID Radiography Deep Learning Model')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=DATASET_PATH,
                        help='path to the COVID Radiography dataset')
    parser.add_argument('--epochs', type=check_positive_int, default=15,
                        help='number of epochs that our model will be trained')
    parser.add_argument('--batch_size', type=check_positive_int, default=256,
                        help='model\'s batch size')
    parser.add_argument('--learning_rate', type=check_positive_float, default=0.0005,
                        help='Learning Rate of Adam optimizer')
    parser.add_argument('--patience', type=check_positive_int, default=20,
                        help='model\'s patience, i.e. the number of epochs that we will \
                              wait for our model to improve from the best one')
    parser.add_argument('--pretrained_path', type=check_is_file, default=None,
                        help='path to pretrained weights of a model in order to use \
                            as a starting point for its training.')
    parser.add_argument('--mask_images', action='store_true', default=False,
                        help='Whether to apply the mask to images or not')
    parser.add_argument('--log', type=str, default=None,
                        help='Specify a log file to redirect stdout')

    args = parser.parse_args()

    model = input_model_to_train()

    # Load pretrained weights(if any given)
    if args.pretrained_path != None:
        model.load_state_dict(torch.load(args.pretrained_path, map_location="cuda:0"))

    if args.log != None:
        sys.stdout = open(args.log, 'w+')

    if isinstance(model, ResNetXRayModel):
        in_channels = 3
    else:
        in_channels = 1

    print('The model that we will train our classifier has the following structure:')
    summary(model, (args.batch_size, in_channels, 128, 128))

    grey_to_rgb_transform   = Lambda(lambda y: torch.stack([torch.tensor(y), torch.tensor(y), torch.tensor(y)]))

    augmentation_transforms = []

    train_transforms        = Compose([grey_to_rgb_transform, Resize((128, 128))] + augmentation_transforms)

    test_transforms         = Compose([grey_to_rgb_transform, Resize((128, 128))])

    if in_channels == 1:
        train_transforms        = Compose([ToTensor(), Resize((128, 128))] + augmentation_transforms)
        test_transforms         = Compose([ToTensor(), Resize((128, 128))])

    # It's better without mask
    train_dataset = xRaysDataset(args.dataset_path, train=True,  to_mask=args.mask_images, transform=train_transforms, target_transform=torch.tensor)
    test_dataset  = xRaysDataset(args.dataset_path, train=False, to_mask=args.mask_images, transform=test_transforms, target_transform=torch.tensor)

    val_dataset_size   = int(len(train_dataset) * VAL_DATASET_PERCENTAGE)
    train_dataset_size = int(len(train_dataset) - val_dataset_size)

    assert(val_dataset_size + train_dataset_size == len(train_dataset))

    train_dataset, val_dataset = random_split(train_dataset, [train_dataset_size, val_dataset_size],
                                            generator=torch.Generator().manual_seed(42))

    val_dataset.transform = test_transforms

    trainer = XRayModel(model, epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, patience=args.patience)

    train_losses, val_losses = trainer.fit(train_dataset, val_dataset)

    plot_train_val(args.epochs, train_losses, val_losses, type(model).__name__)

    test_preds, test_labels = dataset_extract_predictions_and_labels(trainer, test_dataset)


    print('\nAccuracy on test set',    accuracy_score(test_labels, test_preds))
    print('\nRecall on test set',      recall_score(test_labels, test_preds, average='micro'))
    print('\nPrecision on test set',   precision_score(test_labels, test_preds, average='micro'))

    plot_confusion_matrix(confusion_matrix(test_labels, test_preds), [0,1,2,3])

    if args.log != None:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    return 0



if __name__ == "__main__":
    main()
