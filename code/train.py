from datetime import datetime
import os, sys
from copy import deepcopy
from collections import Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
import torch.optim as optim
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torch.utils.data import DataLoader



class XRayModel:
    def __init__(self, model, epochs=10, batch_size=256, learning_rate=0.0005, patience=10):
        assert(isinstance(model, nn.Module))

        self.__device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.__model         = model
        self.__epochs        = epochs
        self.__batch_size    = batch_size
        self.__learning_rate = learning_rate
        self.__patience      = patience       # Used in order for our model to run for unnecessary epochs
        
        self.__model.to(self.__device)

        return



    def __make_weights_for_each_sample(self, train_dataset):
        all_labels = train_dataset.dataset.labels if isinstance(train_dataset, Subset) else train_dataset.labels

        sample_labels = deepcopy(all_labels)

        if isinstance(train_dataset, Subset):
            subset_indices = train_dataset.indices

            sample_labels = [all_labels[index] for index in subset_indices]

        label_counters = Counter(sample_labels)
        sample_weights = []

        for label in sample_labels:
            sample_weights.append(1 / label_counters[label])

        return sample_weights



    def fit(self, train_dataset, val_dataset):
        assert(isinstance(train_dataset, Dataset) or isinstance(train_dataset, Subset))
        assert(isinstance(val_dataset, Dataset) or isinstance(val_dataset, Subset))

        # Initialize DataLoaders
        sampler_weights = self.__make_weights_for_each_sample(train_dataset)
        sampler         = WeightedRandomSampler(sampler_weights, len(train_dataset))

        train_dataloader = DataLoader(train_dataset, self.__batch_size, sampler=sampler)
        val_dataloader   = DataLoader(val_dataset, self.__batch_size)

        # Set loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__model.parameters(), lr=self.__learning_rate)

        # Start Training
        train_losses = []
        val_losses   = []

        min_val_loss     = None
        best_model       = deepcopy(self.__model)
        unchanged_epochs = 0

        start_datetime   = "{:%Y%m%d_%H%M%S}".format(datetime.now())

        if not os.path.isdir('./models'):
            os.mkdir('./models')

        path_to_save_best_model = os.path.join('./models', start_datetime + '.weights')

        for epoch in range(self.__epochs):
            total_train_loss  = 0.0
            num_train_batches = 0

            start = time.time()

            for data in train_dataloader:
                images, labels = data[0].to(self.__device), data[1].to(self.__device)

                # zero the gradient params of neural net
                optimizer.zero_grad()

                # get the output vector
                outputs = self.__model(images)

                loss = criterion(outputs, labels)
                
                # Calculating the gradients and updating the weights
                loss.backward()
                optimizer.step()

                num_train_batches += 1

                total_train_loss += loss.item()

                # Free GPU Memory
                del images
                del labels
                torch.cuda.empty_cache()
            
            total_train_loss = total_train_loss / num_train_batches
            train_elapsed    = time.time() - start

            # Calculation of normalized validation loss
            self.__model.eval()

            num_val_batches = 0
            total_val_loss  = 0.0

            for data in val_dataloader:
                inputs, labels = data[0].to(self.__device), data[1].to(self.__device)

                # get the output vector
                outputs = self.__model(inputs)

                optimizer.zero_grad()
                
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                num_val_batches += 1

                # Free GPU Memory
                del inputs
                del labels
                torch.cuda.empty_cache()
            
            total_val_loss = total_val_loss / num_val_batches

            self.__model.train()

            train_losses.append(total_train_loss)
            val_losses.append(total_val_loss)

            print('Epoch', epoch, 'Train Time Elapsed:', round(train_elapsed,2), \
                's Train Loss(Normalized):', round(total_train_loss, 3), 'Validation Loss(Normalized):', \
                round(total_val_loss, 3))
            
            sys.stdout.flush()

            if (min_val_loss == None) or (total_val_loss < min_val_loss):
                unchanged_epochs = 0
                min_val_loss     = total_val_loss

                del best_model
                best_model = deepcopy(self.__model)

                torch.save(best_model.state_dict(), path_to_save_best_model)
            else:
                unchanged_epochs += 1

                if unchanged_epochs == self.__patience:
                    break
        
        del self.__model
        self.__model = best_model

        return train_losses, val_losses



    def predict(self, test_images):
        self.__model.eval()

        preds = torch.argmax(self.__model(test_images), dim=1)

        self.__model.train()

        return preds
