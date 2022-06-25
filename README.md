# Lung Disease Detection from X-ray Images

## Overview

Implementation of Deep Learning models that can identify COVID-19,
Lung Opacity, Viral Pneumonia or nothing(if patient is healthy)
from lung X-Ray images.

## Manual

### Installation instructions
In order to run the experiments in your local machine you should do the following steps.

1. Clone the repo by running `git clone https://github.com/ManosL/XRay-Lung-Disease-Detection.git`
2. Afterwards, install virtualenv in pip3(if you did not do that already) by running
`pip3 install virtualenv`
3. Then move to this repository directory.
4. Then create and activate the virtual environment by running the following commands
```
virtualenv <venv_name>
source bin/activate
```
5. Afterwards, install the requirements by running `pip3 install -r requirements.txt`
6. Finally, download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) in order to have the dataset in order to run experiments.
7. You are ready to move to `code/` directory and run the experiments and demo programs!

### Experiments instructions

In order to run the experiments done in order to write the report, go into `code/` directory and run the following
command(you can add `-h` to see how you should run it):

```
        python3 main.py --dataset_path <dataset_path> --epochs <epochs>
                        --batch_size <batch_size> --learning_rate <learning_rate>
                        --patience <patience> --pretrained_path <prerained_model>
                        --mask_images --log <log_file>
```
where:

1. <dataset_path>: Path to COVID-19 Radiography Dataset.
2. <epochs>: Number of training epochs.
3. <batch_size>: Training batch size.
4. <learning_rate>: Learning rate of the optimizer used during training.
5. <patience>: Patience of the model, i.e how many epochs will the model wait 
during training to get better Validation Performance from the best epoch.
6. <pretrained_model>: Path to state dict of a model that was trained before in order to use
those weights at start from training.
7. --mask_images: If  set we apply masking to images.
8. <log_file>: Path to redirect `stdout`

Then you just follow the the output tp just specify the model that you want
experiments with.

WARNING: This will take time in order to complete.

While running this program you will see logs in terminal and the graphs,
will be opened in browser.

