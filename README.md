# Reconocimiento de patrones: Practica 2


## Running
There are multiple entry points for the program depending on the intended usage:

### Only train the model
Train the model with all the available training images. \

Run `train_model.py`. The model to be trained can be changed inside the file (variable `model`). 

### Train and test the model
Creates two groups out of the training images. One group (80% of images) will be used to train the model.
The rest (20%) will be used to test it.

Run `train_and_test.py`. The model to be trained can be changed inside the file (variable `model`).

### Test model
Test the model against the competition data. Outputs a file named `Competicion2.txt`.

Run `main_test.py`. The path for the model to be tested can be changed inside the file (variable `model_path`)
with the path to a trained model.


## Folder structure
- `images/`: Contains all the training and test images.
- `src/`: Contains functions, classes and models.
  - `neuralnetworks/`: Contains the Python classes for the deep learning models.
- `outputs/`: Trained models with the hyperparameters used and the loss graphs.
