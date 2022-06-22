# Segmentation models for OPHAI project
This repository contains the code and resources for the benchmark paper on fundus segmentation led by Dr. Eslami and Dr. Kazeminasab of OPHAI. Instructions for reproducing results and using this repository are below. All files for loading, training, and evaluating are located in the `/main` directory, and the code for all models we used are in the `/models` directory.

*Note: Currently, the files that involve ML models are only compatible with TensorFlow/Keras models. PyTorch files will be added soon.

# Preparing the data
To preprocess and then save the data as NumPy files (assuming you have all of the OPHAI data properly saved on your device), use the `main_preparation.py` file.

```
usage: main_preparation.py [-h] --name_csv_train NAME_CSV_TRAIN --name_csv_test NAME_CSV_TEST --data_dir DATA_DIR
                           --path_save_npy PATH_SAVE_NPY --img_size IMG_SIZE

Load data for fundus segmentation (SegLoc, OPHAI).

optional arguments:
  -h, --help            show this help message and exit
  --name_csv_train NAME_CSV_TRAIN
                        Name of the CSV file with training dataset information.
  --name_csv_test NAME_CSV_TEST
                        Name of the CSV file with testing dataset information.
  --data_dir DATA_DIR   Path to the folder with the CSV files and image subfolders.
  --path_save_npy PATH_SAVE_NPY
                        Path to the folder where NumPy files of the dataset will be saved.
  --img_size IMG_SIZE   Size to which the images should be reshaped (ex. 256 or 512).
```

# Training a model
To use one of the models in the `/models` directory, pass in the filename of the file with the code for that model along with the other required arguments to `main_train.py`. Optional arguments include the number of epochs to be trained for and the batch size used during training, which are by default 20 and 4, respectively.

```
usage: main_train.py [-h] [--model_name {unet,attnet,deeplabv3plus,doubleunet,mobilenet_unet,resnet_unet,resunet}] --path_npy
                     PATH_NPY --path_save_model PATH_SAVE_MODEL --img_size IMG_SIZE [--epochs EPOCHS]
                     [--batch_size BATCH_SIZE]

Train model (Keras).

optional arguments:
  -h, --help            show this help message and exit
  --model_name {unet,attnet,deeplabv3plus,doubleunet,mobilenet_unet,resnet_unet,resunet}
                        Name of model to train.
  --path_npy PATH_NPY   Path to the folder with the NPY files for the X_train and y_train data.
  --path_save_model PATH_SAVE_MODEL
                        Path to the folder where model will be saved.
  --img_size IMG_SIZE   Size to which the images should be reshaped (one number, i.e. 256 or 512).
  --epochs EPOCHS       Number of epochs for the model to train.
  --batch_size BATCH_SIZE
                        Batch size for the model during training.
```

# Evaluating a model
To save the predictions of a saved model on a testing dataset and calculate the metrics (code in `metrics.py`), use `main_test.py`.

```
usage: main_test.py [-h] --path_npy PATH_NPY --path_trained_model PATH_TRAINED_MODEL --path_save_results PATH_SAVE_RESULTS

Test model (Keras).

optional arguments:
  -h, --help            show this help message and exit
  --path_npy PATH_NPY   Path to the folder with the NPY files for the X_test and y_test data.
  --path_trained_model PATH_TRAINED_MODEL
                        Path to the saved model.
  --path_save_results PATH_SAVE_RESULTS
                        Path to the folder where results and predictions will be saved.
```

To save the predicted mask of a saved model for one test image, use `main_inference.py`.

```
usage: main_inference.py [-h] --path_img PATH_IMG --path_trained_model PATH_TRAINED_MODEL --path_save PATH_SAVE --img_size
                         IMG_SIZE

Make inference using UNet model.

optional arguments:
  -h, --help            show this help message and exit
  --path_img PATH_IMG   Path to the image.
  --path_trained_model PATH_TRAINED_MODEL
                        Path to the saved model.
  --path_save PATH_SAVE
                        Path to the folder where prediction will be saved.
  --img_size IMG_SIZE   Size to which the image should be reshaped (one number, i.e. 256 or 512).
```


# TO-DO
- Remove `main_preparation.py` + remove .npy file loading from `main_train.py`; add data generator code to `main_train.py`
- Add AG-Net model once finished