# Evaluation of Robustness of Disc/Cup Segmentation in Different Fundus Imaging Conditions
Instructions for reproducing results and using this repository are below. All files for training, evaluating, and plotting are located in the parent directory. The code for all models we used are in `/models`. The code for data loading, metrics, and loss functions can be found in `/utils`.

# Training a model
To use one of the models in the `/models` directory, pass in the filename of the file with the code for that model along with the other required arguments to `main_train.py`.

```
usage: main_train.py [-h] --model-name
                     {attnet,cenet,deeplabv3plus,doubleunet,mnet,mobilenet_unet,resnet_unet,resunet,unet,unetpp}
                     --name-csv-train NAME_CSV_TRAIN --data-dir DATA_DIR
                     --path-save PATH_SAVE --img-size IMG_SIZE
                     [--batch-size BATCH_SIZE] [--binary BINARY]

Train model.

optional arguments:
  -h, --help            show this help message and exit
  --model-name {attnet,cenet,deeplabv3plus,doubleunet,mnet,mobilenet_unet,resnet_unet,resunet,unet,unetpp}
                        Name of model to train.
  --name-csv-train NAME_CSV_TRAIN
                        Name of the CSV file with training dataset
                        information.
  --data-dir DATA_DIR   Path to the folder with the CSV files and image
                        subfolders.
  --path-save PATH_SAVE
                        Path to the folder where model will be saved.
  --img-size IMG_SIZE   Size to which the images should be reshaped (one
                        number, i.e. 256 or 512).
  --batch-size BATCH_SIZE
                        Batch size for the model during training.
  --binary BINARY       Whether the segmentation masks are binary (True) or
                        multi-class (False).
```

# Evaluating a model
To save the predictions of a saved model on a testing dataset and calculate the metrics (code in `metrics.py`), use `main_test.py`.

```
usage: main_test.py [-h]
                    [--model-name {attnet,cenet,deeplabv3plus,doubleunet,mnet,mobilenet_unet,resnet_unet,resunet,unet,unetpp}] 
                    --name-csv-test NAME_CSV_TEST --data-dir DATA_DIR
                    --path-model PATH_MODEL --img-size IMG_SIZE
                    --path-save-results PATH_SAVE_RESULTS
                    [--save-masks SAVE_MASKS] [--binary BINARY]

Test model.

optional arguments:
  -h, --help            show this help message and exit
  --model-name {attnet,cenet,deeplabv3plus,doubleunet,mnet,mobilenet_unet,resnet_unet,resunet,unet,unetpp}
                        Name of model to train.
  --name-csv-test NAME_CSV_TEST
                        Name of the CSV file with testing dataset information.
  --data-dir DATA_DIR   Path to the folder with the CSV files and image
                        subfolders.
  --path-model PATH_MODEL
                        Path to the saved model.
  --img-size IMG_SIZE   Size to which the images should be reshaped (one
                        number, i.e. 256 or 512).
  --path-save-results PATH_SAVE_RESULTS
                        Path to the folder where results and predictions will
                        be saved.
  --save-masks SAVE_MASKS
                        Whether you want to save all predicted masks for test
                        set or not.
  --binary BINARY       Whether the segmentation masks are binary (True) or
                        multi-class (False).
```

# Figures and statistics
To recreate the figures in the abstract, run `main_create_summary_results.py` and then run `main_plots.py`. To obtain the data for the table, run `main_stats.py`.

# Notes
- The abstract only uses the UNet++, DeepLabV3+, and CE-Net model architectures. Others, specifically AttNet and MNet, may not work as expected at the moment.
