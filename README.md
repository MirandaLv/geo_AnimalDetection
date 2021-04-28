# geo_AnimalDetection

This post is used to record the process of the animal detection project.

Step1. Data processing and creating the training and validation dataset
- Create a folder under dataset and name it "processing_data"
- In the above created folder, create two folders and name them "clipped" and "annotation" respectively.
- Run the get_data.py in the script to generate the training dataset and testing dataset, and annotations for each image.
- The results will be saved in the folder created above.

Step2. The train_fcnn.py is used to train the model based on the customed dataset



## Script description
- DataProcessing.py includes a series of support functions to preprocess the raw image and generate the image patches and annotations for training and validation.
- get_data.py is a call script used to generate the image patches and annotations. The dimension of image patches can be assigned in here.
- data_aggregation.py include script to split the raw image into image patches, these image patches will be used later on as input for model prediction.
- jobscript: a script to submit jobs on sciclone


## System configuration
tensorflow-gpu 1.15
CUDA 10.0
keras 2.1.5
python 3.6
geopandas 0.8.1

