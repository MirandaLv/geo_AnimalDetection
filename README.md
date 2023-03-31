# geo_AnimalDetection

## This post is an application of using deep learning method with drone imagery for detection of seabirds. 


Data preprocessing before Deep Learning:
- Need to have a list of images with annotations indicating the bounding box of the class in the image and the associated class name
- Digitize all the animals in the given .tiff file in the raw data folder, and save it as a geojson or shapefile format

Working on Models:

Step1. Data processing and creating the training and validation dataset
- Create a folder under dataset and name it "processing_data"
- In the above created folder, create two folders and name them "clipped" and "annotation" respectively.
- Run the get_data.py in the script to generate the training dataset and testing dataset, and annotations for each image.
- The results will be saved in the folder created above.

Step2. Get pre-trained vgg16 weights, other weights can be found on the same repo, based on the cnn architecture chosen to use.
- https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

Step3. Running on local computer.
The train_fcnn.py is used to train the model based on the customed dataset.Open a terminal, and training the model with: 
- python3 train_frcnn.py -o simple -p directory/to/in/processing_data/train_annotation.txt --num_epochs 1000 --input_weight_path directory/to/the/weights/downloaded/above/vgg16_weights_tf_dim_ordering_tf_kernels.h5 --output_weight_path model_frcnn_vgg.hdf5 --hf True --vf True --rot True --network vgg
- Check on the train_frcnn.py to get detailed calling requirement.

Step 4. Running on sciclone or use GPU to boost the training.
- modify the jobscript file in the script folder, copy above function call in step3 to this file.
- set the gpu call on the top of the file, if gpu is not available, CPUs will be used to run the training.
- submit a jobscript to sciclone by using: qsub jobscript

Step 5. Running on cdsw using GPUs
- Open a terminal, and copy the function call in step 3 to run training on cdsw.

Step 6. Model validation
- Running on testing and validation 
- python3 test_frcnn.py -o simple -p directory/to/in/processing_data/test_annotation.txt --network vgg
- Producing 1). bounding_box_coordinates.csv that hosts the new predicted bounding box of testing data; 2). test_mAPs.csv file that saves the mAP value for each testing image.


## Script description
### Data processing
- DataProcessing.py includes a series of support functions to preprocess the raw image and generate the image patches and annotations for training and validation.
- get_data.py is a call script used to generate the image patches and annotations. The dimension of image patches can be assigned in here.
- jobscript: a script to submit jobs on sciclone
### Model processing
The faster_rcnn folder includes all model related scripts, they are called on training and validation process

## Requirements
- tensorflow-gpu 1.15
- CUDA 10.0
- keras 2.1.5
- python 3.6

