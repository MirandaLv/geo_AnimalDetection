# geo_AnimalDetection

This post is used to record the process of the animal detection project.

Step1. Data processing and creating the training and validation dataset
- Create a folder under dataset and name it "processing_data"
- In the above created folder, create two folders and name them "clipped" and "annotation" respectively.
- Run the get_data.py in the script to generate the training dataset and testing dataset, and annotations for each image.
- The results will be saved in the folder created above.

Step2. The train_fcnn.py is used to train the model based on the customed dataset

