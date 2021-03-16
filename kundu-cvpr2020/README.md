# kundu-cvpr2020 

* [project page](https://sites.google.com/view/pgp-human)

## CODE SETUP
### Running the code
The following instructions guide on running the code for inferring 3D/2D poses and visualizing the cross pose-appearance transfer results.

#### 1.1 Pre-processing images
Before running the code, please make note that the images used to train the model were person centered. The crops are made by ensuring a margin of about 20 pixels between the image boundary and the human part closest to the boundary. One can use a person detector to get a bounding box around a person in an image. The network takes input images of size 224x224 pixels.

#### 1.2 Dependencies and setup
The project demo code can be found in the folder inference_code.

Ensure that you have Python 2.7 and all the required dependencies installed.
```
pip install -r requirements.txt
```

Download the latest weights files and copy them into the folder log_dir.
The config.py file contains other configurable parameters such as batch size, number of gpus etc.


####  1.3 Qualitative results visualization
Run the main program with the source appearance and target pose images as
```
python main.py set1/src.jpg set1/tgt.jpg
```
Once the main file is run, it will run inference on the given pair of images to generate the pose skeleton (17J 3D poses) and the visualization of the cross pose-appearance transfer. The results are saved as pose_pred.jpg and predictions.jpg respectively.



### 2. Evaluation on Standard Benchmarks
The following instructions can be followed to run MPJPE evaluation of the trained model on the H3.6M test set. Please refer to our paper for further details. We provide the pre-processed test set images with the code. Pre-processing steps will be published soon.


#### 2.1 H3.6M test dataset
The processed H3.6M test dataset has been included in the evaluation_code, under the directory H36_cropped.


#### 2.2 Dependencies and setup
Ensure that you have Python 2.7 and all the required dependencies installed.
```
pip install -r requirements.txt
```
Download the latest weights files and copy them into the folder log_dir.
The config.py file contains other configurable parameters such as batch size, number of gpus etc.


#### 2.3 Running evaluation
The trained model can be evaluated by running the below command:
```
python evaluate.py
```
The above program evaluates the MPJPE of the trained model on the H3.6M test set and prints the MPJPE value. In the paper, we report the average of all runs. However, here we have provided the best model reporting 62.25 MPJPE.
