# ColorIdentifiication
Classifies an image of a car to be one of 9 different colors. 

## Description

There are two main approaches.
The first finds the color with the most occupancy by determining all the pixels that lie in the HSV range for a particular color.
A weighted pixel count determines the label of the image. 
This is a fully hard-coded unsupervised method.

The second method uses the H, S and V histograms of an image.
It requires knowledge of the average and standard deviations of the histograms for each color.
The average histogram is calculated from all the images with blue cars.
It is therefore a supervised method that requires training.
During inference, the histograms of an image is compared to the trained average histograms of each color.
The similarity determines the label of the image.

ColorPredictor class has methods for loading, training, infering and evaluating images.

See notebook.ipynb for a demonstration and explanation. 

## Setup Instructions

Run these instructions:

conda create --name myenv python=3.8.18

conda activate myenv

pip install -r requirements.txt

Download the Dataset from:

https://drive.google.com/file/d/19DvNho1_Af3N1qiqz4j7tP-vmTHMof7v/view?usp=sharing

## Usage
python main.py ./dataset_path ./output_path
