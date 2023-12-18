# Capstone1-Zoomcamp

## Description of the problem

The initial experiment is done with 15 types of common vegetables that are found throughout the world. The vegetables that are chosen for the experimentation are- bean, bitter gourd, bottle gourd, brinjal, broccoli, cabbage, capsicum, carrot, cauliflower, cucumber, papaya, potato, pumpkin, radish and tomato. A total of 21000 images from 15 classes are used where each class contains 1400 images of size 224×224 and in *.jpg format. The dataset split 70% for training, 15% for validation, and 15% for testing purpose.

Model can be used to solve next issues: 
 - Help robots to pick up vegetables/berries/fruits in hard acceptable places (jungles, woods, etc.) or on sorted stations
 - Help people to understand what is the fruit/vegetable by making a photo of it and upload it to system

EfficientNetB0 pre-trained model and separated dataset were used for training. 
Overall accuracy is 99,96%. 

Used dataset - https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset.
Notebook with all used models - https://www.kaggle.com/code/idem7x/zoomcamp-capstone1/notebook and also [notebook.py](notebook.py) and [notebook.ipynb](notebook.ipynb).

## Instructions on how to run the project

- `git clone git@github.com:idem7x/Capstone1-Zoomcamp.git`
- `cd Capstone1-Zoomcamp`
- Download https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/download, rename it to vegetables.zip and put it into **data** folder
- `docker build -t vegetable_predict .`
- `docker run -it --rm -p 8080:8080 vegetable_predict`
- `python test.py` - you can put any image URL into that file for check (['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'])
