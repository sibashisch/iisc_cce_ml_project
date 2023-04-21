# Covid-19 Identification from Chest X-Ray with a corolary of Google Search correctness estimation
### This is the final project for IISC CCE - AI & ML with Python course
### Submitted by Sibashis Chatterjee (sibashis1992@gmail.com)

## Code repository
* Github: [sibashis/iisc_cce_mi_project](https://github.com/sibashisch/iisc_cce_ml_project)

## Data Source

* Using Google image search to download Covid 19 affected Lungs' X-Ray images
* Some data published by some universities:
    
    * The researchers of Qatar University have compiled the COVID-QU-Ex dataset, which consists of 33,920 chest X-ray (CXR) images including: 11,956 COVID-19, 11,263 Non-COVID infections (Viral or Bacterial Pneumonia), and 10,701 Normal. This data can be found [here](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
    * University of Montreal released images can be found [here](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/)    

* Combining all these data to create a huge labeled set of Chest X-Ray dataset. I'll select training, cross validation and test data dynamcally.

## Install packages to make workspace ready

```
pip3 install google-api-python-client
pip3 install numpy
pip3 install pandas

## I was not able to install tensorflow using pip3 install tensorflow due to:
## ERROR: Could not find a version that satisfies the requirement tensorflow==1.15.0 (from versions: none)
##ERROR: No matching distribution found for tensorflow==1.15.0
## This also did not work:
## pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl

## Hence running tensorflow on docker using colima
colima start --memory 16 --arch x86_64 
docker pull tensorflow/tensorflow:latest
## Start Jupyter server with my work directory mounted
docker run -v /Users/sichatte/Downloads/iisc_cce_ml_project:/tf/iisc_cce_ml_project -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter

## Jupyter notebook kept becoming unresponsive from the docker, fall back to Google Colab.
```

## Data Collection

* Scrapping Images from Google, refer: [image_scrapping.ipynb](./image_scrapping.ipynb).
* Using Google Custom Search API for the scrapping. API Keys and other secrets are stored in a file ***my_secrets.py*** (Not pushed to remote repository for security reasons).
* The Scrapping worked till 200 images but then started returning **400 Bad Request** errors; probably I reached the free use limit.
* Download and collate data at one place for creating Training data set, refer [classifier.ipynb](./classifier.ipynb).

## Model Training

* Using Tersorflow Sequential model of Convolution Neural Network.
* By trial and error determined that a model with 4 hidden layers is performing best with some parameter adjustments.
* Refer [classifier.ipynb](./classifier.ipynb) for details about data preprocessing and training.

## Model Evaluation

* With the pre labeled downloaded test data reached the accuracy of 86%.

## Alanysis of Google Image search

* I downloaded images from Google by searching with serch phrase ***covid 19 infected lungs x ray***
* Downloaded images have Chest X-ray data, all of them expected to be classified with Covid-19.

