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

## Data Clean up

* Google returned a lot of images with wrong encoding and format information. This was causing model predictions to fail.
* Used Image package from [PIL](https://python-pillow.org/) library to read the images and clean them up.

    ```
    UnidentifiedImageError: cannot identify image file '/content/google-test-images/positive/Covid/image_54.jpeg'
    UnidentifiedImageError: cannot identify image file '/content/google-test-images/positive/Covid/image_59.jpeg'
    UnidentifiedImageError: cannot identify image file '/content/google-test-images/positive/Covid/image_21.jpeg'
    ```

## Model Training

* Using Tersorflow Sequential model of Convolution Neural Network.
* By trial and error determined that a model with 4 hidden layers is performing best with some parameter adjustments.
* Refer [classifier.ipynb](./classifier.ipynb) for details about data preprocessing and training of the final model.
* I had tried with different number of layers and hyper parameters, one such example can be found at [classifier1.ipynb](./classifier1.ipynb) and [classifier_3_hidden_layers.ipynb](./classifier_3_hidden_layers.ipynb) files.

    ```
    Epoch 1/10
    125/125 [==============================] - 432s 3s/step - loss: 0.8519 - accuracy: 0.6316 - val_loss: 0.8355 - val_accuracy: 0.7273
    Epoch 2/10
    125/125 [==============================] - 418s 3s/step - loss: 0.6004 - accuracy: 0.7668 - val_loss: 0.7592 - val_accuracy: 0.7121
    Epoch 3/10
    125/125 [==============================] - 423s 3s/step - loss: 0.5133 - accuracy: 0.7979 - val_loss: 0.8091 - val_accuracy: 0.7727
    Epoch 4/10
    125/125 [==============================] - 424s 3s/step - loss: 0.4358 - accuracy: 0.8256 - val_loss: 0.3413 - val_accuracy: 0.8182
    Epoch 5/10
    125/125 [==============================] - 413s 3s/step - loss: 0.3869 - accuracy: 0.8562 - val_loss: 0.3721 - val_accuracy: 0.8636
    Epoch 6/10
    125/125 [==============================] - 418s 3s/step - loss: 0.3511 - accuracy: 0.8620 - val_loss: 0.3438 - val_accuracy: 0.8636
    Epoch 7/10
    125/125 [==============================] - 420s 3s/step - loss: 0.2826 - accuracy: 0.8862 - val_loss: 0.3524 - val_accuracy: 0.8636
    Epoch 8/10
    125/125 [==============================] - 413s 3s/step - loss: 0.2514 - accuracy: 0.9010 - val_loss: 0.4744 - val_accuracy: 0.8485
    Epoch 9/10
    125/125 [==============================] - 410s 3s/step - loss: 0.2080 - accuracy: 0.9115 - val_loss: 0.2912 - val_accuracy: 0.8636
    Epoch 10/10
    125/125 [==============================] - 412s 3s/step - loss: 0.1721 - accuracy: 0.9296 - val_loss: 0.4677 - val_accuracy: 0.8636
    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_7 (Conv2D)           (None, 254, 254, 32)      896       

     max_pooling2d_7 (MaxPooling  (None, 127, 127, 32)     0         
     2D)                                                             

     conv2d_8 (Conv2D)           (None, 125, 125, 64)      18496     

     max_pooling2d_8 (MaxPooling  (None, 62, 62, 64)       0         
     2D)                                                             

     conv2d_9 (Conv2D)           (None, 60, 60, 128)       73856     

     max_pooling2d_9 (MaxPooling  (None, 30, 30, 128)      0         
     2D)                                                             

     conv2d_10 (Conv2D)          (None, 28, 28, 128)       147584    

     max_pooling2d_10 (MaxPoolin  (None, 14, 14, 128)      0         
     g2D)                                                            

     flatten_2 (Flatten)         (None, 25088)             0         

     dense_4 (Dense)             (None, 512)               12845568  

     dropout_2 (Dropout)         (None, 512)               0         

     dense_5 (Dense)             (None, 3)                 1539      

    =================================================================
    Total params: 13,087,939
    Trainable params: 13,087,939
    Non-trainable params: 0
    _________________________________________________________________

    ```

## Model Evaluation

* With the pre labeled downloaded test data reached the accuracy of 86%.

    ```
    3/3 [==============================] - 3s 460ms/step - loss: 0.4677 - accuracy: 0.8636
    [0.4677329659461975, 0.8636363744735718]
    ```

## Alanysis of Google Image search

* I downloaded images from Google by searching with serch phrase ***covid 19 infected lungs x ray***
* Downloaded images have Chest X-ray data, all of them expected to be classified with Covid-19.
* Analysis of Google Search results can be found at [google_search_analysis.ipynb](./google_search_analysis.ipynb) file.
* I segragated the data scrapped from Google in to two categories.
    
    * In first category, 100 images were labeled as Covid, let's call them positive cases
    * In second category 50 images were wrongly s Normal an Viral Pneumonia each let's call them negative case
    
    ```
    google_images_positive_data_processed.classes
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)
       
    predicted = model.predict(google_images_positive_data_processed)
    
    import numpy as np
    positive_test_case_results = np.argmax(predicted, axis = 1)
    
    model.evaluate(google_images_positive_data_processed)
    4/4 [==============================] - 9s 2s/step - loss: 0.9084 - accuracy: 0.8041
    [0.9083557724952698, 0.8041236996650696]
    
    positive_test_case_results
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1,
       2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2,
       0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2,
       0, 2, 1, 0, 0, 2, 0, 0, 0])
    ```

* The model tried to classify the images and yield ~80% accuracy in both cases. The Summary results are as follows
    ```
    Positive case predictions:  (97,)
    Positive Test cases:
      Total Predictions =  97
      Correct Predictions =  78  that is  80.41237113402062 % of all predictions
      Falsely Predicted as Normal =  5  that is  5.154639175257731 % of all predictions
      Falsely Predicted as Pneumonia  =  14  that is  14.432989690721648 % of all predictions
      Invalid Predictions =  0  that is  0.0 % of all predictions
      
    Negative test case predictions:  (80,)
    Negative Test cases:
      Total Predictions =  80
      Correct Predictions =  64  that is  80.0 % of all predictions
      Falsely Predicted as Normal =  7  that is  8.75 % of all predictions
      Falsely Predicted as Pneumonia  =  9  that is  11.25 % of all predictions
      Invalid Predictions =  0  that is  0.0 % of all predictions
    ```

