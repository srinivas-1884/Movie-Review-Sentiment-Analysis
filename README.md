# Movie Review Sentiment Analysis Project 
This project focuses on sentiment analysis of movie reviews using the IMDb dataset. The dataset consists of 50,000 movie reviews labeled as positive or negative. The main goal of this project is to develop models that can accurately classify the sentiment of movie reviews.


Dataset

Label	Number of Samples

Positive	25000

Negative	25000

Dataset Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile


Three different models are developed for sentiment analysis of movie reviews:

Simple Neural Network: This model architecture consists of a simple feed-forward neural network with fully connected layers. It is trained on the preprocessed movie review data to learn sentiment classification.

Convolutional Neural Network (CNN): The CNN model incorporates convolutional1D layers, which are effective in capturing local patterns and features in text data. It is trained to perform sentiment analysis on the movie reviews.

Long Short-Term Memory (LSTM): The LSTM model is a type of recurrent neural network (RNN) that is particularly effective in capturing long-term dependencies in sequential data. It is trained on the movie reviews to learn sentiment classification.

Model Training and Evaluation
Each of the models is trained for 10 epochs using the preprocessed movie review dataset. The models are optimized to learn the sentiment expressed in the reviews, and their performances are evaluated based on accuracy.

Based on the experimental results, it is observed that the LSTM model performs better than the other models for sentiment analysis of movie reviews.

Feel free to explore this project's code and experiment with different models and configurations to enhance sentiment analysis performance on the IMDb movie review dataset.

Requirements:-

To run this project, the following dependencies are required:

Python 3.x

Numpy

Pandas

NLTK

Keras

TensorFlow

Scikit-Learn

GloVe word embeddings

Please make sure to install the necessary libraries and download the GloVe word embeddings before running the project.

License:-

This project is licensed under the MIT License.




