# Sentiment-Analysis-for-tweets
Sentiment Analysis of tweets using lexicon bag of words model and Convolutional Neural Networks 

Glove file used: https://www.kaggle.com/watts2/glove6b50dtxt

### Requirements:
<br>
* sknn (Scikit Neural Networks)
* Global Vector Encoder (GloVe)/ Word to Vector Encoder (Word2Vec)
* Scipy (And its dependency packages)
* Numpy
* Tweepy (For Twitter data scraping)

### Usage:
Put in the glove and SSTb folders in each of the four model folders, then run python sentiment_xx.py
to run the default training session save the model to "sentiment_model.pkl".

### Training set:
11976 phrases from the Stanford Sentiment Treebank dataset or "SSTb". Every phrase is encoded into a representative matrix with dense 50 dimensional GloVe embeddings. 
The module sets a fixed limit of  15 words per sentence to ensure that the embedding matrix of sentences have a uniform dimension of 15X50.

Sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or
the overall contextual polarity or emotional reaction to a document, interaction, or event. It determines whether a piece of writing is positive,
negative or neutral.
