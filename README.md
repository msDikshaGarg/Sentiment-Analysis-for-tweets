# Sentiment-Analysis-for-tweets
Sentiment Analysis of tweets using lexicon bag of words model and Convolutional Neural Networks 

Glove file used: https://www.kaggle.com/watts2/glove6b50dtxt

### Requirements:
* sknn (Scikit Neural Networks)
* Global Vector Encoder (GloVe)/ Word to Vector Encoder (Word2Vec)
* Scipy (And its dependency packages)
* Numpy
* Tweepy (For Twitter data scraping)

### Usage:
Put in the glove and SSTb folders in each of the four model folders, then run python sentiment_xx.py
to run the default training session save the model to "sentiment_model.pkl".

### About the training set:
* 11976 phrases from the Stanford Sentiment Treebank dataset or "SSTb". 
* Every phrase is encoded into a representative matrix with dense 50 dimensional GloVe embeddings. 
* The module sets a fixed limit of  15 words per sentence to ensure that the embedding matrix of sentences have a uniform dimension of 15X50.

### Description:
Sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or
the overall contextual polarity or emotional reaction to a document, interaction, or event. It determines whether a piece of writing is positive,
negative or neutral.
Sentiment analysis is extremely useful in social media monitoring as it allows us to gain an overview of the wider public opinion behind
certain topics. 

## 1. Lexicon Model: 

This could be considered as a modified Bag-Of-Words model, and the modification being the consideration of context in the sentences. Primarily
the context generated by the preceding words/phrases are taken into consideration. Then we begin by constructing a list inspired by examining
existing well-established sentiment word-banks (we used an existing lexicon dictionary). To this, we next incorporate numerous lexical features
common to sentiment expression in microblogs, including a full list of Western-style emoticons (for example, “:-)” denotes a “smiley face” and
generally indicates positive sentiment), sentiment-related acronyms and initialisms (e.g., LOL and ROFL are both sentiment laden initialisms), and
commonly used slang with sentiment value (e.g., “nah”, “meh” and “giggly”). The Lexicon dictionary we used collected intensity ratings on
each of the candidate lexical features from ten independent human raters (for a total of 90,000+ ratings). Features were rated on a scale from
“Extremely Negative” to “Extremely Positive”, with allowance for “Neutral (or Neither, N/A)”.

### Implementation: 
* Loading the lexicon dictionary file to a dictionary for use during runtime.
* Check for words which are in capitals and increase their intensity score by a factor.
* Check for the effect of preceding word. e.g. “Pretty good” increases the intensity of the positive sentiment of the sentence, while “Pretty bad”
increases the intensity of the negative sentiment of the sentence.
* Check for effect of preceding words like ‘Never’. e.g. “Never so Good” and “Never so Bad” have opposite sentiment of equal intensity, but have
higher intensities than “So Good” and “So Bad”. 
* Check for effect of idioms like "yeah right”, "the bomb”, "bad ass” etcetera.
* Check for effect of the words like “Least”, “But”, for these have an intensity decreasing sense. e.g. “At least better than him” has a lower
intensity positive sentiment than the same sentence without “Least”, “Better than him”.
* Increase sentiment intensity of the text piece for the effect of punctuations; “!”, “?”. “So Good.” has a lower intensity than “So Good!!!”
* Normalise scores between -1 (Extreme negative) to 1 (Extreme positive).
* Separate scores for sentence sentiment polarity; for 0<score<=1: is positive,  0: is neutral and -1<=score<0: is negative.
* Make a dictionary of the scores for every sentence by adding the scores to achive a compound score.

### Results: 
Data which was to be analysed was scraped from twitter, particularly to test sentimental orientation on; The demonetisation steps taken by the Indian Government
and General Tweets by the POTUS Donald Trump
Results for Lexicon and rule-based Model:
##### Demonetisation:
 * Total: 44.0227576975 % Positive Tweets.
 * Total: 29.6854082999 % Neutral Tweets.
 * Total: 26.2918340027 % Negative Tweets.
 * Total: 14940.0 Tweets.
##### Donald Trump Tweets:
 * Total: 56.5576621607 % Positive Tweets.
 * Total: 16.2817551135 % Neutral Tweets.
 * Total: 27.1605827258 % Negative Tweets.
 * Total: 3600.0 Tweets.

## 2. Convolutional Neural Networks:

Particularly Convolutional Neural Networks , CNNs are basically just several layers of convolutions with nonlinear activation
functions like ReLU or tanh applied to the results. In a traditional feedforward neural network we connect each input neuron to each output
neuron in the next layer. That’s also called a fully connected layer, or affine layer. In CNNs we don’t do that. Instead, we use convolutions over the
input layer to compute the output. This results in local connections, where each region of the input is connected to a neuron in the output.
Primarily the CNNs were analysed for their test accuracy on Stanford Sentiment Tree Bank for four different activation functions; Rectifier, Tanh,
ExpLin and Sigmoid.

### Implementation: 
* Encode text pieces to vector representation.
* Load the Stanford Sentiment Treebank (For Training and Testing purposes).
* Train the model.

### Results: 
Using ExpLin as activation function, Test accuracy: 0.693687374749  
Using Rectifier as activation function, Test accuracy: 0.744729451764  
Using Sigmoid as activation function, Test accuracy: 0.639328657315  
Using Tanh as activation function, Test accuracy: 0.672895791583  
Hence Rectifier function turns out to be the best suited for the chosen test dataset.
So the CNN using Rectifier activation function is chosen to analyse the two scraped datasets;
##### Demonetisation:
 * Total: 46.9752026757 % Positive Tweets.
 * Total: 24.2968368231 % Neutral Tweets.
 * Total: 28.7279605012 % Negative Tweets.
 * Total: 14940.0 Tweets. 
##### Donald Trump Tweets:
 * Total: 52.5762796027 % Positive Tweets.
 * Total: 19.2660582715 % Neutral Tweets.
 * Total: 28.1576621258 % Negative Tweets.
 * Total: 3600.0 Tweets.


