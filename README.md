# image-captioning
# Introduction & Problem Statement:
With the exponential growth of image data on the internet and the increasing demand for accessibility in various applications, such as image search engines, social media platforms, and assistive technologies for the visually impaired, the need for effective image captioning systems has become paramount. 
 The objective of this project is to generate descriptive captions for images using deep learning techniques. 
# Proposed Methodology
# Source of Dataset:
https://www.kaggle.com/datasets/adityajn105/flickr8k?
The dataset consists of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

# Overview
Utilized Pretrained convolutional neural networks (CNNs) VGG16 for feature extraction from images and Long Short Term Memory (LSTM) for sequence generation, then created a comprehensive functional API model which integrates image features and textual context to generate descriptive captions for a wide range of images.


# Architecture
Here's a detailed breakdown of how the model works:
# Image Feature Extraction:
The VGG16 model, pre-trained on ImageNet, serves as the backbone for extracting high-level features from input images. In its original form, VGG16 is a deep convolutional neural network designed for image classification tasks. However, for tasks like image captioning, we utilize it as a feature extractor rather than a classifier.
To repurpose VGG16 for feature extraction, removed its last layer, which is typically a dense layer responsible for classifying images into one of the ImageNet categories. The output of the modified VGG16 model is a feature vector that captures the salient characteristics of the input image, effectively summarizing its content. This feature vector serves as a rich representation of the image, which can be further processed and used.


# Sequence Processing:
To process the textual captions and prepare them as inputs for the model, several steps are undertaken:
Data Parsing and Cleaning:
Initially, the captions are parsed from the provided document, splitting them based on delimiters and cleaning any extraneous characters or whitespace.
Each caption is then formatted and organized with respect to its corresponding image, resulting in a mapping between image IDs and their associated captions.
# Text Tokenization:
•	The textual data is tokenized using a tokenizer object, which assigns a unique integer index to each word in the vocabulary. This process converts the raw textual data into a numerical format that can be understood by the model.
Vocabulary Size Determination:
•	The size of the vocabulary is determined based on the total number of unique words present in the tokenized captions, augmented by one to account for an additional "unknown" token.
Maximum Caption Length Calculation:
•	The maximum length of the captions is determined by finding the length of the longest caption among all the provided captions. This ensures that all captions are processed uniformly during training and inference.
Sequence Embedding:
•	Each word index in the captions is embedded into a dense vector representation using an embedding layer. This layer maps each word index to a continuous vector space, allowing the model to learn semantic relationships between words.
Dropout Regularization:
•	To prevent overfitting and improve the generalization capability of the model, dropout regularization is applied to the embedded sequences. This technique randomly drops a fraction of the input units during training, forcing the model to learn more robust and generalized representations.
Sequential Modeling with LSTM:
•	The processed sequences are then passed through a Long Short-Term Memory (LSTM) layer, which is a type of recurrent neural network (RNN) capable of learning long-range dependencies in sequential data.
•	The LSTM layer processes the embedded sequences, capturing temporal dependencies between words and generating contextually rich representations of the captions. Then it is fed to the model combined with features extracted from images using the VGG16 model

# Model Input, Processing, and Output:

Input:
The model receives two inputs:
•	The image feature vector, extracted from the VGG16 model, represents the visual content of the input image.
•	A word index representing the starting token, such as "startseq", is provided as the initial input for caption generation.
Processing:
•	The image feature vector and the word index are processed by the model in conjunction.
•	The model utilizes its learned parameters to generate a probability distribution over the entire vocabulary of words.
•	This probability distribution indicates the likelihood of each word in the vocabulary being the next word in the caption sequence, given the input image features and the current context represented by the word index.
Output:
•	The model produces a vector of probabilities, where each element represents the likelihood of a specific word in the vocabulary being the next word in the caption sequence.
•	These probabilities are normalized to sum up to 1, ensuring that they represent a valid probability distribution.
•	The word with the highest probability is selected as the predicted next word in the caption sequence.

# Converting Probabilities to Text and Generating Captions:
Selection of Next Word:
•	After the model produces a vector of probability indicating the likelihood of each word in the vocabulary being the next word in the caption sequence, the word with the highest probability is selected as the predicted next word.
•	This selection process ensures that the model chooses the most likely word to follow the current context represented by the input image features and the preceding words in the caption sequence.
Word Index Conversion:
•	The index of the selected word is determined from the vocabulary based on its position in the probability vector.
•	This index corresponds to a specific word in the vocabulary, representing the predicted next word in the caption sequence.
Textual Representation:
•	Once the index of the predicted next word is obtained, it is converted back into its corresponding textual representation using the tokenizer employed during model training.
•	This conversion maps the numerical index back to the actual word in the vocabulary, allowing the model to generate a textual representation of the predicted next word.
# Caption Formation:
•	The predicted next word is appended to the existing caption sequence, forming a coherent and contextually relevant caption. This process is repeated iteratively, with each predicted word serving as the input for the next iteration until a maximum caption length is reached or a special end token, such as "endseq", is predicted.
