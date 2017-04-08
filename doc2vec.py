# -*- coding: utf-8 -*-

from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import Doc2Vec

"""
Given a filename/filepath as the input returns an object of tagged document type

Params:
--------
filepath - Filename containing the documents

Returns:
--------
List of tagged documents
"""
def get_tagged_sentences(filepath):
    sentences = TaggedLineDocument(filepath)
    return sentences

"""
Trains a Doc2Vec model on sentences and documents passed

Params:
--------
sentences - List of tagged sentence objects for training
dimension - Length of the feature vector for each document

Returns:
---------
A trained Doc2Vec model 
"""
def get_trained_model(sentences,dimension=300):
    model = Doc2Vec(documents=sentences,size=dimension)
    return model

"""
Trains a given Doc2Vec model in several epochs not in a single shot

Params:
--------
sentences - List of tagged sentence objects to train the Doc2vec model
dimension - Length of the feature vector to be generated in the Doc2Vec model
epoch - Integer containing the number of epochs for which the model needs to be trained

Returns:
--------
A trained Doc2Vec model which is specified by the given parameters
"""
def get_trained_model_in_epoch(sentences,dimension=300,epoch=10):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025,size=dimension)
    model.build_vocab(sentences)
    for i in range(epoch):
        model.train(sentences)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model

"""
Given a trained Doc2Vec model and a new document infers it vector representation

Params:
--------
model - trained Doc2Vec model which is to be used
sentence_word_list - document tokenized as words 

Returns:
--------
numpy ndarray containing the vector representation of the given document
"""
def get_document_vector(model,sentence_word_list):
    return model.infer_vector(sentence_word_list)

"""
For a given document returns the list top N most similar documents

Params:
--------
model - Trained Doc2Vec model which is to be used
sentence_word_list - List containing the tokenized words in a sentence
neighbors - Integer containing the top N nearest documents for a given document

Returns:
---------
List containing tuple of matching document number and match score
"""
def get_most_similar_vectors(model,sentence_word_list,neighbors):
    sentence_vector = get_document_vector(model,sentence_word_list)  
    return model.docvecs.most_similar(positive=[sentence_vector],topn=neighbors)

"""
Persists a trained model in the filesystem

Params:
--------
model - Trained Doc2Vec model which is to be saved
filepath - Path to the file where the model is to be saved

Returns:
--------
Nothing just saves the file duh !!!
"""
def save_model(model,filepath):
    model.save(filepath)

"""
Loads a trained Doc2Vec model from the filesystem

Params:
--------
filepath - Path containing the file where the trained model is to be stored

Returns:
--------
Trained model loaded from memory
"""
def load_model(filepath):
    model = Doc2Vec.load(filepath)
    return model

"""
Creates the pipeline for training and saving the Doc2Vec model

Params:
--------
data_filepath - filepath containing the data to be trained upon
output_model_filepath - Output filename where the model has to be persisted
dimension - Integer containing the length of the feature vector (default value-300)
epoch - Integer containing the number of epochs in which the model is to be trained

Returns:
--------
Nothing dude, just persists the model to file
"""
def model_train_and_save_pipeline(data_filepath,output_model_filepath,dimension=300,epoch=10):
    sentences = get_tagged_sentences(data_filepath)
    model = get_trained_model_in_epoch(sentences,dimension=dimension,epoch=epoch)
    save_model(model,output_model_filepath)
