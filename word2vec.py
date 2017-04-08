# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 17:47:18 2017

"""
import gensim

"""
Creates the word2vec model containing the distributed word representations

Params:
---------
sentences - List of list containing the words
dimension - The dimension of the vector representation of the word

Returns:
----------
Trained word2vec model object
"""
def get_trained_model(sentences,dimension): 
    
    model = gensim.models.Word2Vec(sentences,size=dimension,min_count=1) 
    
    return model    

"""
Processes a sentence by spliting it up into words

Params:
--------
filename - file from which the words have to be read

Returns:
---------
List of list containing the sentences split up by individual words
"""
def get_processed_sentences(filename): 
    
    sentences = gensim.models.word2vec.LineSentence(filename) 
    
    return sentences

"""
Converts a numpy array into a csv

Params:
---------
nparray - numpy array containing the numbers

Returns:
---------
String containing the csv representation of the numpy array
"""
def array_to_string(nparray): 
    
    s = str(nparray)
    s = s.replace('[','').replace(']','')
    s = s.replace('\n','')
    csv = ','.join(s.split())     
    
    return csv

"""
Given the trained model containing the vocabulary writes it to a file

Params:
--------
model - trained word2vec model
sentences - List of sentences trained upon
file_to_write - Output file to which the word vectors are written to

Returns:
----------
Nothing just writes the output to a given file
"""
def write_vectors_to_file(model,sentences,file_to_write): 
    
    write_file = open(file_to_write,'w') 
    
    for sentence in sentences:
        for word in sentence:
            vector = array_to_string(model[word]) #s.encode('ascii','ignore')
            write_file.write(str(word.encode('ascii','ignore')) + '=' + vector + '\n')
            write_file.flush()
    
    write_file.close()
    
"""
Defines the entire pipeline for converting the words into their vectorized representations

Params:
--------
input_file - file containing the input words
output_file - file where the output vector representations will be written
dimension - The dimension of the word vector being formed

Returns:
---------
Doesn't return anything just writes it out to a file
"""
def word_vector_process_pipeline(input_file,output_file,dimension):  
    
    sentences = get_processed_sentences(input_file)
    model = get_trained_model(sentences,dimension)
    write_vectors_to_file(model,sentences,output_file)

"""
Defines the pipeline for converting words into their vectorized representation and saving the trained model

Params:
--------
input_file - Input file containing the text for the given model
model_file - Output file where the serialized model is stored
dimension - Dimension size of the output vector

Returns:
---------
Nothing just saves the model to the filesystem
"""
def word_vector_model_save_pipeline(input_file,model_file,dimension):
    
    sentences = get_processed_sentences(input_file)
    model = get_trained_model(sentences,dimension)
    model.save(model_file)

"""
Loads a trained word2vec model from the filesystem

Params:
--------
model_file - File containing the binarized representation of the model

Returns:
---------
The trained word2vec model from the filesystem 
"""
def get_trained_model_from_file(model_file):
    
    model = gensim.models.Word2Vec.load(model_file)
    return model
