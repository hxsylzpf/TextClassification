# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 22:46:01 2017
"""

from interfacefactory import FeatureExtractionInterface
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
import classificationutils
import doc2vec
import numpy as np

"""
"""
class TfIdf(FeatureExtractionInterface):
    
    def __init__(self,max_feature_num=5000,ngram_range_tuple=(1,3)):
        self.tf_idf = TfidfVectorizer(max_features=max_feature_num,ngram_range=ngram_range_tuple)
    
    def get_feature_set(self,train_data):
        return self.tf_idf.transform(train_data)
    
    def fit_feature_model(self,train_data):
        self.tf_idf.fit(train_data)
    
    def save_feature_model(self,filepath):
        classificationutils.save_classifier(filepath,self.tf_idf)
    
    def load_feature_model(self,filepath):
        return classificationutils.load_classifier(filepath)
    
    def get_feature_model(self):
        return self.tf_idf

"""
"""
class BagOfWords(FeatureExtractionInterface):
    
    def __init__(self,max_feature_num=5000,ngram_range_tuple=(1,3)):
        self.bag_of_words = CountVectorizer(max_features=max_feature_num,ngram_range=ngram_range_tuple)
    
    def get_feature_set(self,train_data):
        return self.bag_of_words.transform(train_data)
    
    def fit_feature_model(self,train_data):
        self.bag_of_words.fit(train_data)
    
    def save_feature_model(self,filepath):
        classificationutils.save_classifier(filepath,self.bag_of_words)
    
    def load_feature_model(self,filepath):
        return classificationutils.load_classifier(filepath)
    
    def get_feature_model(self):
        return self.bag_of_words

"""
"""

class FeatureHash(FeatureExtractionInterface):
    
    def __init__(self,max_feature_num=5000,input_data_type='string'):
        self.feature_hash = FeatureHasher(n_features=max_feature_num,input_type=input_data_type)
    
    def get_feature_set(self,train_data):
        return self.feature_hash.transform(train_data)
    
    def fit_feature_model(self,train_data):
        self.feature_hash.fit(train_data)
    
    def save_feature_model(self,filepath):
        classificationutils.save_classifier(filepath,self.feature_hash)
    
    def load_feature_model(self,filepath):
        return classificationutils.load_classifier(filepath)
    
    def get_feature_model(self):
        return self.feature_hash

"""
"""

class DocumentVector(FeatureExtractionInterface):
    
    def __init__(self,filepath):
        self.filepath = filepath
    
    def get_feature_set(self,filepath=None):
        if filepath != None:
            model = doc2vec.load_model(self.filepath)
            return np.array(model.docvecs)
        return np.array(self.model.docvecs)
    
    def fit_feature_model(self,datafilepath=None):
        if datafilepath != None:
            self.filepath = datafilepath
        sentences = doc2vec.get_tagged_sentences(self.filepath)
        self.model = doc2vec.get_trained_model_in_epoch(sentences)
    
    def save_feature_model(self,filepath):
        doc2vec.save_model(self.model,filepath) 
    
    def load_feature_model(self,filepath):
        return doc2vec.load_model(filepath)
    
    def get_feature_model(self):
        return self.model 
