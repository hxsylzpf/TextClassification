# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 22:15:52 2017

"""

import pandas as pd
import preprocess
from textfeatures import TfIdf
from textfeatures import BagOfWords
from textfeatures import FeatureHash
from textfeatures import DocumentVector
import time

input_data_filepath = 'data/train_input.csv'
input_label_filepath = 'data/train_output.csv'

thread_dataframe = pd.read_csv(input_data_filepath)
label_dataframe = pd.read_csv(input_label_filepath)

conversation_list = list(thread_dataframe['conversation'].values)
cleaned_conversation_list = preprocess.text_clean_pipeline_list(conversation_list)

start = time.time()

preprocess.write_sentences_to_file(cleaned_conversation_list,'data/cleaned_conversation.txt')

tf_idf = TfIdf()
bag_of_words = BagOfWords()
feature_hash = FeatureHash() 

doc2vec_feat = DocumentVector(filepath='data/cleaned_conversation.txt')

doc2vec_feat.fit_feature_model()
doc2vec_feat.save_feature_model('trained_moodel/doc2vec_feature_model.bin')

feature_hash.fit_feature_model(preprocess.tokenize_string_list(cleaned_conversation_list,separator=' '))
feature_hash.save_feature_model('trained model/featurehash_model.bin')

tf_idf.fit_feature_model(cleaned_conversation_list)
tf_idf.save_feature_model('trained model/tfidf_feature_model.bin')

bag_of_words.fit_feature_model(cleaned_conversation_list)
bag_of_words.save_feature_model('trained model/bagofwords_feature_model.bin')

end = time.time()

print 'Total time',(end-start)/1000.0
