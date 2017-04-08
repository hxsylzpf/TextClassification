# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 00:01:29 2017
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import classificationutils
import doc2vec
import numpy as np

input_data_filepath = 'data/cleaned_conversation.txt'
input_label_filepath = 'data/train_output.csv'
output_file = 'results.txt'

model = doc2vec.load_model('C:\\Users\\senjuti\\Desktop\\Reddit\\trained model\\doc2vec_feature_model.bin')
document_vectors = np.array(model.docvecs)

##train_dataframe = pd.read_csv(input_data_filepath,sep='\n',header=None)
#label_dataframe = pd.read_csv(input_label_filepath)
#out_file = open(output_file,'w')
#
##conversation_list = list(train_dataframe[0].values)
#train_labels = list(label_dataframe['category'].values)
#
##conversation_list = conversation_list[:5000]
##train_labels = train_labels[:5000]
#
#rf = RandomForestClassifier(n_estimators=101)
#linear_svm = SVC(kernel='linear')
#grad = GradientBoostingClassifier()
#
#classifiers = [rf,linear_svm,grad]
#classifier_names = ['Random Forests','Linear SVM','Gradient Boosting'] 
#
##tf_idf = classificationutils.load_classifier('trained model/tfidf_feature_model.bin')
##bag_of_words = classificationutils.load_classifier('trained model/bagofwords_feature_model.bin')
#
#feature_sets = [document_vectors] #,bag_of_words]
#feature_set_names = ['Doc2Vec'] #,'Bag of words']
#
#for feature_model,feature_name in zip(feature_sets,feature_set_names): 
#    
#    print '****** For Feature : ', feature_name
#    out_file.write(str('****** For Feature : ' + feature_name + ' ********\n'))
#    out_file.flush()
#    
#    train_data = document_vectors
#    X_train,X_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=42) 
#    
#    for classifier,classifier_name in zip(classifiers,classifier_names): 
#        
#        print '******* For Classifier : ',classifier_name
#        out_file.write(str('******* For Classifier : ' + classifier_name + ' **********\n'))
#        out_file.flush() 
#        
#        classifier.fit(X_train,y_train)
#        predicted_values = classifier.predict(X_test)
#        print metrics.classification_report(y_test,predicted_values)
#        out_file.write(str(metrics.classification_report(y_test,predicted_values)))
#        print 'Accuracy : ',metrics.accuracy_score(y_test,predicted_values)
#        out_file.write(str('Accuracy : ' + str(metrics.accuracy_score(y_test,predicted_values)) + '\n'))
#        print '*******************'
#        out_file.write('*******************\n')
#        out_file.flush()
#        classificationutils.save_classifier(feature_name + '_' + classifier_name + '.bin',classifier)
#
#out_file.close()
