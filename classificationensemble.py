# -*- coding: utf-8 -*-

from interfacefactory import EnsembleInterface
from sklearn.ensemble import VotingClassifier
import classificationutils

class EnsembleVotingClassifier(EnsembleInterface):
    
    def __init__(self,estimators,voting='hard',weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.ensemble = VotingClassifier(estimators=self.estimators,voting=self.voting,weights=self.weights)
    
    def fit_ensemble_model(self,train_data,train_labels):
        self.ensemble.fit(train_data,train_labels)
    
    def get_predicted_labels_ensemble(self,test_labels):
        return self.ensemble.predict(test_labels)
    
    def save_ensemble(self,filepath):
        classificationutils.save_classifier(filepath,self.ensemble)
    
    def load_ensemble(self,filepath):
        return classificationutils.load_classifier(filepath)
