# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod

"""
"""
class FeatureExtractionInterface:
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_feature_set(self,train_data):
        pass
    
    @abstractmethod
    def fit_feature_model(self,train_data):
        pass
    
    @abstractmethod
    def save_feature_model(self,filepath):
        pass
        
    @abstractmethod
    def load_feature_model(self,filepath):
        pass 
    @abstractmethod
    def get_feature_model(self):
        pass

"""
Defines the interface for building a general purpose classifier
"""
class ClassifierInterface:
    
    __metaclass__ = ABCMeta
    
    """
    Abstract method for fitting a classification model with data
    
    Params:
    -------
    train_data - N-d array or list of lists containing the training data
    train_labels - Labels containing the class labels
    classifier - The classification model which is to be fitted with data
    
    Returns:
    --------
    Depends on the implementation, generally nothing
    """
    @abstractmethod
    def fit_classification_model(self,train_data,train_labels):
        pass
    
    """
    For a trained classifier given the test data predicts the class labels
    
    Params:
    -------
    test_data - N-d array or list of lists containing the test data
    classifier - Trained classification model
    
    Returns:
    --------
    List containing the predicted class labels
    """
    @abstractmethod
    def get_predicted_class_labels(self,test_data):
        pass
    
    """
    Persists a given classifier in the filesystem
    
    Params:
    --------
    filepath - Path containing the name of the file where the model is to be saved
    classifier - The trained classification model which is to be persisted
    
    Returns:
    --------
    Nothing just saves it
    """
    @abstractmethod
    def save_classifier(self,filepath):
        pass
    
    """
    Loads a given classifier from the filesystem
    
    Params:
    -------
    filepath - Path containing the name of the trained model file
    
    Returns:
    --------
    trained model which has been persisted
    """
    @abstractmethod
    def load_classifier(self,filepath):
        pass

"""
Defines the interface for a evaluation/scoring model (i.e. can be classification/regression/clustering/ranking)
"""
class EvaluationInterface:
    
    __metaclass__ = ABCMeta
    
    """
    Scores a given model using the actual_labels and predicted labels
    
    Params:
    -------
    actual_labels - List containing the ground truth of the observations
    predicted_labels - List containing the predicted class labels
    
    Returns
    --------
    Depends on the implementation
    """
    @abstractmethod
    def score_model(self,actual_labels,predicted_labels):
        pass

class EnsembleInterface:
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit_ensemble_model(self,train_data,train_labels):
        pass
    
    @abstractmethod
    def get_predicted_labels_ensemble(self,test_data):
        pass
    
    @abstractmethod
    def save_ensemble(self,filepath):
        pass
    
    @abstractmethod
    def load_ensemble(self,filepath):
        pass

class PredictPipelineInterface:
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_predicted_values(self,model,test_data):
        pass

class TrainPipelineInterface:
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def transform_raw_data_into_features(self,raw_data):
        pass
    
    @abstractmethod
    def train_model(self,model,train_data,train_labels):
        pass

    @abstractmethod
    def save_trained_model(self,model,filepath):
        pass

class ValidatePipelineInterface:
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def transform_raw_data_into_features(self,raw_data):
        pass
    
    @abstractmethod
    def get_model_output(self,model,test_data):
        pass
    
class ClassImbalanceInterface:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_balanced_input(self,X,y):
        pass
