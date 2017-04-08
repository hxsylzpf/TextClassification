# -*- coding: utf-8 -*-

from interfacefactory import EvaluationInterface
from sklearn import metrics

"""
Provides a concrete implementation of the EvaluationInterace and also adds more functions
"""
class ClassificationMetrics(EvaluationInterface):
    
    def __init__(self):
        pass
    
    """
    Implementation of the score model abstract method
    
    Params:
    -------
    actual_labels - Ground truth labels for a given classification task
    predicted_labels - Model predicted labels for a given task
    
    Returns:
    ---------
    F1-score for the given model
    """
    def score_model(self,actual_labels,predicted_labels):
        return metrics.f1_score(actual_labels,predicted_labels)
    
    """
    Returns the accuracy of a given classification model
    
    Params:
    -------
    actual_labels - Ground truth labels for a given classification task
    predicted_labels - Model predicted labels for a given task
    
    Returns:
    ---------
    Accuracy of the given model
    """
    
    def get_accuracy(self,predicted_values,actual_values):
        return metrics.accuracy_score(actual_values,predicted_values) 

    """
    Returns the classification report of a given classification model
    
    Params:
    -------
    actual_labels - Ground truth labels for a given classification task
    predicted_labels - Model predicted labels for a given task
    
    Returns:
    ---------
    String containing the classification report of the given model
    """
    def get_classification_report(self,predicted_values,actual_values):
        return metrics.classification_report(actual_values,predicted_values)
        
    """
    Returns the precision of a given classification model
    
    Params:
    -------
    actual_labels - Ground truth labels for a given classification task
    predicted_labels - Model predicted labels for a given task
    
    Returns:
    ---------
    Precision of the given model
    """
    def get_precision(self,predicted_values,actual_values):
        return metrics.average_precision_score(actual_values,predicted_values)
    
    """
    Returns the recall of a given classification model
    
    Params:
    -------
    actual_labels - Ground truth labels for a given classification task
    predicted_labels - Model predicted labels for a given task
    
    Returns:
    ---------
    Recall of the given model
    """
    def get_recall(self,predicted_values,actual_values):
        return metrics.recall_score

