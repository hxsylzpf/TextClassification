# -*- coding: utf-8 -*-

from interfacefactory import ClassifierInterface
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.svm import SVC
import classificationutils

"""
Wrapper around the scikit-learn Random forests implementation using the classifier interface
"""
class RandomForest(ClassifierInterface):
    
    """
    Initializes the Random Forest object with the default number of estimators
    
    Params:
    -------
    n_estimators - Number of decision stumps to be built (we have set n_estimators=51)
    
    Returns:
    --------
    Nothing dude, it's a constructor
    """ 
    
    def __init__(self,n_estimators=51):
        self.num_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators = self.num_estimators)
    
    """
    Given the training data and the training class labels fits the classification model
    
    Params:
    ------
    train_data - N-d array or equivalent data-structure containing the training data
    train_labels - List containing the class labels
    
    Returns:
    --------
    Nothing dude, get over it. Just fits the model !!!
    """ 
    
    def fit_classification_model(self,train_data,train_labels):
        self.model.fit(train_data,train_labels)
    
    """
    Predicts the class-labels for a given set of data points
    
    Params:
    -------
    test_data - N-d array containing the test data points
    
    Returns:
    --------
    List containing the test data class labels
    """
    
    def get_predicted_class_labels(self,test_data):
        return self.model.predict(test_data)
    
    """
    Returns the probabilities associated with each class in the test-data
    
    Params:
    -------
    test-data - Nd-array containing the test data points
    
    Returns:
    --------
    N-d array containing the class probabilities of each data point 
    the order of the classes is the same as given in the classes_ attributes (usually in sorted order)
    """
    
    def get_predicted_class_prob(self,test_data):
        return self.model.predict_proba(test_data)
    
    """
    Saves a given classifier to the file system
    
    Params:
    -------
    filepath - filename/filepath to which to persist the model
    
    Returns:
    --------
    Nah,nothing doc just persists it
    """
    
    def save_classifier(self,filepath):
        classificationutils.save_classifier(filepath,self.model)
    
    """
    Loads a given classification model from the filesystem (which is already persisted)
    
    Params:
    -------
    filepath - filename/filepath containing the trained model
    
    Returns:
    ---------
    Trained model which has already been saved
    """
    
    def load_classifier(self,filepath):
        return classificationutils.load_classifier(filepath) 
    
    """
    Returns the scikit-learn model which has been trained
    
    Params: None
    -------
    
    Returns:
    --------
    sklearn classifier object associated with the class
    """
    
    def get_classification_model(self):
        return self.model

"""
Wrapper around the scikit-learn Gradient Boosting Classifer implementation using the classifier interface
"""

class GradientBoost(ClassifierInterface):
    
    def __init__(self,n_estimators=101):
        self.num_estimators = n_estimators
        self.model = GradientBoostingClassifier(n_estimators = self.num_estimators)
    
    """
    Given the training data and the training class labels fits the classification model
    
    Params:
    ------
    train_data - N-d array or equivalent data-structure containing the training data
    train_labels - List containing the class labels
    
    Returns:
    --------
    Nothing dude, get over it. Just fits the model !!!
    """
    
    def fit_classification_model(self,train_data,train_labels):
        self.model.fit(train_data,train_labels)
    
    """
    Predicts the class-labels for a given set of data points
    
    Params:
    -------
    test_data - N-d array containing the test data points
    
    Returns:
    --------
    List containing the test data class labels
    """
    
    def get_predicted_class_labels(self,test_data):
        return self.model.predict(test_data)
    
    """
    Returns the probabilities associated with each class in the test-data
    
    Params:
    -------
    test-data - Nd-array containing the test data points
    
    Returns:
    --------
    N-d array containing the class probabilities of each data point 
    the order of the classes is the same as given in the classes_ attributes (usually in sorted order)
    """

    def get_predicted_class_prob(self,test_data):
        return self.model.predict_proba(test_data)
    
    """
    Saves a given classifier to the file system
    
    Params:
    -------
    filepath - filename/filepath to which to persist the model
    
    Returns:
    --------
    Nah,nothing doc just persists it
    """
    
    def save_classifier(self,filepath):
        classificationutils.save_classifier(filepath,self.model)
    
    """
    Loads a given classification model from the filesystem (which is already persisted)
    
    Params:
    -------
    filepath - filename/filepath containing the trained model
    
    Returns:
    ---------
    Trained model which has already been saved
    """
    
    def load_classifier(self,filepath):
        return classificationutils.load_classifier(filepath)
        
    """
    Returns the scikit-learn model which has been trained
    
    Params: None
    -------
    
    Returns:
    --------
    sklearn classifier object associated with the class
    """
    
    def get_classification_model(self):
        return self.model

"""
Wrapper around the Logistic Regression implementation of scikit-learn using the ClassifierInterface
"""        
class LogisticRegression(ClassifierInterface):
    
    def __init__(self,penalty='l2',solver='sag'):
        self.penalty = penalty
        self.solver = solver
        self.model = linear_model.LogisticRegression(penalty=self.penalty ,solver=self.solver)
    
    """
    Given the training data and the training class labels fits the classification model
    
    Params:
    ------
    train_data - N-d array or equivalent data-structure containing the training data
    train_labels - List containing the class labels
    
    Returns:
    --------
    Nothing dude, get over it. Just fits the model !!!
    """

    def fit_classification_model(self , train_data, train_labels):
        self.model.fit(train_data,train_labels)
    
    """
    Predicts the class-labels for a given set of data points
    
    Params:
    -------
    test_data - N-d array containing the test data points
    
    Returns:
    --------
    List containing the test data class labels
    """

    def get_predicted_class_labels(self,test_data):
        return self.model.predict(test_data)
    
    """
    Returns the probabilities associated with each class in the test-data
    
    Params:
    -------
    test-data - Nd-array containing the test data points
    
    Returns:
    --------
    N-d array containing the class probabilities of each data point 
    the order of the classes is the same as given in the classes_ attributes (usually in sorted order)
    """
    
    def get_predicted_class_prob(self,test_data):
        return self.model.predict_proba(test_data)
    
    """
    Saves a given classifier to the file system
    
    Params:
    -------
    filepath - filename/filepath to which to persist the model
    
    Returns:
    --------
    Nah,nothing doc just persists it
    """
    
    def save_classifier(self,filepath):
        classificationutils.save_classifier(filepath,self.model)
    
    """
    Loads a given classification model from the filesystem (which is already persisted)
    
    Params:
    -------
    filepath - filename/filepath containing the trained model
    
    Returns:
    ---------
    Trained model which has already been saved
    """
    
    def load_classifier(self,filepath):
        return classificationutils.load_classifier(filepath)
        
    """
    Returns the scikit-learn model which has been trained
    
    Params: None
    -------
    
    Returns:
    --------
    sklearn classifier object associated with the class
    """
    
    def get_classification_model(self):
        return self.model

"""
Wrapper around the SVM implementation of scikit-learn using the ClassifierInterface
"""
class SVM(ClassifierInterface):
    
    def __init__(self,kernel='linear',class_weight='balanced',probability=True):
        self.kernel = kernel
        self.class_weight = class_weight
        self.probability = probability
        self.model = SVC(kernel=self.kernel,class_weight=self.class_weight,probability=self.probability)
    
    """
    Given the training data and the training class labels fits the classification model
    
    Params:
    ------
    train_data - N-d array or equivalent data-structure containing the training data
    train_labels - List containing the class labels
    
    Returns:
    --------
    Nothing dude, get over it. Just fits the model !!!
    """
    
    def fit_classification_model(self,train_data,train_labels):
        self.model.fit(train_data,train_labels)
    
    """
    Predicts the class-labels for a given set of data points
    
    Params:
    -------
    test_data - N-d array containing the test data points
    
    Returns:
    --------
    List containing the test data class labels
    """
    
    def get_predicted_class_labels(self,test_data):
        return self.model.predict(test_data)
    
    """
    Returns the probabilities associated with each class in the test-data
    
    Params:
    -------
    test-data - Nd-array containing the test data points
    
    Returns:
    --------
    N-d array containing the class probabilities of each data point 
    the order of the classes is the same as given in the classes_ attributes (usually in sorted order)
    """
    
    def get_predicted_class_prob(self,test_data):
        return self.model.predict_proba(test_data)
    
    """
    Saves a given classifier to the file system
    
    Params:
    -------
    filepath - filename/filepath to which to persist the model
    
    Returns:
    --------
    Nah,nothing doc just persists it
    """
    
    def save_classifier(self,filepath):
        classificationutils.save_classifier(filepath,self.model)
    
    """
    Loads a given classification model from the filesystem (which is already persisted)
    
    Params:
    -------
    filepath - filename/filepath containing the trained model
    
    Returns:
    ---------
    Trained model which has already been saved
    """
    
    def load_classifier(self,filepath):
        return classificationutils.load_classifier(filepath)
    
    """
    Returns the scikit-learn model which has been trained
    
    Params: None
    -------
    
    Returns:
    --------
    sklearn classifier object associated with the class
    """
    def get_classification_model(self):
        return self.model
