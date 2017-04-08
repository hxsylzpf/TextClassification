""" 
Created 5 Feb, 2017
Author : Inderjot Kaur Ratol
"""
import numpy as np
import pandas  as pd
import warnings
import matplotlib.pyplot as plt
import os
import csv
import re
import sys
import operator
import PreProcess
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score,classification_report
from nltk.probability import FreqDist


warnings.filterwarnings("ignore")

"""final dictionary containing all categories as keys,
and values as the dictionaryies of words with frequency counts."""
documentClass={}

""" this dictionary is used as one per category 
and includes thwe words with their corressponsing counts"""
bag_of_words={}
"""  used to calculate the TF-IDF values"""
words_in_multiple_conv={}
totalConversations=0

word_count_per_category={}
total_words=0
""" contains the calculated conditional probabilites of words occurring in a conversation """
prob_hockey={}
prob_movies={}
prob_news={}
prob_nba={}
prob_nfl={}
prob_politics={}
prob_soccer={}
prob_worldnews={}
""" initializing prior probabilites of each category"""
prior_hockey=1
prior_movies=1
prior_nba=1
prior_news=1
prior_nfl=1
prior_politics=1
prior_soccer=1
prior_worldnews=1



""" PreProcess every conversation and 
calculate the frequency of each word in the conversation 
and put it in the bag_of_words"""
def processConversation(conversation):
	global bag_of_words
	bag_of_words={}
	sentences=conversation.split(".")
	tokenized=PreProcess.tokenize_sentences(sentences)
	filtered=PreProcess.RemovePunctAndStopWords(tokenized)
	bag_of_words= FreqDist(word.lower() for word in filtered)

""" add the bag_of_words to a class category . 
This method is called after processing each conversation."""
def addWordsInClassCategory(category):
	global documentClass,bag_of_words
	if category in documentClass.keys():
		new_dict=merge_two_dicts(documentClass[category],bag_of_words)
		documentClass[category]=new_dict
	else:
		documentClass[category]=bag_of_words
		
""" This method is used to add a word's frequency calculated over multiple conversations.
 Needed to calculate TF-IDF values"""
def addTermFrequency(words):
	global words_in_multiple_conv
	for word in words:
		if word in words_in_multiple_conv:
			words_in_multiple_conv[word]=int(words_in_multiple_conv[word])+1
		else:
			words_in_multiple_conv[word]=1

"""Given two dicts, merge them into a new dict as a shallow copy."""
def merge_two_dicts(x, y):
	z = x.copy()
	z.update(y)
	return z

def finalRun(train_set,y_values,test_set,test_ids):
	prior_counts=y_values.groupby('category')['category'].count()
	initializePriors(prior_counts)
	NaiveBayesImplemented(train_set,y_values.values,test_set,test_ids)

def initProcessing():
	global documentClass,totalConversations
	
	conversations= pd.read_csv('train_input.csv')
	conversations_test= pd.read_csv('test_input.csv')
	# selecting only the conversations column 
	conversOnly=conversations[["conversation"]]
	test_set=conversations_test["conversation"]
	test_ids=conversations_test["id"]
	totalConversations=conversOnly.shape[0]
	y_data=pd.read_csv('train_output.csv')

	# selecting only the category column 
	y_values=y_data[["category"]]
	
	
	""" run 6-folds cross validation"""
	k_fold = KFold(n=totalConversations, n_folds=6)
	for train_indices, test_indices in k_fold:
		train_text = conversOnly.iloc[train_indices]['conversation'].values
		train_y = y_values.iloc[train_indices]['category'].values.astype(str)
		for_counts=y_values
		prior_counts=for_counts.groupby('category')['category'].count()
		initializePriors(prior_counts)
		test_text = conversOnly.iloc[test_indices]['conversation'].values
		test_y = y_values.iloc[test_indices]['category'].values.astype(str)
		NaiveBayesImplemented(train_text, train_y,test_text,test_y)
		
	finalRun(conversOnly.values,y_values,test_set.values,test_ids.values)
	
""" intializing prior probabilites of all categories"""
def initializePriors(prior_counts):
	total=sum(prior_counts)
	print total
	global prior_hockey,prior_movies,prior_nba,prior_news
	global prior_nfl,prior_politics,prior_soccer,prior_worldnews
	#prior_counts are in alphabetical order
	prior_hockey=prior_counts[0]/float(total)
	prior_movies=prior_counts[1]/float(total)
	prior_nba=prior_counts[2]/float(total)
	prior_news=prior_counts[3]/float(total)
	prior_nfl=prior_counts[4]/float(total)
	prior_politics=prior_counts[5]/float(total)
	prior_soccer=prior_counts[6]/float(total)
	prior_worldnews=prior_counts[7]/float(total)
	
def NaiveBayesImplemented(train_text, train_y,test_text,test_y):
	global documentClass
	count=0
	documentClass={}
	for con,y in zip(train_text,train_y):
		count=count+1
		print "processing conversation number",count,"\n"
		processConversation(con)
		addWordsInClassCategory(y)
	getTopWordsInEachCategory()
	predictResults(test_text,test_y)
	
def predictResults(test_text,test_y):
	global scores
	predictions=[]
	#get metrics
	initializeCounts()
	count=0
	for con,y in zip(test_text,test_y):
		count=count+1
		print "processing conversation number",count,"\n"
		processConversation(con)
		predictedValue=calculateBayesianProb()
		print predictedValue, y
		predictions.append(predictedValue)
	score = f1_score(test_y, predictions)
	print score
	accuracy=accuracy_score(test_y,predictions)
	print accuracy
	overall=classification_report(test_y,predictions)
	print overall
		
""" Calculating Bayesian conditional probabilites for ever word in a conversation"""
def calculateBayesianProb():
	global prob_hockey,prob_movies,prob_nba,prob_news
	global prob_nfl,prob_politics,prob_soccer,prob_worldnews
	global bag_of_words
	prob_hockey={}
	prob_movies={}
	prob_news={}
	prob_nba={}
	prob_nfl={}
	prob_politics={}
	prob_soccer={}
	prob_worldnews={}
	for word in bag_of_words:
		getHockeyProbability(word)
		getMoviesProbability(word)
		getNflProbability(word)
		getNbaProbability(word)
		getNewsProbability(word)
		getPoliticsProbability(word)
		getSoccerProbability(word)
		getWorldNewsProbability(word)
	return getMaxLikelihood()
	
""" Getting Maximum likelihoods,i.e. the final prediction."""
def getMaxLikelihood():
	global prob_hockey,prob_movies,prob_nba,prob_news
	global prob_nfl,prob_politics,prob_soccer,prob_worldnews
	global prior_hockey,prior_movies,prior_nba,prior_news
	global prior_nfl,prior_politics,prior_soccer,prior_worldnews
	likelihoods={}
	#prob_hockey = {k:np.log(v) for k, v in prob_hockey.items()}
	likelihoods['hockey']=prod(prob_hockey.values())*1000*prior_hockey
	#prob_movies = {k:np.log(v) for k, v in prob_movies.items()}
	likelihoods['movies']=prod(prob_movies.values())*1000*prior_movies
	#prob_nba = {k:np.log(v) for k, v in prob_nba.items()}
	likelihoods['nba']=prod(prob_nba.values())*1000*prior_nba
	#prob_news = {k:np.log(v) for k, v in prob_news.items()}
	likelihoods['news']=prod(prob_news.values())*1000*prior_news
	#prob_nfl = {k:np.log(v) for k, v in prob_nfl.items()}
	likelihoods['nfl']=prod(prob_nfl.values())*1000*prior_nfl
	#prob_politics = {k:np.log(v) for k, v in prob_politics.items()}
	likelihoods['politics']=prod(prob_politics.values())*1000*prior_politics
	#prob_soccer = {k:np.log(v) for k, v in prob_soccer.items()}
	likelihoods['soccer']=prod(prob_soccer.values())*1000*prior_soccer
	#prob_worldnews = {k:np.log(v) for k, v in prob_worldnews.items()}
	likelihoods['worldnews']=prod(prob_worldnews.values())*1000*prior_worldnews
	maxValue=max(likelihoods, key=lambda key: likelihoods[key])
	return maxValue
	
def prod( iterable ):
	p= 1
	for n in iterable:
		p *= n
	return p

"""initialze word counts per category and get the total number of unique words"""
def initializeCounts():
	all_words={}
	global documentClass,word_count_per_category,total_words
	for category,words in documentClass.items():
		number_of_words=len(words)
		all_words=merge_two_dicts(all_words,words)
		word_count_per_category[category]=number_of_words
	total_words=len(all_words)
	print "total words in vocabulary-",total_words

""" Given a word (target), calculate Bayesian probability w.r.t to hockey"""
def getHockeyProbability(target):
	global documentClass,total_words,prob_hockey
	words=dict(documentClass['hockey'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_hockey[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to movies"""
def getMoviesProbability(target):
	global documentClass,total_words,prob_movies
	words=dict(documentClass['movies'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_movies[target]=prob
		
""" Given a word (target), calculate Bayesian probability w.r.t to NFL"""
def getNflProbability(target):
	global documentClass,total_words,prob_nfl
	words=dict(documentClass['nfl'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=(occurrence+1)/float(len(words)+total_words)
	if prob>0:
		prob_nfl[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to NBA"""
def getNbaProbability(target):
	global documentClass,total_words,prob_nba
	words=dict(documentClass['nba'])
	if target in words:
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_nba[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to News"""
def getNewsProbability(target):
	global documentClass,total_words,prob_news
	words=dict(documentClass['news'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_news[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to politics"""
def getPoliticsProbability(target):
	global documentClass,total_words,prob_politics
	words=dict(documentClass['politics'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_politics[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to soccer"""
def getSoccerProbability(target):
	global documentClass,total_words,prob_soccer
	words=dict(documentClass['soccer'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_soccer[target]=prob

""" Given a word (target), calculate Bayesian probability w.r.t to world news"""
def getWorldNewsProbability(target):
	global documentClass,total_words,prob_worldnews
	words=dict(documentClass['worldnews'])
	if target in words.keys():
		occurrence=words[target]
	else:
		occurrence=0
	prob=((occurrence+1)/float(len(words)+total_words))
	if prob>0:
		prob_worldnews[target]=prob


def applyTFIDF():
	global documentClass,words_in_multiple_conv,totalConversations
	for classCategory, words in documentClass.items():
		for word,count in words.items():
			if word in words_in_multiple_conv:
				termFrequency=words_in_multiple_conv[word]
				idf=np.log(totalConversations/float(termFrequency+1))
				tfidf=count*idf
				words[word]=tfidf
				
def getTopWordsInEachCategory():
	global documentClass
	for classCategory, words in documentClass.items():
		sorted_words = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
		documentClass[classCategory]=sorted_words

if __name__ == "__main__":
	initProcessing()