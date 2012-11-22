# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:05:54 2012

@author: Kunj Karia
"""
import sys
import re
import nltk

#start processing the tweets
def processTweet(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s+])', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

#Read the tweets and process it
file = open('sampletweets.txt', 'r')
line = file.readline()

while line:
    processedTweet = processTweet(line)
    line = file.readline()
    
file.close()
#end

stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#Get StopWord List
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    file = open('stopWords.txt', 'r')
    line = file.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = file.readline()
    file.close()
    return stopWords
#end

#Get Feature Vector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

#Read the tweets one by one and process it
file = open('sampletweets.txt', 'r')
line = file.readline()

st = open('stopWords.txt', 'r')
stopWords = getStopWordList('stopWords.txt')

#create featureList and featureVector
featureList = []

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    featureList = featureList + featureVector
    line = file.readline()
#end loop
file.close()

#list of 'positive/negative' label on tweet/text from training data
sentiment = [] 

#Extracts 'positive/negative' sentiment from training data and stores in sentiment list
def ExtractSentiment(pattern, text):
    match = re.search(pattern, text)
    if match:
        sentiment.append(match.group())
#end

file = open('SampleLabelledTweets.txt','r')
line = file.readlines()
for lines in line:    
    ExtractSentiment('negative', lines)
    ExtractSentiment('positive', lines)

#list of tweet/text from training data
listOfTweet = []
        
#Extracts tweet from training data and stores in tweet list
def ExtractTweet(pattern, text):
    match = re.search(pattern, text)
    if match:
        listOfTweet.append(match.group())
#end

for lines in line:
    ExtractTweet('\s+.*', lines)

#"Tweets" Variable
tweets = []

for i in range(1000):
    sentiment1 = sentiment[i]
    tweet1 = listOfTweet[i]
    processedTweet = processTweet(tweet1)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment1));

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

# Extract feature vector for all tweets in one shot
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# In terminal type python tsa.py hello world
testTweet = str((' ').join(sys.argv[1:]))
processedTestTweet = processTweet(testTweet)
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
