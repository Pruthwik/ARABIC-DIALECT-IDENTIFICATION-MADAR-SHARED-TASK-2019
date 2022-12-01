import pandas as pd
from sys import argv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
import re
from sklearn.naive_bayes import MultinomialNB
from pickle import load
from scipy.sparse import coo_matrix
import numpy as np


def readCSVFile(csvFilePath):
    return pd.read_csv(csvFilePath, delimiter='\t', names=['Data', 'Label'])


def createTestTfIdf(testData, tfIdfVect):
    return tfIdfVect.transform(testData)


def createTFIDFVectorsFromTrainData(trainData, analyzer='word', ngram_range=(1, 1)):
    tfIDFVect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    trainTFIdf = tfIDFVect.fit_transform(trainData)
    return trainTFIdf, tfIDFVect


def dumpObjectIntoFile(filePath, dataObject):
    with open(filePath, 'wb') as fileDump:
        dump(dataObject, fileDump)


def fitTrainDataWithClassifier(clf, trainData, trainLabels):
    clf.fit(trainData, trainLabels)
    return clf


def SVMClassifier(trainData, trainLabels):
    svm = LinearSVC()
    svm = fitTrainDataWithClassifier(svm, trainData, trainLabels)
    return svm


def logisticClassifier(trainData, trainLabels):
    logit = LogisticRegression()
    logit = fitTrainDataWithClassifier(logit, trainData, trainLabels)
    return logit


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB()
    mNB = fitTrainDataWithClassifier(mNB, trainData, trainLabels)
    return mNB


def loadObjectFromFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def main():
    csvFilePath = argv[1]
    dataWithLabels = readCSVFile(csvFilePath)
    trainData = dataWithLabels['Data']
    trainLabels = dataWithLabels['Label']
    # print(trainData.shape, trainLabels.shape)
    classifier = argv[2]
    coarseClassifier = loadObjectFromFile(argv[3])
    trainVectWordCoarse = loadObjectFromFile(argv[4])
    trainVectCharCoarse = loadObjectFromFile(argv[5])
    char_analyzer = 'char'
    char_ngram_range = (2, 5)
    word_analyzer = 'word'
    word_ngram_range = (1, 1)
    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, word_analyzer, word_ngram_range)
    charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, char_analyzer, char_ngram_range)
    combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf])
    coarseWordTFIdf = createTestTfIdf(trainData, trainVectWordCoarse)
    coarseCharTFIdf = createTestTfIdf(trainData, trainVectCharCoarse)
    coarseCombined = hstack([coarseWordTFIdf, coarseCharTFIdf])
    coarseProbs = coarseClassifier.predict_proba(coarseCombined)
    maxCoarseProbs = np.argmax(coarseProbs, axis=1)
    maxCoarseProbs = maxCoarseProbs.reshape(maxCoarseProbs.shape[0], 1)
    coarseProbsMatrix = coo_matrix(maxCoarseProbs)
    combinedTrainTfIdfWithCoarseProb = hstack([combinedTrainTfIdf, coarseProbsMatrix])
    if re.search('logistic', classifier, re.I):
        classifierToSelect = logisticClassifier(combinedTrainTfIdfWithCoarseProb, trainLabels)
    elif re.search('multi-nb', classifier, re.I):
        classifierToSelect = multinomialNBClassifier(combinedTrainTfIdfWithCoarseProb, trainLabels)
    elif re.search('svm', classifier, re.I):
        classifierToSelect = SVMClassifier(combinedTrainTfIdfWithCoarseProb, trainLabels)
    dumpObjectIntoFile('train-vect-with-argmax-coarse-pandas-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + classifier + '.pkl', wordTfIdfVect)
    dumpObjectIntoFile('train-vect-with-argmax-coarse-pandas-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-' + classifier + '.pkl', charTfIdfVect)
    dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-with-argmax-coarse-pandas.pkl', classifierToSelect)
    np.save('train-argmax-coarse-prob', maxCoarseProbs)


if __name__ == '__main__':
    main()
