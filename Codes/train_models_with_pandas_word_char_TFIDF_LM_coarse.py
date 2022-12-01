import pandas as pd
from sys import argv
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np


def readCSVFile(csvFilePath, columnNames, typeFile=1):
    if typeFile == 1:
        return pd.read_csv(csvFilePath, delimiter='\t', names=columnNames)
    else:
        return pd.read_csv(csvFilePath, delimiter='\t', names=columnNames)


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


def gradientDescent(trainData, trainLabels):
    sgd = SGDClassifier(loss='perceptron')
    sgd = fitTrainDataWithClassifier(sgd, trainData, trainLabels)
    return sgd


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB()
    mNB = fitTrainDataWithClassifier(mNB, trainData, trainLabels)
    return mNB


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def applyPowerFunctionToArray(logProbs):
    vfunc = np.vectorize(lambda t: 10 ** t)
    return vfunc(logProbs)


def main():
    csvFilePath = argv[1]
    columnNames = ['Data', 'Label']
    dataWithLabels = readCSVFile(csvFilePath, columnNames)
    trainData = dataWithLabels['Data']
    trainLabels = dataWithLabels['Label']
    classifier = argv[2]
    lmFile = argv[3]
    dialectFile = argv[4]
    coarseArgmax = np.load(argv[5])
    dialects = readLinesFromFile(dialectFile)
    lmScores = readCSVFile(lmFile, dialects, 0)
    lmScoresArray = lmScores.values
    char_analyzer = 'char'
    char_ngram_range = (2, 5)
    word_analyzer = 'word'
    word_ngram_range = (1, 1)
    # if re.search('multi-nb', classifier, re.I):
    lmScoresArray = applyPowerFunctionToArray(lmScoresArray)
    lmScoresArrayMax = np.argmax(lmScoresArray, axis=1)
    lmScoresArrayMax = lmScoresArrayMax.reshape((lmScoresArrayMax.shape[0], 1))
    print(lmScoresArray.shape, lmScoresArrayMax.shape)
    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, word_analyzer, word_ngram_range)
    charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, char_analyzer, char_ngram_range)
    # lmScoresMatrix = coo_matrix(lmScoresArray)
    lmScoresMaxMatrix = coo_matrix(lmScoresArrayMax)
    # combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf, lmScoresMatrix])
    combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf, lmScoresMaxMatrix, coarseArgmax])
    if re.search('svm', classifier, re.I):
        classifierToSelect = SVMClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('logistic', classifier, re.I):
        classifierToSelect = logisticClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('multi-nb', classifier, re.I):
        classifierToSelect = multinomialNBClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('sgd', classifier, re.I):
        classifierToSelect = gradientDescent(combinedTrainTfIdf, trainLabels)
    dumpObjectIntoFile('train-vect-pandas-with-combined-LM-max-coarse-max-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + classifier + '.pkl', wordTfIdfVect)
    dumpObjectIntoFile('train-vect-pandas-with-combined-LM-max-coarse-max-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-' + classifier + '.pkl', charTfIdfVect)
    dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-pandas-with-combined-LM-max-coarse-max.pkl', classifierToSelect)


if __name__ == '__main__':
    main()
