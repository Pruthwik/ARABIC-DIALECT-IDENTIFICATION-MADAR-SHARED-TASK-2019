import pandas as pd
from sys import argv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier


def readCSVFile(csvFilePath):
    return pd.read_csv(csvFilePath, delimiter='\t', names=['Data', 'Label'])


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


def gradientBoost(trainData, trainLabels):
    gradBoost = GradientBoostingClassifier()
    gradBoost = fitTrainDataWithClassifier(gradBoost, trainData, trainLabels)
    return gradBoost


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB()
    mNB = fitTrainDataWithClassifier(mNB, trainData, trainLabels)
    return mNB


def main():
    csvFilePath = argv[1]
    dataWithLabels = readCSVFile(csvFilePath)
    trainData = dataWithLabels['Data']
    trainLabels = dataWithLabels['Label']
    # print(trainData.shape, trainLabels.shape)
    classifier = argv[2]
    # char_analyzer = 'char'
    # char_ngram_range = (2, 5)
    word_analyzer = 'word'
    word_ngram_range = (1, 2)
    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, word_analyzer, word_ngram_range)
    # charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(
    #     trainData, char_analyzer, char_ngram_range)
    # combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf])
    combinedTrainTfIdf = wordTrainTFIdf
    if re.search('svm', classifier, re.I):
        classifierToSelect = SVMClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('logistic', classifier, re.I):
        classifierToSelect = logisticClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('multi-nb', classifier, re.I):
        classifierToSelect = multinomialNBClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('sgd', classifier, re.I):
        classifierToSelect = gradientDescent(combinedTrainTfIdf, trainLabels)
    elif re.search('gradient-boosting', classifier, re.I):
        classifierToSelect = gradientBoost(combinedTrainTfIdf, trainLabels)
    dumpObjectIntoFile('train-vect-pandas-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + classifier + '.pkl', wordTfIdfVect)
    dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-pandas.pkl', classifierToSelect)
    # dumpObjectIntoFile('train-vect-pandas-' + char_analyzer + '-' +
    #                    '-'.join(map(str, char_ngram_range)) + '-' + classifier + '.pkl', charTfIdfVect)
    # dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
    #                    '-'.join(map(str, word_ngram_range)) + '-' + char_analyzer + '-' +
    #                    '-'.join(map(str, char_ngram_range)) + '-pandas.pkl', classifierToSelect)


if __name__ == '__main__':
    main()

