import kenlm
# import pandas as pd
from sys import argv
# from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import csr_matrix
# import numpy as np
from pickle import load


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


def findValidNGramsInSentence(sentence, nGram=5):
    tokens = sentence.split()
    totalTokens = len(tokens)
    totalNgrams = list()
    for j in range(totalTokens - nGram + 1):
        totalNgrams.append(' '.join(tokens[j: j + nGram]))
    addedToStart, prefixString = list(), '<bos>'
    for i in range(nGram - 1):
        if i < len(tokens):
            prefixString += ' ' + tokens[i]
            addedToStart.append(prefixString)
    if len(tokens) >= nGram:
        totalNgrams = addedToStart + totalNgrams + [' '.join(tokens[-nGram + 1:]) + ' <eos>']
    else:
        totalNgrams = addedToStart + [' '.join(tokens) + ' <eos>']
    return totalNgrams


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def loadObjectFromFile(pickleFile):
    with open(pickleFile, 'rb') as loadFile:
        return load(loadFile)


def splitDataAndLabels(lines):
    return zip(* ([line.split('\t') for line in lines]))


def findValidNGramsMatchingWithLM(sentence, lmScores, nGram=5):
    totalNGrams = findValidNGramsInSentence(sentence, nGram)
    assert len(lmScores) == len(totalNGrams)
    finalNGrams = list()
    for indexNgram, ele in enumerate(totalNGrams):
        if lmScores[indexNgram][1] == 5:
            finalNGrams.append(ele)
        else:
            nGramSplit = ele.split()
            finalNGrams.append(' '.join(nGramSplit[: lmScores[indexNgram][1]]))
    return finalNGrams


def main():
    inputFilePath = argv[1]
    labelsFile = argv[2]
    pickleFilePath = argv[3]
    char5GramToIndex = loadObjectFromFile(pickleFilePath)
    print(len(char5GramToIndex))
    inputLines = readLinesFromFile(inputFilePath)
    data = inputLines
    classifier = argv[4]
    trainedLMFile = argv[5]
    trainedLM = kenlm.Model(trainedLMFile)
    trainLabels = readLinesFromFile(labelsFile)
    rows, columns, scores = list(), list(), list()
    for index, sample in enumerate(data):
        lmScores = list(trainedLM.full_scores(sample))
        validNGrams = findValidNGramsMatchingWithLM(sample, lmScores, 5)
        assert len(validNGrams) == len(lmScores)
        for indexNgram, nGram in enumerate(validNGrams):
            rows.append(index)
            columns.append(char5GramToIndex[nGram])
            scores.append(10 ** lmScores[indexNgram][0])
    lmScoresMatrix = csr_matrix((scores, (rows, columns)), (len(inputLines), len(char5GramToIndex) + 1))
    if re.search('svm', classifier, re.I):
        classifierToSelect = SVMClassifier(lmScoresMatrix, trainLabels)
    elif re.search('logistic', classifier, re.I):
        classifierToSelect = logisticClassifier(lmScoresMatrix, trainLabels)
    elif re.search('multi-nb', classifier, re.I):
        classifierToSelect = multinomialNBClassifier(lmScoresMatrix, trainLabels)
    elif re.search('sgd', classifier, re.I):
        classifierToSelect = gradientDescent(lmScoresMatrix, trainLabels)
    elif re.search('gradient-boosting', classifier, re.I):
        classifierToSelect = gradientBoost(lmScoresMatrix, trainLabels)
    dumpObjectIntoFile(classifier + '-with-LM-scores-of-ngrams.pkl', classifierToSelect)


if __name__ == '__main__':
    main()
