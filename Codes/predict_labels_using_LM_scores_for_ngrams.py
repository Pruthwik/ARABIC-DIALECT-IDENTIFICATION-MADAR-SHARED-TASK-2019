import kenlm
# import pandas as pd
from sys import argv
# from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
# import re
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
    # print(sentence, len(tokens), totalNgrams)
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
    # toAdd = list()
    # for nGrmSample in finalNGrams:
    #     sampleSplit = nGrmSample.split()
    #     for i in range(-2, - len(sampleSplit) - 1, -1):
    #         itemToAdd = ' '.join(sampleSplit[: i + 1])
    #         if itemToAdd and itemToAdd != '<bos>':
    #             toAdd.append(itemToAdd)
    # finalNGrams += toAdd
    return finalNGrams


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def loadObjectFromFile(pickleFile):
    with open(pickleFile, 'rb') as loadFile:
        return load(loadFile)


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


def predictOnFeatures(classifier, features):
    return classifier.predict(features)


def main():
    inputFilePath = argv[1]
    pickleFilePath = argv[2]
    char5GramToIndex = loadObjectFromFile(pickleFilePath)
    inputLines = readLinesFromFile(inputFilePath)
    data = inputLines
    classifier = loadObjectFromFile(argv[3])
    trainedLMFile = argv[4]
    predFile = argv[5]
    trainedLM = kenlm.Model(trainedLMFile)
    rows, columns, scores = list(), list(), list()
    oovs = 0
    for index, sample in enumerate(data):
        print(index)
        lmScores = list(trainedLM.full_scores(sample))
        validNGrams = findValidNGramsMatchingWithLM(sample, lmScores, 5)
        assert len(validNGrams) == len(lmScores)
        for indexNgram, ele in enumerate(validNGrams):
            rows.append(index)
            if ele in char5GramToIndex:
                columns.append(char5GramToIndex[ele])
                scores.append(10 ** lmScores[indexNgram][0])
            else:
                print(ele, lmScores[indexNgram])
                lmScore = list(trainedLM.full_scores(ele, eos=False, bos=False))[-1]
                scores.append(10 ** lmScore[0])
                columns.append(len(char5GramToIndex))
                oovs += 1
    print(len(columns), len(rows), len(scores), oovs)
    lmScoresMatrix = csr_matrix((scores, (rows, columns)), (len(inputLines), len(char5GramToIndex) + 1))
    predictions = predictOnFeatures(classifier, lmScoresMatrix)
    writeListToFile(predFile, predictions)


if __name__ == '__main__':
    main()
