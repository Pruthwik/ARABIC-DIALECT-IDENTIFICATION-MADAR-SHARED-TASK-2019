import pandas as pd
from sys import argv
# import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load


def readCSVFile(csvFilePath, columnNames, typeOfFile=1):
    if typeOfFile:
        return pd.read_csv(csvFilePath, delimiter='\t', names=columnNames)
    else:
        return pd.read_csv(csvFilePath, delimiter='\t', names=columnNames)


def createTFIDFVectorsFromTrainData(trainData, analyzer='word', ngram_range=(1, 1)):
    tfIDFVect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    trainTFIdf = tfIDFVect.fit_transform(trainData)
    return trainTFIdf, tfIDFVect


def createTestTfIdf(testData, tfIdfVect):
    return tfIdfVect.transform(testData)


def loadObjectFromFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


def predictOnFeatures(classifier, features):
    return classifier.predict(features)


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def applyPowerFunctionToArray(logProbs):
    vfunc = np.vectorize(lambda t: 10 ** t)
    return vfunc(logProbs)


def findConditionalProbForLM(lmIndices, trainLabels):
    indexWiseLabelCount = dict()
    for index, lmIndex in enumerate(lmIndices):
        if lmIndex not in indexWiseLabelCount:
            indexWiseLabelCount[lmIndex] = {trainLabels[index]: 1}
        else:
            indexWiseLabelCount[lmIndex][trainLabels[index]] = indexWiseLabelCount[lmIndex].get(
                trainLabels[index], 0) + 1
    indexWiseLabelProb = dict()
    for index in indexWiseLabelCount:
        tempDict = dict()
        for label in indexWiseLabelCount[index]:
            tempDict[label] = indexWiseLabelCount[index][label] / sum(indexWiseLabelCount[index].values())
        indexWiseLabelProb[index] = tempDict
    return indexWiseLabelProb


def createIndexItemMappings(items):
    indexToItemDict = dict(enumerate(items))
    itemToIndexDict = {val: key for key, val in indexToItemDict.items()}
    return indexToItemDict, itemToIndexDict


def convertIntoOneHot(indexArray, totalCols):
    rows, columns, scores = list(), list(), list()
    for i in range(indexArray.shape[0]):
        rows.append(i)
        columns.append(indexArray[i][0])
        scores.append(1.)
    return coo_matrix((scores, (rows, columns)), shape=(indexArray.shape[0], totalCols))


def main():
    csvFilePath = argv[1]
    classifierFile = argv[2]
    classifier = loadObjectFromFile(classifierFile)
    wordTfIdfVect = loadObjectFromFile(argv[3])
    charTfIdfVect = loadObjectFromFile(argv[4])
    predFile = argv[5]
    typeOfFile = int(argv[6])
    dialectFile = argv[7]
    testLMFile = argv[8]
    coarseProbFile = argv[9]
    dialects = readLinesFromFile(dialectFile)
    testLMScores = readCSVFile(testLMFile, dialects, 0)
    testLMScoresArray = testLMScores.values
    testLMScoresArray = applyPowerFunctionToArray(testLMScoresArray)
    testLMScoresArrayMax = np.argmax(testLMScoresArray, axis=1)
    testLMScoresArrayMax = testLMScoresArrayMax.reshape((testLMScoresArrayMax.shape[0], 1))
    if typeOfFile == 1:
        columnNames = ['Data', 'Label']
        allTestData = readCSVFile(csvFilePath, columnNames, typeOfFile)
    else:
        columnNames = ['Data']
        allTestData = readCSVFile(csvFilePath, columnNames, typeOfFile)
    testData = allTestData['Data']
    testCoarseProbs = np.load(coarseProbFile)
    testCoarseProbsMax = np.argmax(testCoarseProbs, axis=1)
    testCoarseProbsMax = testCoarseProbsMax.reshape((testCoarseProbsMax.shape[0], 1))
    oneHotTestCoarseProbsMax = convertIntoOneHot(testCoarseProbsMax, len(dialects))
    testLMSIndicesOneHot = convertIntoOneHot(testLMScoresArrayMax, len(dialects))
    wordTestTfIdf = createTestTfIdf(
        testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(
        testData, charTfIdfVect)
    combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf, testLMSIndicesOneHot, oneHotTestCoarseProbsMax])
    testPredictions = predictOnFeatures(classifier, combinedTestTfIdf)
    writeListToFile(predFile, testPredictions)


if __name__ == '__main__':
    main()
