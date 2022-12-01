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
    classifier1File = argv[2]
    classifier2File = argv[3]
    classifier1 = loadObjectFromFile(classifier1File)
    classifier2 = loadObjectFromFile(classifier2File)
    wordTfIdfVect = loadObjectFromFile(argv[4])
    charTfIdfVect = loadObjectFromFile(argv[5])
    predFile = argv[6]
    typeOfFile = int(argv[7])
    dialectFile = argv[8]
    testLMFile = argv[9]
    # coarseProbFile = argv[10]
    # trainLMIndexWiseProbs = loadObjectFromFile(argv[10])
    # trainCoarseWiseProbs = loadObjectFromFile(argv[11])
    # predFileWithLMIndex = argv[9]
    # predFileWithCoarseProb = argv[10]
    dialects = readLinesFromFile(dialectFile)
    # testLMScores = readCSVFile(testLMFile, dialects, 0)
    # testLMScores = readCSVFile(testLMFile, dialects, 0)
    # testLMScoresArray = testLMScores.values
    testLMScoresArray = np.load(testLMFile)
    # if re.search('multi-nb', classifierFile, re.I):
    # testLMScoresArray = applyPowerFunctionToArray(testLMScoresArray)
    testLMScoresArrayMax = np.argmax(testLMScoresArray, axis=1)
    testLMScoresArrayMax = testLMScoresArrayMax.reshape((testLMScoresArrayMax.shape[0], 1))
    # coarseProbIndices = np.load(coarseProbFile)
    # print(coarseProbIndices.shape)
    # testLMScoresMaxMatrix = coo_matrix(testLMScoresArrayMax)
    if typeOfFile == 1:
        columnNames = ['Data', 'Label']
        allTestData = readCSVFile(csvFilePath, columnNames, typeOfFile)
    else:
        columnNames = ['Data']
        allTestData = readCSVFile(csvFilePath, columnNames, typeOfFile)
    testData = allTestData['Data']
    # testCoarseProbs = np.load(coarseProbFile)
    # testCoarseProbsMax = np.argmax(testCoarseProbs, axis=1)
    # testCoarseProbIndices = testCoarseProbsMax.reshape((testCoarseProbsMax.shape[0], 1))
    # # testCoarseProbIndices = testCoarseProbsMax.reshape((testCoarseProbsMax.shape[0],)).tolist()
    # oneHotTestCoarseProbsMax = convertIntoOneHot(testCoarseProbIndices, len(dialects))
    # testLabels = allTestData['Label'].values
    # testLMIndices = testLMScoresArrayMax.reshape((testLMScoresArrayMax.shape[0],)).tolist()
    # testLMIndices = testLMScoresArrayMax.reshape((testLMScoresArrayMax.shape[0], 1))
    testLMSIndicesOneHot = convertIntoOneHot(testLMScoresArrayMax, len(dialects))
    # lmIndexWiseLabels = findConditionalProbForLM(testLMIndices, testLabels)
    # testCoarseProbIndices = coarseProbIndices.reshape((coarseProbIndices.shape[0],)).tolist()
    # coarseProbWiseLabels = findConditionalProbForLM(coarseProbIndices, testLabels)
    indexToDialectMap, dialectToIndexMap = createIndexItemMappings(dialects)
    wordTestTfIdf = createTestTfIdf(
        testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(
        testData, charTfIdfVect)
    # combinedTestTfIdf = hstack(
    #     [wordTestTfIdf, charTestTfIdf, testLMScoresMatrix])
    # combinedTestTfIdf = hstack(
    #     [wordTestTfIdf, charTestTfIdf, testLMScoresMaxMatrix])
    # combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf])
    # combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf, testLMIndices, testCoarseProbIndices])
    # combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf, testLMSIndicesOneHot, oneHotTestCoarseProbsMax])
    combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf, testLMSIndicesOneHot])
    print(combinedTestTfIdf.shape)
    # combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf, testLMSIndicesOneHot, oneHotTestCoarseProbsMax])
    # testPredictions = predictOnFeatures(classifier, combinedTestTfIdf)
    testPredictionProb1 = classifier1.predict_proba(combinedTestTfIdf)
    testPredictionProb2 = classifier2.predict_proba(combinedTestTfIdf)
    avgTestPredictions = np.argmax(0.2 * testPredictionProb1 + 0.8 * testPredictionProb2, axis=1)
    finalTestPredictions = list()
    for i in range(avgTestPredictions.shape[0]):
        finalTestPredictions.append(indexToDialectMap[avgTestPredictions[i]])
        # print(i, avgTestPredictions[i])
    # predictionsUsingLMIndex = list()
    # for index, lmIndex in enumerate(testLMIndices):
    #     lmIndexProbs = trainLMIndexWiseProbs[lmIndex]
    #     coarseIndexProbs = trainCoarseWiseProbs[testCoarseProbIndices[index]]
    #     lmIndexProbsArray = np.zeros((len(dialects),))
    #     coarseIndexProbsArray = np.zeros((len(dialects),))
    #     for label in lmIndexProbs:
    #         lmIndexProbsArray[dialectToIndexMap[label]] = lmIndexProbs[label]
    #     finalProduct = lmIndexProbsArray * testPredictionProb[index]
    #     for label in coarseIndexProbs:
    #         coarseIndexProbsArray[dialectToIndexMap[label]] = coarseIndexProbs[label]
    #     finalProduct *= coarseIndexProbsArray
    #     predictionsUsingLMIndex.append(indexToDialectMap[np.argmax(finalProduct)])
    # writeListToFile(predFile, testPredictions)
    writeListToFile(predFile, finalTestPredictions)
    # writeListToFile(predFile, predictionsUsingLMIndex)
    # writeListToFile(predFileWithLMIndex, predictionsUsingLMIndex)
    # writeListToFile(predFileWithCoarseProb, predictionsUsingLMIndex)


if __name__ == '__main__':
    main()
