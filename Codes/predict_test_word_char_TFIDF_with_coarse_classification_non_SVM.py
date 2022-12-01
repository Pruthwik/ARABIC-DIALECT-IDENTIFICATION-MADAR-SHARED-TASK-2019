from pickle import load
from sys import argv
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
import numpy as np


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def createTestTfIdf(testData, tfIdfVect):
    return tfIdfVect.transform(testData)


def loadObjectFromFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def predictOnFeatures(classifier, features):
    return classifier.predict(features)


def separateDataAndLabels(lines):
    splitByTabs = [line.split('\t')[0] for line in lines]
    return splitByTabs
    # return list(zip(* splitByTabs))


def writeListToFile(dataList, filePath):
    with open(filePath, 'w') as fileWrite:
        fileWrite.write('\n'.join(dataList) + '\n')


def convertToOneHotCoarse(predictedCoarse, labelToIndexDict):
    totalLabels = len(labelToIndexDict)
    totalSamples = len(predictedCoarse)
    oneHotRepr = np.zeros((totalSamples, totalLabels))
    rows, columns = list(), list()
    for index, sample in enumerate(predictedCoarse):
        rows.append(index)
        columns.append(labelToIndexDict[sample])
    oneHotRepr[rows, columns] = 1
    return oneHotRepr


def main():
    testFile = argv[1]
    testLines = readLinesFromFile(testFile)
    testData = separateDataAndLabels(testLines)
    classifier = loadObjectFromFile(argv[2])
    wordTfIdfVect = loadObjectFromFile(argv[3])
    charTfIdfVect = loadObjectFromFile(argv[4])
    coarseModel = loadObjectFromFile(argv[5])
    coarseWordTfIdfVect = loadObjectFromFile(argv[6])
    coarseCharTfIdfVect = loadObjectFromFile(argv[7])
    coarseWordTfIdf = createTestTfIdf(testData, coarseWordTfIdfVect)
    coarseCharTfIdf = createTestTfIdf(testData, coarseCharTfIdfVect)
    combinedCoarseTfIdf = hstack([coarseWordTfIdf, coarseCharTfIdf])
    # coarseLabelToIndex = loadObjectFromFile(argv[8])
    predCoarseProb = coarseModel.predict_proba(combinedCoarseTfIdf)
    predProbVector = coo_matrix(predCoarseProb)
    predFile = argv[8]
    wordTestTfIdf = createTestTfIdf(testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(testData, charTfIdfVect)
    combinedTestTFIdf = hstack([wordTestTfIdf, charTestTfIdf])
    combinedWithCoarse = hstack([combinedTestTFIdf, predProbVector])
    predLabels = predictOnFeatures(classifier, combinedWithCoarse)
    writeListToFile(predLabels, predFile)


if __name__ == '__main__':
    main()
