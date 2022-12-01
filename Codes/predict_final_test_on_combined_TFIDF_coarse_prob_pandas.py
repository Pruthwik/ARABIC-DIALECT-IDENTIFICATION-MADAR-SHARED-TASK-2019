import pandas as pd
from sys import argv
# import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load


def readCSVFile(csvFilePath, typeOfFile=1):
    if typeOfFile:
        return pd.read_csv(csvFilePath, delimiter='\t', names=['Data', 'Label'])
    else:
        return pd.read_csv(csvFilePath, delimiter='\t', names=['Data'])


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


def main():
    csvFilePath = argv[1]
    classifier = loadObjectFromFile(argv[2])
    wordTfIdfVect = loadObjectFromFile(argv[3])
    charTfIdfVect = loadObjectFromFile(argv[4])
    predFile = argv[5]
    coarseProbClassifier = loadObjectFromFile(argv[6])
    trainVectWordCoarse = loadObjectFromFile(argv[7])
    trainVectCharCoarse = loadObjectFromFile(argv[8])
    typeOfFile = int(argv[9])
    allTestData = readCSVFile(csvFilePath, typeOfFile)
    testData = allTestData['Data']
    coarseWordTFIdf = createTestTfIdf(testData, trainVectWordCoarse)
    coarseCharTFIdf = createTestTfIdf(testData, trainVectCharCoarse)
    coarseCombined = hstack([coarseWordTFIdf, coarseCharTFIdf])
    coarseProbs = coarseProbClassifier.predict_proba(coarseCombined)
    coarseProbsMax = np.argmax(coarseProbs, axis=1)
    coarseProbsMax = coarseProbsMax.reshape(coarseProbsMax.shape[0], 1)
    coarseProbsMatrix = coo_matrix(coarseProbsMax)
    wordTestTfIdf = createTestTfIdf(
        testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(
        testData, charTfIdfVect)
    combinedTestTfIdf = hstack(
        [wordTestTfIdf, charTestTfIdf])
    combinedTestTfIdfWithCoarseProb = hstack([combinedTestTfIdf, coarseProbsMatrix])
    testPredictions = predictOnFeatures(classifier, combinedTestTfIdfWithCoarseProb)
    writeListToFile(predFile, testPredictions)
    np.save('dev-argmax-coarse-prob', coarseProbsMax)


if __name__ == '__main__':
    main()
