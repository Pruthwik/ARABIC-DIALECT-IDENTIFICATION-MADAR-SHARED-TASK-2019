from pickle import load
from sys import argv
from scipy.sparse import hstack


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
    splitByTabs = [line.split('\t') for line in lines]
    return list(zip(* splitByTabs))


def writeListToFile(dataList, filePath):
    with open(filePath, 'w') as fileWrite:
        fileWrite.write('\n'.join(dataList) + '\n')


def main():
    testFile = argv[1]
    testLines = readLinesFromFile(testFile)
    testData, testLabels = separateDataAndLabels(testLines)
    classifier = loadObjectFromFile(argv[2])
    wordTfIdfVect = loadObjectFromFile(argv[3])
    charTfIdfVect = loadObjectFromFile(argv[4])
    predFile = argv[5]
    wordTestTfIdf = createTestTfIdf(testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(testData, charTfIdfVect)
    combinedTestTFIdf = hstack([wordTestTfIdf, charTestTfIdf])
    predLabels = predictOnFeatures(classifier, combinedTestTFIdf)
    writeListToFile(predLabels, predFile)


if __name__ == '__main__':
    main()
