from pickle import load
from sys import argv
from sklearn.metrics import classification_report


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
    tfIdfVect = loadObjectFromFile(argv[3])
    predFile = argv[4]
    testTfIdf = createTestTfIdf(testData, tfIdfVect)
    predLabels = predictOnFeatures(classifier, testTfIdf)
    writeListToFile(predLabels, predFile)


if __name__ == '__main__':
    main()
