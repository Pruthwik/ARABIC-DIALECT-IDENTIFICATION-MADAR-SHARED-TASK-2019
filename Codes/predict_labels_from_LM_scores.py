import numpy as np
from sys import argv
import pandas as pd


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def createIndexItemMappings(items):
    indexToItemDict = dict(enumerate(items))
    itemToIndexDict = {val: key for key, val in indexToItemDict.items()}
    return indexToItemDict, itemToIndexDict


def main():
    lmFile = argv[1]
    dialectsFile = argv[2]
    predFile = argv[3]
    typeFile = int(argv[4])
    dialects = readLinesFromFile(dialectsFile)
    if typeFile:
        lmScores = np.load(lmFile)
    else:
        lmScoresString = readLinesFromFile(lmFile)
        lmScoresAsFloat = [list(map(lambda x: float(x), line.split())) for line in lmScoresString]
        lmScores = np.array(lmScoresAsFloat)
        print(type(lmScores), lmScores[1])
    indexToDialectMap, dialectToIndexMap = createIndexItemMappings(dialects)
    maxIndexes = np.argmax(lmScores, axis=1)
    finalPredictions = list()
    for i in range(maxIndexes.shape[0]):
        finalPredictions.append(indexToDialectMap[maxIndexes[i]])
    writeListToFile(predFile, finalPredictions)


if __name__ == '__main__':
    main()
