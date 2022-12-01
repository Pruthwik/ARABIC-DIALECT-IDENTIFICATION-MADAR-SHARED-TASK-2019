from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sys import argv
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


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


def separateDataAndLabels(lines):
    splitByTabs = [line.split('\t') for line in lines]
    return list(zip(* splitByTabs))


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB()
    mNB = fitTrainDataWithClassifier(mNB, trainData, trainLabels)
    return mNB


def main():
    trainFile = argv[1]
    trainLines = readLinesFromFile(trainFile)
    trainData, trainLabels = separateDataAndLabels(trainLines)
    classifier = argv[2]
    char_analyzer = 'char'
    char_ngram_range = (1, 3)
    word_analyzer = 'word'
    word_ngram_range = (1, 1)
    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, word_analyzer, word_ngram_range)
    charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, char_analyzer, char_ngram_range)
    combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf])
    # svm = SVMClassifier(combinedTrainTfIdf, trainLabels)
    mNB = multinomialNBClassifier(combinedTrainTfIdf, trainLabels)
    dumpObjectIntoFile('train-vect-combine-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + classifier + '.pkl', wordTfIdfVect)
    dumpObjectIntoFile('train-vect-combine-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-' + classifier + '.pkl', charTfIdfVect)
    dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '.pkl', mNB)


if __name__ == '__main__':
    main()
