import os
import json
import pickle
import tensorflow as tf
import numpy as np
from Config import *

class Database():

    sentencesFname = 'sentences'
    aspectLabelsFname = 'aspect_labels'
    opinionLabelsFname = 'opinion_labels'
    annotationSuffix = '_ann'
    cleanSuffix = '_clean'
    removedSuffix = '_removed'
    trainPrefix = 'train_'
    testPrefix = 'test_'
    parameters = 'parameters'
    dependency = 'dependency'
    weights = 'weights'
    combined = 'combined'
    tokenLableClass = 'tokenLabelClass'
    history = 'history'
    cache = 'cache'
    databaseSuffix = '_database'
    embedderSuffix = '_embedder'
    parserSuffix = '_parser'
    log = 'log'
    dataFolder = ''

    def __init__(self, homePath):
        self.homePath = homePath

        absHomePath = os.path.abspath(self.homePath)

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname) + '.txt'
        self.pTrainSentences = path
        
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname) + '.txt'
        self.pTestSentences = path

        self.pTrainAnnotations = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        self.pTestAnnotations = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        
        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.aspectLabelsFname) + '.txt'
        self.pTrainAspectLables = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.aspectLabelsFname) + '.txt'
        self.pTestAspectLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.opinionLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainOpinionLables = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.opinionLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTestOpinionLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTrainSentencesClean = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTestSentencesClean = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTrainSentencesRemoved = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTestSentencesRemoved = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.dependency) + '.txt'
        self.pTrainDependency = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.dependency) + '.txt'
        self.pTestDependency = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.combined) + '.txt'
        self.pTrainCombined = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.combined) + '.txt'
        self.pTestCombined = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.tokenLableClass) + '.txt'
        self.pTrainTokenLabelClass = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.tokenLableClass) + '.txt'
        self.pTestTokenLabelClass = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.history) + '.bin'
        self.pTrainHistory = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.history) + '.bin'
        self.pTestHistory = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.cache + self.databaseSuffix) + '.bin'
        self.pTrainCacheDatabase = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.cache + self.databaseSuffix) + '.bin'
        self.pTestCacheDatabase = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.cache + self.embedderSuffix) + '.bin'
        self.pTrainCacheEmbedder = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.cache + self.embedderSuffix) + '.bin'
        self.pTestCacheEmbedder = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.cache + self.parserSuffix) + '.bin'
        self.pTrainCacheParser = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.cache + self.parserSuffix) + '.bin'
        self.pTestCacheParser = path

        path = os.path.join( absHomePath, self.parameters) + '.txt'
        self.pParameters = path

        path = os.path.join( absHomePath, self.weights) + '.bin'
        self.pWeights = path

        path = os.path.join( absHomePath, self.log) + '.txt'
        self.pLog = path

        self.lbOther = 'Oth'; self.lbBeginAsp = 'bAS'; self.lbInsideAsp = 'iAS'
        self.lbBeginPosOpn = 'bO+'; self.lbInsidePosOpn = 'iO+'; self.lbBeginNegOpn = 'bO-'; self.lbInsideNegOpn = 'iO-'
        
        self.tokenLabelStringList = \
            [ self.lbOther, self.lbBeginAsp, self.lbInsideAsp, self.lbBeginPosOpn, self.lbInsidePosOpn, self.lbBeginNegOpn, self.lbInsideNegOpn ]

        self.trainSize = self.CreateFullDataset(trainNotTest = True, addLastPeriod = True, returnSizeOnly = True)
        self.testSize = self.CreateFullDataset(trainNotTest = False, addLastPeriod = True, returnSizeOnly = True)

        self.trainCache = [None] * self.trainSize
        self.testCache = [None] * self.testSize

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def OnStartEpoch(self, epoch):
        self.LoadCache()

    def OnEndEpoch(self, epoch):
        self.SaveCache()

    def RemoveFile(self, path):
        try:
            os.remove(path)
        except:
            print("Error while deleting file ", path)

    def RemoveCache(self):
        self.RemoveFile(self.pTrainCacheDatabase)
        self.RemoveFile(self.pTestCacheDatabase)

    def LoadCache(self):
        if self.Exists(self.pTrainCacheDatabase):
            self.trainCache = self.LoadBinaryData(self.pTrainCacheDatabase)
        if self.Exists(self.pTestCacheDatabase):
            self.testCache = self.LoadBinaryData(self.pTestCacheDatabase)

    def SaveCache(self):
        self.SaveBinaryData(self.trainCache, self.pTrainCacheDatabase)
        self.SaveBinaryData(self.testCache, self.pTestCacheDatabase)

    def GetListOfLines(self, path, addLastPeriod = False):
        try:
            file = open(path, 'rt')
            text = file.read()
            file.close()
        except:
            raise Exception("Couldn't open/read/close file: " + path)
        
        lines = text.split('\n')

        nlines = []
        if addLastPeriod == True:
            for line in lines:
                if line[-1] != '?' and line[-1] != '.' :
                    line = line + '.'
                nlines.append(line)

        if addLastPeriod == False:
            return lines
        else:
            return nlines

    def SaveJsonData(self, data, path):
        try:
            file = open(path, 'wt+')
            json.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

    def LoadJsonData(self, path):
        try:
            file = open(path, 'rt')
            data = json.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)
        
        return data

    def SaveBinaryData(self, data, path):
        try:
            file = open(path, 'wb+')
            pickle.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

    def LoadBinaryData(self, path):
        try:
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

        return data

    def Exists(self, path):
        return os.path.exists(path)

    def CreateTextLinesDataset(self, trainNotTest = True, addLastPeriod = False):
        if trainNotTest:
            pSentences = self.pTrainSentences
            pAspects = self.pTrainAspectLables
            pOpinions = self.pTrainOpinionLables
        else:
            pSentences = self.pTestSentences
            pAspects = self.pTestAspectLables
            pOpinions = self.pTestOpinionLables

        sentencesList = self.GetListOfLines(pSentences, addLastPeriod = addLastPeriod)
        aspectsList = self.GetListOfLines(pAspects, addLastPeriod = False )
        opinionsList = self.GetListOfLines(pOpinions, addLastPeriod = False)

        assert len(sentencesList) == len(aspectsList)
        assert len(aspectsList) == len(opinionsList)

        concatList = []; lineId = 0
        for sen, asp, opn in zip(sentencesList, aspectsList, opinionsList):
            concatList.append( ( sen, asp, opn, str(lineId) ) )
            lineId += 1
        dataset = tf.data.Dataset.from_tensor_slices(concatList)

        return dataset

    def CreateFullDataset(self, trainNotTest = True, addLastPeriod = False, returnSizeOnly = False):
        if trainNotTest:
            pSentences = self.pTrainSentences
            pAspects = self.pTrainAspectLables
            pOpinions = self.pTrainOpinionLables
            pLabels = self.pTrainTokenLabelClass
        else:
            pSentences = self.pTestSentences
            pAspects = self.pTestAspectLables
            pOpinions = self.pTestOpinionLables
            pLabels = self.pTestTokenLabelClass

        sentencesList = self.GetListOfLines(pSentences, addLastPeriod = addLastPeriod)
        aspectsList = self.GetListOfLines(pAspects, addLastPeriod = False )
        opinionsList = self.GetListOfLines(pOpinions, addLastPeriod = False)
        labelsList = self.LoadJsonData(pLabels)

        assert len(sentencesList) == len(aspectsList)
        assert len(aspectsList) == len(opinionsList)

        if returnSizeOnly: 
            return len(sentencesList)
        else:
            concatList = []; lineId = 0
            for sen, asp, opn, label in zip(sentencesList, aspectsList, opinionsList, labelsList):
                label = json.dumps(label) 
                concatList.append( ( sen, asp, opn, label, str(lineId) ) )
                lineId += 1
            dataset = tf.data.Dataset.from_tensor_slices(concatList)

            return dataset

    def DecodeTextLineDatasetRecord(self, dataset_record):
        sentence = dataset_record[0].numpy().decode()
        aspect = dataset_record[1].numpy().decode()
        opinion = dataset_record[2].numpy().decode()
        lineId = int(dataset_record[3].numpy())

        return sentence, aspect, opinion, lineId

    def DecodeDatasetRecord(self, dataset_record):
        sentence = dataset_record[0].numpy().decode()
        aspect = dataset_record[1].numpy().decode()
        opinion = dataset_record[2].numpy().decode()
        label = dataset_record[3].numpy().decode()
        labelClassList = json.loads(label)
        lableStringList = [self.tokenLabelStringList[labelClass] for labelClass in labelClassList]
        lineId = int(dataset_record[4].numpy())

        return sentence, aspect, opinion, labelClassList, lableStringList, lineId

    def CombineBasicFiles(self, trainNotTest = True, addLastPeriod = False, saveCombined = False):

        if trainNotTest:
            pSentences = self.pTrainSentences
            pAspects = self.pTrainAspectLables
            pOpinions = self.pTrainOpinionLables
            pCombined = self.pTrainCombined
        else:
            pSentences = self.pTestSentences
            pAspects = self.pTestAspectLables
            pOpinions = self.pTestOpinionLables
            pCombined = self.pTestCombined

        sentencesList = self.GetListOfLines(pSentences, addLastPeriod = addLastPeriod)
        aspectsList = self.GetListOfLines(pAspects, addLastPeriod = False )
        opinionsList = self.GetListOfLines(pOpinions, addLastPeriod = False)

        assert len(sentencesList) == len(aspectsList)
        assert len(aspectsList) == len(opinionsList)

        combinedList = []; lineId = 1
        
        if saveCombined: linefeed = '\n' 
        else: linefeed = '' 
        
        for sen, asp, opn in zip(sentencesList, aspectsList, opinionsList):
            combinedList.append(str(lineId) + linefeed)
            combinedList.append(sen + linefeed); combinedList.append(asp + linefeed); combinedList.append(opn + linefeed); combinedList.append('<>:' + linefeed)
            lineId += 1
        if saveCombined:
            combinedList[-1] = combinedList[-1][:-1] 
        if saveCombined:
            combinedFile = open(pCombined, 'wt+')
            combinedFile.writelines(combinedList)
            combinedFile.close()

        return combinedList

    def GetRefeinedLabels(self, sentence, aspect, opinion, cachetag, lineId):
        assert isinstance(cachetag, CacheTag)

        cache = None
        if cachetag == CacheTag.Train: cache = self.trainCache
        elif cachetag == CacheTag.Test: cache = self.testCache

        if cache is not None and cache[lineId] != None:
            consistent, aspectList, opinionList, wrongAspList, wrongOpnList = cache[lineId]

        else:
            wrongAspList = []; wrongOpnList = []

            aspectList = self.__GetRefinedAspectList__(aspect)
            opinionList = self.__GetRefinedOpinionList__(opinion)

            consistent = True
            sentence = sentence.lower()

            for asp in aspectList:
                if 0 > sentence.find(asp.lower()):
                    consistent = False
                    wrongAspList.append(asp)
            
            for opnText, opnScore in opinionList:
                if 0 > sentence.find(opnText.lower()):
                    consistent = False
                    wrongOpnList.append(opnText)

            if cache is not None and cache[lineId] == None:
                cache[lineId] = (consistent, aspectList, opinionList, wrongAspList, wrongOpnList)
        
        return consistent, aspectList, opinionList, wrongAspList, wrongOpnList

    def __GetRefinedAspectList__(self, aspect):
        aList = aspect.split(','); nList = []
        for aspect in aList:
            asp = aspect.lower().strip()

            if asp != 'nil':
                nList.append(asp)

        return nList

    def __GetRefinedOpinionList__(self, opinion):
        oList = opinion.split(','); nList = []
        for opinion in oList:
            opnStr = opinion.lower().strip()
            if opnStr != 'nil':
                if opnStr.find('+1') >= 0:
                    opnScore = 1
                elif opnStr.find('-1') >= 0:
                    opnScore = -1
                else: opnScore = -1 
                opnStr = opnStr.replace('+1', ''); opnStr = opnStr.replace('-1', ''); opnStr = opnStr.strip()            
                nList.append( (opnStr, opnScore) )
        
        return nList

    def RefineCombinedData(self, trainNotTest = True, addLastPeriod = False, fromCombined = False, saveCombined = False, regenerateBasicFiles = False):
        if trainNotTest: pCombined = self.pTrainCombined
        else: pCombined = self.pTestCombined

        if not fromCombined:
            comLines = self.CombineBasicFiles(trainNotTest, addLastPeriod, saveCombined = False)
        else:
            comLines = self.GetListOfLines(pCombined, addLastPeriod = False)

        lineCount = len(comLines); lineId = 0

        refinedLines = []
        while lineId < lineCount:
            sentenceId = comLines[lineId]; lineId += 1
            sentence = comLines[lineId]; lineId += 1; sentence = " ".join(sentence.split())
            aspect = comLines[lineId]; lineId += 1; aspect = " ".join(aspect.split())
            opinion = comLines[lineId]; lineId += 1; opinion = " ".join(opinion.split())
            check = comLines[lineId]; lineId += 1

            consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.GetRefeinedLabels(sentence, aspect, opinion, CacheTag.Real, lineId = -1)
            if consistent: check = ""
            else: check = "ERROR: asp: {}, opn: {}".format(wrongAspList, wrongOpnList)

            if saveCombined: linefeed = '\n'
            else: linefeed = '' 
        
            refinedLines.append(sentenceId + linefeed)
            refinedLines.append(sentence + linefeed)
            refinedLines.append(aspect + linefeed)
            refinedLines.append(opinion + linefeed)
            refinedLines.append(check + linefeed)

        if saveCombined: refinedLines[-1] = refinedLines[-1][:-1]

        if saveCombined:
            if trainNotTest: pCombined = self.pTrainCombined
            else: pCombined = self.pTestCombined

            combinedFile = open(pCombined, 'wt+')
            combinedFile.writelines(refinedLines)
            combinedFile.close()

        if regenerateBasicFiles:

            if trainNotTest:
                pSentences = self.pTrainSentences
                pAspects = self.pTrainAspectLables
                pOpinions = self.pTrainOpinionLables
            else:
                pSentences = self.pTestSentences
                pAspects = self.pTestAspectLables
                pOpinions = self.pTestOpinionLables
                pCombined = self.pTestCombined

            senList = []; aspList = []; opnList = []; linefeed = '\n'
            
            lineId = 0
            while lineId < lineCount:
                sentenceId = comLines[lineId]; lineId += 1
                sentence = comLines[lineId]; lineId += 1
                aspect = comLines[lineId]; lineId += 1
                opinion = comLines[lineId]; lineId += 1
                check = comLines[lineId]; lineId += 1

                senList.append(sentence + linefeed)
                aspList.append(aspect + linefeed)
                opnList.append(opinion + linefeed)
            
            senList[-1] = senList[-1][:-1]
            aspList[-1] = aspList[-1][:-1]
            opnList[-1] = opnList[-1][:-1]
            
            combinedFile = open(pSentences, 'wt+')
            combinedFile.writelines(senList)
            combinedFile.close()

            combinedFile = open(pAspects, 'wt+')
            combinedFile.writelines(aspList)
            combinedFile.close()

            combinedFile = open(pOpinions, 'wt+')
            combinedFile.writelines(opnList)
            combinedFile.close()

        return refinedLines

    def GetTokenLabelList(self, tokenList, aspList, opnList):
        nO = self.GetTokenLabelClass(self.lbOther) # for other
        tokenLabelStringList = [self.lbOther] * len(tokenList)
        tokenLabelClassList = [nO] * len(tokenList)

        for asp in aspList:
            aspTokenList = asp.split()
            if len(aspTokenList) <= 0: continue
            location = self.FindSeries( tokenList, aspTokenList )
            if location >= 0:
                BA = self.lbBeginAsp; IA = self.lbInsideAsp
                tokenLabelStringList[location] = BA
                tokenLabelClassList[location] = self.GetTokenLabelClass(BA)
                for inc in range( 1, len(aspTokenList) ):
                    tokenLabelStringList[location + inc] = IA
                    tokenLabelClassList[location + inc] = self.GetTokenLabelClass(IA)
        
        for opn in opnList:
            opnString, opnScore = opn
            opnTokenList = opnString.split()
            if len(opnTokenList) <= 0: continue
            location = self.FindSeries( tokenList, opnTokenList )
            if location >= 0:
                if opnScore > 0: BO = self.lbBeginPosOpn; IO = self.lbInsidePosOpn
                else: BO = self.lbBeginNegOpn; IO = self.lbInsideNegOpn
                tokenLabelStringList[location] = BO
                tokenLabelClassList[location] = self.GetTokenLabelClass(BO)
                for inc in range( 1, len(opnTokenList) ):
                    tokenLabelStringList[location + inc] = IO
                    tokenLabelClassList[location + inc] = self.GetTokenLabelClass(IO)

        return tokenLabelClassList, tokenLabelStringList

    def FindSeries(self, superSeries, subSeries):
        assert len(superSeries) > 0
        assert len(subSeries) > 0
        
        location = -1
        if len(subSeries) > len(superSeries):
            location = -1
        else:
            start = 0
            while len(superSeries) - start >= len(subSeries):
                if self.FindHeadSeries(superSeries[start:], subSeries):
                    location = start; break
                start += 1
        return location

    def FindHeadSeries(self, superSeries, subSeries):
        assert len(superSeries) > 0
        assert len(subSeries) > 0
        
        found = True
        if len(subSeries) > len(superSeries):
            found = False
        else:               
            for a, b in zip(superSeries, subSeries):
                if a.lower() != b.lower():
                    found = False
                    break
            return found

    def NLabelClass(self):
        return len(self.tokenLabelStringList)

    def GetTokenLabelString(self, numeral):
        return self.tokenLabelStringList[numeral]

    def GetTokenLabelClass(self, string):
        numeral = 0; count = len(self.tokenLabelStringList)
        while numeral < count and self.tokenLabelStringList[numeral] != string:
            numeral += 1
        if numeral >= count:
            raise Exception('Token label string not found: ', string)
        else:
            return numeral

    def GetHitRate(self, scoreHistory):
        hitRate = []
        for batchScore in scoreHistory:
            cntHit = 0; cntMiss = 0
            for sentenceScore in batchScore:
                for trueClass, predClass in sentenceScore:
                    if trueClass != 0:  
                        if trueClass == predClass: cntHit += 1
                        else: cntMiss += 1
            hitRate.append( 1.0 * cntHit / (cntHit + cntMiss + 1) )

        return hitRate

    def GetF1Score(self, scoreHistory) :
        classes = len(self.tokenLabelStringList)
        batches = len(scoreHistory)
        precision = np.zeros( (classes, batches) ); precision[:] = np.nan
        recall = np.zeros( (classes, batches) ); recall[:] = np.nan

        for batchId in range(batches) :
            batchScore = scoreHistory[batchId]

            relavant = [None] * classes; truePositive = [None] * classes; falsePositive = [None] * classes

            for sentenceScore in batchScore :
                for trueClass, predClass in sentenceScore: 
                    if relavant[trueClass] == None : relavant[trueClass] = 1
                    else: relavant[trueClass] += 1

                    if trueClass == predClass : 
                        if truePositive[predClass] == None : truePositive[predClass] = 1
                        else: truePositive[predClass] += 1
                    else: 
                        if falsePositive[predClass] == None : falsePositive[predClass] = 1
                        else: falsePositive[predClass] += 1

            for clsId in range(classes) :
                if truePositive[clsId] != None :
                    if falsePositive[clsId] != None :
                        precision[clsId, batchId] = 1.0 * truePositive[clsId] / (truePositive[clsId] + falsePositive[clsId])
                    if relavant[clsId] != None :
                        recall[clsId, batchId] = 1.0 * truePositive[clsId] / relavant[clsId]

        return precision, recall