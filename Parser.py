from stanfordnlp.server import CoreNLPClient

from Database import Database
from Config import *

class Parser():

    sentencesFname = 'sentences'
    aspectLabelsFname = 'aspect_labels'
    opinionLabelsFname = 'opinion_labels'
    annotationSuffix = '_ann'
    trainPrefix = 'train_'
    testPrefix = 'test_'
    dataFolder = 'dataset'

    dependencyOfChoice = 'basicDependencies' #'enhancedPlusPlusDependencies'

    def __init__(self, database, dummyMode = False, testMode = False):
        assert isinstance(database, Database)
        self.database = database

        if dummyMode:
            self.dummyMode = True
            return
        else:
            self.dummyMode = False

        self.client = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','depparse','coref'],\
             timeout=30000, memory='4G', output_format='json') #----------------------------------------------- output_format='json'
        
        self.trainAnnotation = self.LoadGenerateAnnotation( trainNotTest = True )
        self.testAnnotation = self.LoadGenerateAnnotation( trainNotTest = False )
        self.parameters = self.LoadParameterDictionary( self.database.pParameters )

        self.trainCache = [{} for _ in range(self.database.trainSize)] 
        self.testCache = [{} for _ in range(self.database.testSize)]

        self.testMode = testMode

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def OnStartEpoch(self, epoch):
        self.LoadCache()

    def OnEndEpoch(self, epoch):
        self.SaveCache()

    def RemoveCache(self):
        self.database.RemoveFile(self.database.pTrainCacheParser)
        self.database.RemoveFile(self.database.pTestCacheParser)

    def LoadCache(self):
        if self.database.Exists(self.database.pTrainCacheParser):
            self.trainCache = self.database.LoadBinaryData(self.database.pTrainCacheParser)
        if self.database.Exists(self.database.pTestCacheParser):
            self.testCache = self.database.LoadBinaryData(self.database.pTestCacheParser)

    def SaveCache(self):
        self.database.SaveBinaryData(self.trainCache, self.database.pTrainCacheParser)
        self.database.SaveBinaryData(self.testCache, self.database.pTestCacheParser)

    def GenerateAnnotation(self, sentence, noramlized, cacheTag, lineId):
        cache = None
        if cacheTag == CacheTag.Train: cache = self.trainCache
        elif cacheTag == CacheTag.Test: cache = self.testCache

        if noramlized : key = 'NAN'
        else: key = 'AN'

        if cache is not None and cache[lineId].get(key, None) != None:
            annotation = cache[lineId][key]
        else:
            annotation = self.client.annotate(sentence)
            if cache is not None and cache[lineId].get(key, None) == None:
                cache[lineId][key] = annotation
    
        return annotation

    def NormalizeSentence(self, line, cacheTag, lineId):
        cache = None
        if cacheTag == CacheTag.Train: cache = self.trainCache
        elif cacheTag == CacheTag.Test: cache = self.testCache

        if cache is not None and cache[lineId].get('NR', None) != None:
            sen = cache[lineId]['NR']
            if sen.find("n't") >= 0:
                sen = sen 
        else:
            annotationSen = self.GenerateAnnotation(line, False, cacheTag, lineId)

            words = []
            tokenMapList = self.GetTokenMapList(annotationSen)
            for tmap in tokenMapList:
                w_org = tmap['originalText']
                w_word = tmap['word']
                w_lemma = tmap['lemma']

                if w_org.lower() == "n't" and w_lemma == "not":               
                    token = "not"

                    if len(words) > 0:
                        if words[-1].lower() == 'ca':   # for "can't"
                            if words[-1][0] == 'C':
                                words[-1] = 'Can'
                            else:
                                words[-1] = 'can'
                        elif words[-1].lower() == 'wo': # for "won't"
                            if words[-1][0] == 'W':
                                words[-1] = 'Will'
                            else:
                                words[-1] = 'will'
                        elif words[-1].lower() == 'sha': 
                            if words[-1][0] == 'S':
                                words[-1] = 'Shall'
                            else:
                                words[-1] = 'shall'
                else:
                    if (tmap['originalText']).isalnum(): 
                        token = tmap['originalText']
                    else:
                        token = tmap['lemma']

                words.append(token)

            sen = ''
            for word in words:
                if len(sen) > 0 and not word[0:1].isalnum():
                    sen = sen[:-1] 
                sen += (word + ' ')

            sen = sen[:-1] 
            sen += '\n' 

            if cache is not None and cache[lineId].get('NR', None) == None:
                cache[lineId]['NR'] = sen

        if sen.find("n't") >= 0:
            sen = sen 

        return sen

    def LoadParameterDictionary(self, path): 
        parameters = {}

        if self.database.Exists(path):
            parameters = self.database.LoadJsonData(path)
        else:
            trainDependencies = set()
            for ann in self.trainAnnotation:
                for b in ann['sentences'][0][self.dependencyOfChoice]:
                    trainDependencies.add(b['dep'])

            testDependencies = set()
            for ann in self.testAnnotation:
                for b in ann['sentences'][0][self.dependencyOfChoice]:
                    testDependencies.add(b['dep'])
            
            parameters['trainDependenciesList'] = sorted(list(trainDependencies))
            parameters['testDependenciesList'] = sorted(list(testDependencies))
            parameters['dependenciesList'] = sorted(list(trainDependencies | testDependencies))

            self.database.SaveJsonData(parameters, path)

        return parameters

    def LoadGenerateAnnotation(self, trainNotTest = True):
        if trainNotTest:
            pathToAnn = self.database.pTrainAnnotations
            pathToSen = self.database.pTrainSentences
        else:
            pathToAnn = self.database.pTestAnnotations
            pathToSen = self.database.pTestSentences

        if self.database.Exists(pathToAnn):
            annotations = self.database.LoadJsonData(pathToAnn)
        else:
            print("Producing annotaton file " + pathToAnn + " from " + pathToSen)
            self.GenerateAnnotations(sourcePath = pathToSen, destPath = pathToAnn)
            annotations = self.database.LoadJsonData(pathToAnn)
        return annotations

    def GenerateAnnotations(self, sourcePath, destPath):
        print('Producing annotations of file: ' + sourcePath)

        lines = self.database.GetListOfLines(sourcePath, addLastPeriod = True)

        annotations = []
        for line in lines:
            ann = self.GenerateAnnotation(line, False, CacheTag.Real, lineId = -1)
            annotations.append(ann)
        
        print('Saving annotations to file: ' + destPath)
        self.database.SaveJsonData(annotations, destPath)

    def GetDepMapList(self, annotation):
        return annotation['sentences'][0][self.dependencyOfChoice]

    def GetTokenMapList(self, annotation):
        return annotation['sentences'][0]['tokens']

    def GetRootNode(self, depMapList):
        rootNode = -1
        for depMap in depMapList:
            if depMap['dep'] == 'ROOT':
                rootNode = depMap['governor'] 
                break
        if rootNode < 0:
            raise Exception('Rood node not found.')
        else:
            return rootNode

    def GetDependentNodeList(self, governor, depMapList ):
        depList = []
        for depMap in depMapList:
            if depMap['governor'] == governor:
                depList.append(depMap['dependent'])
        return depList

    def GetTokenList(self, sentence, normalized, cacheTag, lineId, lineFeed = False):
        """
        Better call this method with normalized sentences only.
        """
        cache = None
        if cacheTag == CacheTag.Train: cache = self.trainCache
        elif cacheTag == CacheTag.Test: cache = self.testCache

        if cache is not None and cache[lineId].get('TL', None) != None:
            tList, tString, depMatList, tokenMapList = cache[lineId]['TL']
        else:
            tList = []; tString = ''
            annotation = self.GenerateAnnotation(sentence, normalized, cacheTag, lineId)
            depMatList = self.GetDepMapList(annotation)
            tokenMapList = self.GetTokenMapList(annotation)
            for tmap in tokenMapList:
                token = tmap['originalText'] 
                tList.append(token)
                tString += (' ' + token + ',')
            tString = tString[1:-1]
            if lineFeed:
                tString += '\n'

            if cache is not None and cache[lineId].get('TL', None) == None:
                cache[lineId]['TL'] = (tList, tString, depMatList, tokenMapList)

        return tList, tString, depMatList, tokenMapList

    def LoadGenerateDepMapList(self, trainNotTest = True):
        if trainNotTest:
            pSentences = self.database.pTrainSentences
            pDependencies = self.database.pTrainDependency
            annotation = self.trainAnnotation
        else:
            pSentences = self.database.pTestSentences
            pDependencies = self.database.pTestDependency
            annotation = self.testAnnotation

        if self.database.Exists(pDependencies):
            pass
        else:
            lines = self.database.GetListOfLines(pSentences, addLastPeriod = True)
            assert len(lines) == len(annotation)

            dictionalry = {}
            for id in range(len(lines)):
                dictionalry['sentence-' + str(id+1)] = lines[id]
                dictionalry['dependency-' + str(id+1)] = self.GetDepMapList(annotation[id])

            self.database.SaveJsonData(dictionalry, pDependencies)


    def LookupForTokenLowered(self, tokenIndex, tokenMapListSen):
        token = ''
        for tokenMap in tokenMapListSen:
            if tokenMap['index'] == tokenIndex:
                token = tokenMap['originalText'].lower()
                break
        if token == '':
            raise Exception("Couldn't find token for index {}".format(tokenIndex))
        else:
            return token

    def LookupForDependencyLabel(self, governorNode, dependentNode, depMapListSen, parTokenListSen):
        depLabel = ''
        for depMap in depMapListSen:
           if depMap['governor'] == governorNode and depMap['dependent'] == dependentNode:
                depLabel = depMap['dep']
                break
        if depLabel == '':
            raise Exception("Couldn't find dependency lable for governor {} and dependent {}".format(governorNode, dependentNode))
        elif self.testMode:
            govIndex = governorNode - 1; depIndex = dependentNode - 1
            print("({} {}) ----> ({} {}) : {}".format(govIndex, parTokenListSen[govIndex], depIndex, parTokenListSen[depIndex], depLabel))

        return depLabel

    def LookupForDependencyId(self, dependencyLabel):
        dependenciesList = self.parameters['dependenciesList']
        depId = -1
        for num in range(len(dependenciesList)):
            if dependenciesList[num] == dependencyLabel:
                depId = num; break
        if depId < 0:
            raise Exception("Couldn't find dependency label in dependency dictionary.")
        else:
            return depId

    def RetrieveAnnotation(self, lineId, trainNotTest = True):
        if trainNotTest:
            return self.trainAnnotation[lineId]
        else:
            return self.testAnnotation[lineId]
