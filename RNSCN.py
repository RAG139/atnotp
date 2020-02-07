import tensorflow as tf
import numpy as np 

from Embedder import Embedder
from Parser import Parser
from Config import *


class RNSCN():
    def __init__(self, database, embedder, parser, dim_hidden, topdown = False, testMode = False):
        self.database = database

        assert isinstance(embedder, Embedder)
        self.embedder = embedder
        assert isinstance(parser, Parser)
        self.parser = parser

        if embedder.dummyMode or parser.dummyMode :
            self.dummyMode = True
            return
        else:
            self.dummyMode = False

        n_dependency_types = len(self.parser.parameters['dependenciesList'])
        dim_wordVec = self.embedder.lastNLayers * self.embedder.firstNHiddens

        cnt = 0
        self.w_wordvec_hidden = tf.Variable(initial_value = tf.random_normal_initializer()(shape = (dim_hidden, dim_wordVec), dtype = weightDType), trainable = True)
        cnt += 1
        self.b_wordvec_hidden = tf.Variable(initial_value = tf.zeros_initializer()(shape = (dim_hidden, 1), dtype = weightDType), trainable = True)
        cnt += 1
        self.w_hidden_relation = tf.Variable(initial_value = tf.random_normal_initializer()(shape = (dim_hidden, dim_hidden), dtype = weightDType), trainable = True)
        cnt += 1

        self.ws_relation_hidden = []
        for _ in range(n_dependency_types):
            w = self.w_hidden_relation = tf.Variable(initial_value = tf.random_normal_initializer()(shape = (dim_hidden, dim_hidden), dtype = weightDType), trainable = True)
            self.ws_relation_hidden.append(w)
            cnt += 1

        self.w_relation_probability = tf.Variable(initial_value = tf.random_normal_initializer()(shape = (n_dependency_types, dim_hidden), dtype = weightDType), trainable = True)
        cnt += 1 
        self.b_relation_probability = tf.Variable(initial_value = tf.zeros_initializer()(shape = (n_dependency_types, 1), dtype = weightDType), trainable = True)
        cnt += 1

        self.n_dependency_types = n_dependency_types
        self.dim_wordVec = dim_wordVec
        self.dim_hidden = dim_hidden

        self.topdown = topdown
        self.testMode = testMode

        self.nWeightTensors = cnt

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def SaveWeights(self):
        pass

    def LoadWeights(self):
        pass

    def RemoveCache(self):
        self.database.RemoveCache()
        self.parser.RemoveCache()
        self.embedder.RemoveCache()

    def OnStartEpoch(self, epoch):
        self.database.OnStartEpoch(epoch)
        self.parser.OnStartEpoch(epoch)
        self.embedder.OnStartEpoch(epoch)

    def OnEndEpoch(self, epoch):
        self.embedder.OnEndEpoch(epoch)
        self.parser.OnEndEpoch(epoch)
        self.database.OnEndEpoch(epoch)

    def __GetWeightsTensorList(self):
        list = []

        list.append(self.w_wordvec_hidden)
        list.append(self.b_wordvec_hidden)
        list.append(self.w_hidden_relation)
        
        for w in self.ws_relation_hidden:
            list.append(w)
        
        list.append(self.w_relation_probability)
        list.append(self.b_relation_probability)

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        
        self.w_wordvec_hidden = list[cnt]; cnt += 1
        self.b_wordvec_hidden = list[cnt]; cnt += 1
        self.w_hidden_relation = list[cnt]; cnt += 1
        
        for index in range(len(self.ws_relation_hidden)):
            self.ws_relation_hidden[index] = list[cnt]; cnt += 1
        
        self.w_relation_probability = list[cnt]; cnt += 1
        self.b_relation_probability = list[cnt]; cnt += 1

        self.nWeightTensors = cnt

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n

        return sum

    def Predict(self, sentence):
        prediction = []
        return prediction

    def GenerateStatesSingleSetence(self, sentence, cacheTag, lineId):
        assert isinstance(cacheTag, CacheTag)

        compatible, parToEmbMapping, normalizedLine, _, parDepMatList, parTokenMapList, parTokenList, _, embTokenList \
            = self.GetNormalizedTokenMapping(sentence, cacheTag, lineId, lineFeed = False)
            
        success = True

        if compatible == False:
            success = False
            print("Couldn't find token mapping.")
        else:
            self.parToEmbTokenMappingSen = parToEmbMapping
            self.embLayersExtSen = self.embedder.GetEmbLayersExtended(normalizedLine)
            self.parDepMapListSen = parDepMatList
            self.parTokenListSen = parTokenList

            rootNode = self.parser.GetRootNode(self.parDepMapListSen)
            topNodesList = self.parser.GetDependentNodeList(rootNode, self.parDepMapListSen )
            if len(topNodesList) != 1:
                raise Exception("More than 1 top nodes found.")

            if self.testMode:
                self.PrintParToEmbMapping(parTokenList, embTokenList, parToEmbMapping, topNodesList)

            self.hiddenListSen = [None] * len(parTokenMapList)
            self.relationListSen = [None] * len(parTokenMapList)

            if self.topdown :
                self.relationListSen[ 0 ] = []
                self.ProcessNodeTopDown(topNodesList[0], 0, tf.zeros_initializer()(shape = [self.dim_hidden, 1], dtype = weightDType ) ) # The goal is to populate self.hiddenListSen and self.relationListSen
            else: 
                self.ProcessNodeBottomUp(topNodesList[0]) 

            success = True
        
        return success, self.hiddenListSen, self.relationListSen
    

    def ProcessNodeBottomUp(self, focusDepNode):
        dependentNodesList = self.parser.GetDependentNodeList(focusDepNode, self.parDepMapListSen )
        if self.testMode and len(dependentNodesList) > 0:
            text = 'dependency: ' + str(focusDepNode - 1) + ' ---> '
            for dep in dependentNodesList: text += (str( dep-1 ) + ', ')

        if focusDepNode <= 0 :
            focusWordVec = tf.Variable( tf.zeros( shape = [self.dim_wordVec, 1], dtype = weightDType )  )
        else:
            focusWordVec = self.embedder.AverageWordVector( self.embLayersExtSen, self.parToEmbTokenMappingSen[ focusDepNode - 1 ] )
            assert not np.isnan(focusWordVec).any()
            focusWordVec = tf.convert_to_tensor([ focusWordVec ], dtype = weightDType )
            tf.debugging.assert_all_finite(focusWordVec, message = 'wordVec is a nan at 2.')
            focusWordVec = tf.transpose( focusWordVec )

        focusLinearPart = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-01')
        focusHiddenSimple = tf.tanh ( tf.add( focusLinearPart, self.b_wordvec_hidden ) )

        if len(dependentNodesList) == 0:    
            if focusDepNode > 0: 
                focusHidden = focusHiddenSimple
            else:
                raise Exception('Root node with node dependents.')
        else:
            self.relationListSen[ focusDepNode - 1 ] = [] #

            sumDependentInfluence = tf.Variable( tf.zeros( shape = [self.dim_hidden, 1], dtype = weightDType ) )

            for dependentDepNode in dependentNodesList:                
                dependendHidden = self.ProcessNodeBottomUp(dependentDepNode) 
                
                dependentRole = tf.matmul( self.w_hidden_relation, dependendHidden, name = 'matmul-02' )
                focusRole = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-03') 
                dependentRelation = tf.tanh( tf.add( dependentRole, focusRole ) ) 
                tf.debugging.assert_all_finite(dependentRelation, message = 'dependentRelation is a nan.')
                self.relationListSen[ focusDepNode - 1 ].append( (dependentDepNode - 1, dependentRelation) ) 
                
                depLabel = self.parser.LookupForDependencyLabel(focusDepNode, dependentDepNode, self.parDepMapListSen, self.parTokenListSen) 
                depId = self.parser.LookupForDependencyId(depLabel)

                dependentInfluence = tf.matmul( self.ws_relation_hidden[depId], dependentRelation, name = 'matmul-04')
                sumDependentInfluence = tf.add( sumDependentInfluence, dependentInfluence )

            focusHidden = tf.tanh ( tf.add( sumDependentInfluence, focusHiddenSimple ) )

        self.hiddenListSen[ focusDepNode - 1 ] = focusHidden 
        tf.debugging.assert_all_finite(focusHidden, message = 'ficusHidden is a nan.')

        return focusHidden


    def ProcessNodeTopDown(self, focusDepNode, governorDepNode, governorHidden):
        dependentNodesList = self.parser.GetDependentNodeList(focusDepNode, self.parDepMapListSen )

        if self.testMode and len(dependentNodesList) > 0:
            text = 'dependency: ' + str(focusDepNode - 1) + ' ---> '
            for dep in dependentNodesList: text += (str( dep-1 ) + ', ')

        if focusDepNode <= 0 :
            focusWordVec = tf.Variable( tf.zeros( shape = [self.dim_wordVec, 1], dtype = weightDType )  )
            assert focusWordVec.shape == [self.dim_wordVec, 1]
        else:
            focusWordVec = self.embedder.AverageWordVector( self.embLayersExtSen, self.parToEmbTokenMappingSen[ focusDepNode - 1 ] )
            assert not np.isnan(focusWordVec).any()
            focusWordVec = tf.convert_to_tensor([ focusWordVec ], dtype = weightDType )
            assert focusWordVec.shape == [1, self.dim_wordVec]
            tf.debugging.assert_all_finite(focusWordVec, message = 'wordVec is a nan at 2.')
            focusWordVec = tf.transpose( focusWordVec )
            assert focusWordVec.shape == [self.dim_wordVec, 1]
        
        self.relationListSen[ focusDepNode - 1 ] = []

        focusLinearPart = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-01')
        focusHiddenSimple = tf.tanh( tf.add( focusLinearPart, self.b_wordvec_hidden ) )
        assert focusHiddenSimple.shape == [self.dim_hidden, 1]

        governorRole = tf.matmul( self.w_hidden_relation, governorHidden, name = 'matmul-02' )
        focusRole = tf.matmul(self.w_wordvec_hidden, focusWordVec, name = 'matmul-03') 
        governorRelation = tf.tanh( tf.add( governorRole, focusRole ) ) 
        tf.debugging.assert_all_finite(governorRelation, message = 'governorRelation is a nan.')
        assert governorRelation.shape == [self.dim_hidden, 1]
        self.relationListSen[ focusDepNode - 1 ].append( (governorDepNode - 1, governorRelation) ) 

        depLabel = self.parser.LookupForDependencyLabel(governorDepNode, focusDepNode, self.parDepMapListSen, self.parTokenListSen)
        depId = self.parser.LookupForDependencyId(depLabel)

        governorInfluence = tf.matmul( self.ws_relation_hidden[depId], governorRelation , name = 'matmul-04')
        focusHidden = tf.tanh ( tf.add( governorInfluence, focusHiddenSimple ) )
        assert focusHidden.shape == [self.dim_hidden, 1]

        tf.debugging.assert_all_finite(focusHidden, message = 'hidden is a nan.')
        self.hiddenListSen[ focusDepNode - 1 ] = focusHidden 
        print("Node finished: ", focusDepNode - 1)

        for dependentDepNode in dependentNodesList:
            self.ProcessNodeTopDown(dependentDepNode, focusDepNode, focusHidden) 


    def GetNormalizedTokenMapping(self, line, cacheTag, lineId, lineFeed = True):
        nline = self.parser.NormalizeSentence(line, cacheTag, lineId)
        parTokenList, parTokenString, parDepMatList, parTokenMapList = self.parser.GetTokenList(nline, True, cacheTag, lineId, lineFeed)
        embTokenList, embTokenString = self.embedder.GetTokenList(nline, cacheTag, lineId, lineFeed)
        compatible, mapping = self.embedder.GetTokenMapping(parTokenList, embTokenList, cacheTag, lineId)
        
        return compatible, mapping, nline, parTokenString, parDepMatList, parTokenMapList, parTokenList, embTokenString, embTokenList 

    def GetTokenLabelList(self, sentence, aspectList, opinionList, cacheTag, lineId):
        normalizedLine = self.parser.NormalizeSentence(sentence, cacheTag, lineId)
        parTokenList, _, _, _ = self.parser.GetTokenList(normalizedLine, True, cacheTag, lineId, lineFeed = False)
        tokenLabelClassList, tokenLabelStringList = self.database.GetTokenLabelList(parTokenList, aspectList, opinionList)

        return tokenLabelClassList, tokenLabelStringList

    def PrintParToEmbMapping(self, parTokenList, embTokenList, parToEmbMapping, topNodesList):
        text = "parTokens: "            
        for id in range(len(parTokenList)):
            text += ( '(' + str(id) + ' ' + parTokenList[id].lower() + ') ' )
        print(text)

        text = "embTokens: "
        for id in range(len(embTokenList)):
            text += ( '(' + str(id) + ' ' + embTokenList[id] + ') ' )
        print(text)

        text = 'parToEmb mapping: '
        for parId in range(len(parToEmbMapping)):
            text += (str(parId) + '-')
            if len(parToEmbMapping[parId]) <= 1: text +=  str(parToEmbMapping[parId][0]) # the length is one, not zero.
            else:
                text += '('
                for embId in range(len(parToEmbMapping[parId])):
                    text += (str(parToEmbMapping[parId][embId]) + ' ')
                text = text[:-1]; text += ')'
            text += ' '
        print(text)

        print('top node = ', topNodesList[0])

    def DemonstrateTokenMapping(self, trainNotTest, addLastPeriod = True):
        if trainNotTest:
            pSentences = self.database.pTrainSentences
            pSentencesClean = self.database.pTrainSentencesClean
            pSentencesRemoved = self.database.pTrainSentencesRemoved
        else:
            pSentences = self.database.pTestSentences
            pSentencesClean = self.database.pTestSentencesClean
            pSentencesRemoved = self.database.pTestSentencesRemoved
        
        nRemoved = 0

        cleanLines = []; removedLines = []
        lines = self.database.GetListOfLines(pSentences, addLastPeriod)
        for line in lines:
            compatible, mapping, nline, parTokensString, parDepMatList, parTokenMapList, parTokenList, embTokensString, embTokenList = \
                self.GetNormalizedTokenMapping(line, lineFeed = True)

            if not compatible :
                nRemoved += 1
                removedLines.append(nline)
                removedLines.append(parTokensString)
                removedLines.append(embTokensString)
            else:
                cleanLines.append(nline)

        cleanFile = open(pSentencesClean, 'wt+')
        cleanFile.writelines(cleanLines)
        cleanFile.close()

        cleanFile = open(pSentencesRemoved, 'wt+')
        cleanFile.writelines(removedLines)
        cleanFile.close()

        return nRemoved

    def GenerateLabelFile(self, trainNotTest = True):
        labelList = []

        dataset = self.database.CreateTextLinesDataset(trainNotTest, addLastPeriod = True)

        for dataset_record in dataset:
            sentence, aspect, opinion, lineId = self.database.DecodeTextLineDatasetRecord(dataset_record)

            consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.database.GetRefeinedLabels(sentence, aspect, opinion)
                
            normalizedLine = self.parser.NormalizeSentence(sentence)
            parTokenList, _, _, _ = self.parser.GetTokenList(normalizedLine, True, CacheTag.Real, lineId = -1, lineFeed = False)
            tokenLabelStringList, tokenLabelNumeralList = self.database.GetTokenLabelList(parTokenList, aspectList, opinionList)

            labelList.append(tokenLabelNumeralList)

        if trainNotTest: pFile = self.database.pTrainTokenLabelClass
        else: pFile = self.database.pTestTokenLabelClass 
        self.database.SaveJsonData(labelList, pFile)