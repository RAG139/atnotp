import datetime
import tensorflow as tf
import numpy as np

from Database import Database
from Parser import Parser
from Embedder import Embedder
from RNSCN import RNSCN
from GRU import GRU
from CNN import CNN
from MultiFCN import MultiFCN
from VisualClass import VisualClass
from Supervisor import Supervisor

from Config import *

class Model():
    
    def __init__(self, metaParams, testMode = False ):
        if metaParams == None or metaParams.get('dataPath', None) == None:
            print('Invalid Model Metaparameters (dataPath).')
            return

        self.database = Database(metaParams['dataPath'])
        dummyParser = Parser(self.database, dummyMode = True, testMode = testMode)
        dummyEmbedder = Embedder( bertPath = None, database = self.database ) # bertPath = None signals dummyMode
        dummyRnscn = RNSCN(self.database, dummyEmbedder, dummyParser, -1)

        if self.database.Exists(self.database.pTrainHistory):
            history = self.database.LoadBinaryData(self.database.pTrainHistory)
            fileMetaParams, _, _ = history

            createFromNewParams = False

            if  metaParams.get('dataPath', None) != None and metaParams['dataPath'] != fileMetaParams['dataPath'] or \
                metaParams.get('embPath', None) != None and metaParams['embPath'] != fileMetaParams['embPath'] :

                dummyRnscn.RemoveCache(); createFromNewParams = True

            if  metaParams.get('emb_Layers', None) != None and metaParams['emb_Layers'] != fileMetaParams['emb_Layers'] or \
                metaParams.get('emb_dim', None) != None and metaParams['emb_dim'] != fileMetaParams['emb_dim'] or \
                metaParams.get('RNSCN_dim', None) != None and metaParams['RNSCN_dim'] != fileMetaParams['RNSCN_dim'] or \
                metaParams.get('GRU_dim', None) != None and metaParams['GRU_dim'] != fileMetaParams['GRU_dim'] :

                self.RemoveCheckpointFiles(); createFromNewParams = True

            if createFromNewParams:
                self.__Create__( metaParams = metaParams, testMode = testMode )
            else:
                self.__Create__( metaParams = fileMetaParams, testMode = testMode )
        
        else:
            createFromNewParams = False

            if  metaParams.get('dataPath', None) != None and \
                metaParams.get('embPath', None) != None and \
                metaParams.get('emb_Layers', None) != None and \
                metaParams.get('emb_dim', None) != None and \
                metaParams.get('RNSCN_dim', None) != None and \
                metaParams.get('GRU_dim', None) != None :

                self.RemoveCheckpointFiles(); dummyRnscn.RemoveCache()
                self.__Create__( metaParams = metaParams, testMode = testMode )

            else:
                print("Metaparameters missing.")
                return


    def __Create__(self, metaParams, testMode = False ):
        print("Model Parameters: {}".format( metaParams))
 
        dataPath = metaParams['dataPath']
        embPath = metaParams['embPath']
        embLastNLayers = metaParams['emb_Layers']
        embFirstNHiddens = metaParams['emb_dim']
        rnscnDimHidden = metaParams['RNSCN_dim']
        gruDimHidden = metaParams['GRU_dim']

        self.metaParms = metaParams

        self.database = Database(dataPath)
        
        self.parser = Parser(self.database, testMode = testMode)
        self.embedder = Embedder(embPath, self.database, lastNLayers = embLastNLayers, firstNHiddens = embFirstNHiddens, testMode = testMode) # Take [lastNLayers layers by firstNHiddens hiddens].

        self.metaParms['embedder'] = 'BERT'
        self.metaParms['parser'] = 'CoreNLP'

        self.rnscn = RNSCN(self.database, self.embedder, self.parser, dim_hidden = rnscnDimHidden, topdown = False, testMode = testMode )
        self.rnscn2 = RNSCN(self.database, self.embedder, self.parser, dim_hidden = rnscnDimHidden, topdown = True, testMode = testMode )

        self.metaParms['rnscn_topdown'] = 'Yes'
        self.metaParms['rnscn_bottomup'] = 'Yes'

        self.gru = GRU(self.database, dim_input = self.rnscn.dim_hidden + self.rnscn2.dim_hidden, dim_hidden = gruDimHidden, normalizeLayer = True)
        self.gru2 = GRU(self.database, dim_input = self.rnscn.dim_hidden + self.rnscn2.dim_hidden, dim_hidden = self.gru.dim_hidden, normalizeLayer = True)

        self.metaParms['gru_leftward'] = 'Yes'
        self.metaParms['gru_rightward'] = 'Yes'

        dim_input = self.gru.dim_hidden + self.gru2.dim_hidden
        dim_outputs = [dim_input, self.database.NLabelClass()]
        activations = ['tanh', 'tanh']
        self.multifcn = MultiFCN(dim_batch = 1, dim_input = dim_input, dim_outputs = dim_outputs, activations = activations, useLayerNormalizer = True)

        self.nWeightTensors = self.rnscn.nWeightTensors + self.rnscn2.nWeightTensors + self.gru.nWeightTensors + self.gru2.nWeightTensors + self.multifcn.nWeightTensors

        self.testMode = testMode

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.03)
        self.weightsSnapshot = self.weights

        self.lossHistory = []
        self.scoreHistory = []

    def Initialize(self):
        self.database.Initialize()
        self.embedder.Initialize()
        self.parser.Initialize()
        self.rnscn.Initialize()
        self.rnscn2.Initialize()
        self.gru.Initialize()
        self.gru2.Initialize()
        self.multifcn.Initialize()

        self.lossHistory.clear()
        self.scoreHistory.clear()

    def Finalize(self):
        self.scoreHistory.clear()
        self.lossHistory.clear()
        self.multifcn.Finalize()
        self.gru2.Finalize()
        self.gru.Finalize()
        self.rnscn2.Finalize()
        self.rnscn.Finalize()
        self.parser.Finalize()
        self.embedder.Finalize()
        self.database.Finalize()


    def __GetWeightsTensorList(self):
        list = []
        list = list + self.rnscn.weights
        list = list + self.rnscn2.weights
        list = list + self.gru.weights
        list = list + self.gru2.weights
        list = list + self.multifcn.weights

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        self.rnscn.weights = list[cnt:]; cnt += self.rnscn.nWeightTensors
        self.rnscn2.weights = list[cnt:]; cnt += self.rnscn2.nWeightTensors
        self.gru.weights = list[cnt:]; cnt += self.gru.nWeightTensors
        self.gru2.weights = list[cnt:]; cnt += self.gru2.nWeightTensors
        self.multifcn.weights = list[cnt:]; cnt += self.multifcn.nWeightTensors
        
        self.nWeightTensors == cnt

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n
        
        return sum

    def RemoveWeights(self):
        self.database.RemoveFile( self.database.pWeights )

    def SaveWeights(self):
        self.database.SaveBinaryData( self.weights, self.database.pWeights )

    def LoadWeights(self):
        if self.database.Exists(self.database.pWeights):
            self.weights = self.database.LoadBinaryData(self.database.pWeights)

    def DemonstrateTokenMapping(self):
        self.rnscn.DemonstrateTokenMapping(trainNotTest = True, addLastPeriod = True)
        self.rnscn.DemonstrateTokenMapping(trainNotTest = False, addLastPeriod = True)

    def Train(self, shuffleBuffer = 1000, miniBatchSize = 20, epochs = 5):

        if  self.metaParms.get('shuffle', None) != None and self.metaParms['shuffle'] != shuffleBuffer or \
            self.metaParms.get('MBSize', None) != None and self.metaParms['MBSize'] != miniBatchSize :

            self.RemoveCheckpointFiles()

        self.metaParms['shuffle'] = shuffleBuffer
        self.metaParms['MBSize'] = miniBatchSize

        self.metaParms['tensors'] = self.nWeightTensors
        self.metaParms['weights'] = self.NWeights()

        self.Initialize()

        dsTrain = self.database.CreateFullDataset(trainNotTest = True, addLastPeriod = True, returnSizeOnly = False)
        dsTrain = dsTrain.shuffle(buffer_size = shuffleBuffer, reshuffle_each_iteration = True).batch(batch_size = miniBatchSize, drop_remainder = True)

        accumEpoch = 0; step = 0
        self.weightsSnapshot = self.weights
        epochRange = range(epochs)

        for epoch in epochRange:   
            tf.compat.v1.global_variables_initializer()

            accumEpoch = self.OnStartEpoch(epoch) + 1
            self.rnscn.OnStartEpoch(accumEpoch)

            lossHistory = []; scoreHistory = []
            for miniBatch in dsTrain :
                self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.03) 
                mbGradient, mbLoss, mbScore = self.LearnFromMiniBatch(batch = miniBatch, step = step)
                self.lossHistory.append(mbLoss); lossHistory.append(mbLoss)
                self.scoreHistory.append(mbScore); scoreHistory.append(mbScore)
                step += 1

            self.LogAccuracy( scoreHistory )

            avgLoss, avgHitRate = self.GetAverageHistory(lossHistory, scoreHistory)
            lossTest, scoresTest = self.EvaluateOnTestData()
            avgLossTest, avgHitRateTest = self.GetAverageHistory([lossTest], [scoresTest])


            self.metaParms['Epochs'] = accumEpoch
            self.metaParms['LossTrain'] = round(avgLoss, 2); self.metaParms['hitRateTrain'] = round(avgHitRate, 2)
            self.metaParms['LossTest'] = round(avgLossTest, 2); self.metaParms['hitRateTest'] = round(avgHitRateTest, 2)

            self.rnscn.OnEndEpoch(accumEpoch)
            self.OnEndEpoch(accumEpoch)

    def LogAccuracy(self, scoreHistory):
        hitRateList = self.database.GetHitRate(scoreHistory)
        precisionArray, recallArray = self.database.GetF1Score(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)

        print( 'Average hit rate : ', round( np.average(hitRateList), 2) )

        for c in range(nClasses):
            print( 'Average f1 precision for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(precisionArray[c, :]), 2) )

        for c in range(nClasses):
            print( 'Average f1 recall for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(recallArray[c, :]), 2) )

    def RemoveCheckpointFiles(self):
        self.RemoveWeights()
        self.database.RemoveFile(self.database.pTrainHistory)

    def OnStartEpoch(self, accumEpoch):
        self.LoadWeights()

        accumEpoch = 0
        if self.database.Exists(self.database.pTrainHistory):
            history = self.database.LoadBinaryData(self.database.pTrainHistory)
            self.metaParms, self.lossHistory, self.scoreHistory = history
            accumEpoch = self.metaParms['Epochs']

        return accumEpoch

    def OnEndEpoch(self, accumEpoch):
        history = (self.metaParms, self.lossHistory, self.scoreHistory)
        self.database.SaveBinaryData(history, self.database.pTrainHistory)
        self.SaveWeights()

    def LearnFromMiniBatch(self, batch, step):
        batchLoss = 0.0; scoreList = []

        sumGradient = []
        for weight in self.weightsSnapshot:
            sumGradient.append( tf.zeros_like( weight ) )

        batchSize = 0
        for dataset_record in batch:
            batchSize += 1

            with tf.GradientTape() as tape:
                tape.watch(self.weightsSnapshot)

                loss, score = self.GetLossForSingleExample(dataset_record, CacheTag.Train)
                batchLoss += loss.numpy()
                scoreList.append(score)
                
                tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
                print( '\nloss =', loss.numpy() )

            grad = tape.gradient(loss, self.weightsSnapshot)

            for n in range(len(sumGradient)):
                if grad[n] is not None:
                    tf.debugging.assert_all_finite(grad[n], message = 'Gradient is a nan.')
                    sumGradient[n] = tf.add( sumGradient[n], grad[n] )
                else: pass 


        self.optimizer.apply_gradients( zip(sumGradient, self.weightsSnapshot) )

        self.DoSomethingGreat()

        return sumGradient, batchLoss / batchSize, scoreList

    def DoSomethingGreat(self):
        pass

    def EvaluateOnTestData(self):
        batchLoss = 0.0; scoreList = []

        dsTest = self.database.CreateFullDataset(trainNotTest = False, addLastPeriod = True, returnSizeOnly = False)

        batchSize = 0
        for dataset_record in dsTest:
            batchSize += 1

            loss, score = self.GetLossForSingleExample(dataset_record, CacheTag.Test) 
            batchLoss += loss.numpy()
            scoreList.append(score)
                
            tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
            print( '\nloss =', loss.numpy() )

        return batchLoss / batchSize, scoreList

    def GetLossForSingleExample(self, dataset_record, cacheTag):
        assert isinstance(cacheTag, CacheTag)
        sentence, aspect, opinion, _, _, lineId = self.database.DecodeDatasetRecord(dataset_record)
        consistent, aspectList, opinionList, wrongAspList, wrongOpnList = self.database.GetRefeinedLabels(sentence, aspect, opinion, cacheTag, lineId)

        probDistList = self.GetProbabilityDistributionList(sentence, cacheTag, lineId)
        tokenLabelClassList, tokenLabelStringList = self.rnscn.GetTokenLabelList(sentence, aspectList, opinionList, cacheTag, lineId)


        assert len(probDistList) == len(tokenLabelClassList)
        trueDistList = [None] * len(probDistList)
        nClass = self.database.NLabelClass()

        for tokenId in range(len(trueDistList)):
            labelClassDist = tf.Variable(tf.zeros( shape = [nClass], dtype = weightDType ) )
            sensitivity = 1.0
            labelClassDist[ tokenLabelClassList[tokenId]  ].assign( sensitivity )
            trueDistList[tokenId] = labelClassDist

            text = 'PredDist: '
            for n in range(nClass):
                if n == tokenLabelClassList[tokenId]:
                    text = text + " [{0:2.0f}]".format(probDistList[tokenId][n].numpy() * 100)
                else:
                    text = text + "  {0:2.0f} ".format(probDistList[tokenId][n].numpy() * 100)
            if tokenLabelClassList[tokenId] != tf.argmax(probDistList[tokenId]): text = text + '   No'
        scoreList = []
        lossTotal = tf.constant(value = 0.0, dtype = weightDType )
        for probDist, trueDist in zip(probDistList, trueDistList):
            a = - tf.multiply( trueDist, tf.math.log(probDist) )
            assert a.shape == [self.database.NLabelClass()]
            tf.debugging.assert_all_finite(a, message = 'a is a nan.')
            assert len(probDistList) > 0
            a = tf.reduce_sum(a)
            loss = a / ( 2.0 * len(probDistList) )
            tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
            lossTotal = tf.add( lossTotal, loss )

            scoreList.append( (tf.argmax(trueDist).numpy(), tf.argmax(probDist).numpy() ) )

        return lossTotal, scoreList

    def Predict(self, sentence):
        probDistList = self.GetProbabilityDistributionList(sentence, CacheTag.Real, lineId = -1)
        predLabelStringList = [None] * len(probDistList)

        for tokenId in range(len(probDistList)):
            probDist = probDistList[tokenId]
            predClass = tf.argmax(probDist)
            assert predClass.shape == []
            predLabelString = self.database.GetTokenLabelString(predClass)
            predLabelStringList[tokenId] = predLabelString

        return predLabelStringList

    def GetProbabilityDistributionList(self, sentence, cacheTag, lineId):
        assert isinstance(cacheTag, CacheTag)
        successRnscn, hiddenListRnscn, _ = self.rnscn.GenerateStatesSingleSetence(sentence, cacheTag, lineId)
        successRnscn, hiddenListRnscn2, _ = self.rnscn2.GenerateStatesSingleSetence(sentence, cacheTag, lineId)

        if not successRnscn:
            raise Exception('RNSCN failed: ', sentence)

        for n in range(len(hiddenListRnscn)):
            hiddenListRnscn[n] = tf.concat( [ hiddenListRnscn[n], hiddenListRnscn2[n] ], axis = 0 )
        hiddenListGru = self.gru.GenerateStates(hiddenListRnscn)

        hiddenListRnscn.reverse()
        hiddenListGru2 = self.gru2.GenerateStates(hiddenListRnscn)
        hiddenListGru2.reverse()

        assert len(hiddenListGru) == len(hiddenListGru2)
        hiddenListBiGru = []
        for a, b in zip(hiddenListGru, hiddenListGru2):
            tf.debugging.assert_all_finite(a, message = 'GRU produced a nan.')
            tf.debugging.assert_all_finite(b, message = 'GRU2 produced a nan.')
            assert a.shape == [self.gru.dim_hidden, 1]
            assert b.shape == [self.gru2.dim_hidden, 1]
            hiddenListBiGru.append( tf.transpose( tf.concat([a, b], axis = 0) ) )

        probDistList = [None] * len(hiddenListBiGru)
        nClass = self.database.NLabelClass()
        for tokenId in range(len(hiddenListBiGru)):
            logits = self.multifcn.FeedForward( hiddenListBiGru[tokenId] )
            assert logits.shape == [ 1, nClass ]
            logits = tf.reshape(logits, [-1])
            probDist = tf.nn.softmax(logits = logits, axis = 0)
            assert probDist.shape == [ nClass ]
            probDistList[tokenId] = probDist

        return probDistList


    def VisualizeTrainHistory(self):
        seriesDict = {}

        history = self.database.LoadBinaryData(self.database.pTrainHistory)

        metaParams, lossHistory, scoreHistory = history
        lossSereis = lossHistory
        seriesDict['loss'] = lossSereis

        hitRateSeries = self.database.GetHitRate(scoreHistory)
        seriesDict['hitRate'] = hitRateSeries
        
        precisionArray, recallArray = self.database.GetF1Score(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)

        for clsId in range(nClasses):
            seriesDict['p.' + self.database.tokenLabelStringList[clsId]] = precisionArray[clsId, :]
            seriesDict['r.' + self.database.tokenLabelStringList[clsId]] = recallArray[clsId, :]

        for clsId in range(nClasses) :
            assert len( precisionArray[clsId] ) == len(lossHistory)

        avgPrecision = [None] * len(lossHistory); avgRecall = [None] * len(lossHistory)
        for step in range(len(lossHistory)):
                      
            avgPrecision[step] = np.nanmean( precisionArray[:, step] )
            avgRecall[step] = np.nanmean( recallArray[:, step] )
        
        seriesDict['avgPrecision'] = avgPrecision
        seriesDict['avgRecall'] = avgRecall

        title = "Training History: loss and accuracy by steps"

        vc = VisualClass()
        vc.PlotStepHistory(title, seriesDict, self.metaParms)


    def GetAverageHistory(self, lossHistory, scoreHistory):

        lossSereis = lossHistory
        
        avgLoss = 0.0; cnt = 0
        for loss in lossSereis:
            if loss is not None:
                avgLoss += loss
                cnt += 1

        assert cnt > 0
        avgLoss = avgLoss / cnt

        hitRateSeries = self.database.GetHitRate(scoreHistory)

        avgHitRate = 0.0; cnt = 0
        for hitRate in hitRateSeries:
            if hitRate is not None:
                avgHitRate += hitRate
                cnt += 1

        assert cnt > 0
        avgHitRate = avgHitRate / cnt

        return avgLoss, avgHitRate
