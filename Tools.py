from Database import Database
from VisualClass import VisualClass


class Tools():
    def __init__(self, databasePath):
        self.database = Database(databasePath)
        
    def VisualizeTrainHistory(self):
        seriesDict = {}

        history = self.database.LoadBinaryData(self.database.pTrainHistory)

        metaParams, lossHistory, scoreHistory = history
        lossSereis = lossHistory
        seriesDict['loss'] = lossSereis

        hitRateSeries = self.database.GetHitRate(scoreHistory)
        seriesDict['hitRate'] = hitRateSeries
        
        precisionSList, recallSList = self.database.GetF1Score(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)

        for clsId in range(nClasses) :
            assert len( precisionSList[clsId] ) == len(lossHistory)

        avgPrecision = [None] * len(lossHistory); avgRecall = [None] * len(lossHistory)
        for step in range(len(lossHistory)):
            
            precision = None; cntPrecision = 0
            recall = None; cntRecall = 0
            for clsId in range(nClasses):
                if precisionSList[clsId][step] == None: pass
                else:
                    cntPrecision += 1
                    if precision == None: precision = precisionSList[clsId][step]
                    else: precision += precisionSList[clsId][step]

                if recallSList[clsId][step] == None: pass
                else:
                    cntRecall += 1
                    if recall == None: recall = recallSList[clsId][step]
                    else: recall += recallSList[clsId][step]
            if cntPrecision > 0: precision /= cntPrecision
            if cntRecall > 0: recall /= cntRecall
            
            avgPrecision[step] = precision
            avgRecall[step] = recall
        
        seriesDict['avgPrecision'] = avgPrecision

        title = "Model meta params: {}".format(metaParams)

        vc = VisualClass()
        vc.PlotStepHistory(title, seriesDict)


    def GetAverageHistory(self, history):
        metaParams, lossHistory, scoreHistory = history
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