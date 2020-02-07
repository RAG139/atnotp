import tensorflow as tf 

from Database import Database
from Parser import Parser
from Embedder import Embedder
from RNSCN import RNSCN
from GRU import GRU
from Autoencoder import Autoencoder
from Supervisor import Supervisor

from TestLayer import TestLayer

class Core():
    def __init__(self, embedderPath, database):

        self.database = database
        self.parser = Parser(self.database)
        self.embedder = Embedder(embedderPath)
        self.rnscn = RNSCN()
        self.gru = GRU()
        self.autoencoder = Autoencoder()
        self.supervisor = Supervisor()

        self.testlayer1 = TestLayer(units = 8, input_dim = 2)
        self.testlayer2 = TestLayer(units = 8, input_dim = 8)
    
    def Precict(features):
        predict = []

        return predict
