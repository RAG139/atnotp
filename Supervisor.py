import tensorflow as tf




class Supervisor():
    def __init__(self):
        self.optimizeer = tf.keras.optimizers.SGD(learning_rate = 0.01)

    def Initialize(self):
        pass

    def GetLoss(prediction, target):
        return tf.reduce_mean(tf.square(prediction, target))



"""
# Unit Test

O = Supervisor()
"""