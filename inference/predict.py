from keras.models import load_model
import numpy as np
import pickle
import os

class PredictSentiment():
    def __init__(self) -> None:
        with open(os.path.join("models", "clf.pkl"), 'rb') as f:
            self.model_mlp = pickle.load(f)
        self.model_lstm = load_model(os.path.join("models","model_lstm.h5"))       
        pass

    def predict_ann(self,bow) -> np.int64:
        return self.model_mlp.predict(bow)[0]

    def predict_lstm(self,input_ids) -> np.int64:
        input_ids = np.reshape(input_ids, (1,78))
        result = self.model_lstm.predict(input_ids,batch_size=1,verbose = 0)[0]
        return np.argmax(result)
