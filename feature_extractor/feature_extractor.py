from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import model_from_json
import numpy as np

def save_model_json(model, model_path = 'models/',model_name = 'model' ):
    model_json = model.to_json()
    with open(model_path+model_name+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_path+model_name+".h5")
    return

def load_model_json(model_path = 'model.json', weights_path="model.h5"):

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)

    return loaded_model

def preprocess_image(x):
    x = x.convert('RGB')
    x = x.resize((224,224))
    x = np.array(x)
    return preprocess_input(x)


class feature_extractor:
    def __init__(self, model_path = 'models/feature_extractor1.json',weights_path = 'models/feature_extractor1.h5'):

        self.model = load_model_json(model_path=model_path, weights_path=weights_path)


    def predict(self,x):
        #if(x.shape == (224,224,3)):
        ##x = np.expand_dims(x,axis = 0)

        return self.model.predict(x)
