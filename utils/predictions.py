"""
handle image processing for uploaded images
and inference from saved model
"""

import os
import pathlib

import numpy as np
import pandas as pd
import pickle
from utils.utils import jpeg_to_tensor

#tensorflow loading
from tensorflow.keras.models import load_model
from tensorflow import expand_dims


#set project path
PROJ_PATH = os.getcwd() #should be repo since this module is called from main.py

#set up model path (use relative because this is being called from main.py)
MODEL_PATH = os.path.join(PROJ_PATH, "static/model.h5")
assert os.path.isfile(MODEL_PATH), f"{MODEL_PATH} is not valid, check path"
model = load_model(MODEL_PATH)

#class names path (use relative becasue this is being called from main.py)
CLASSES_PATH = os.path.join(PROJ_PATH, "static/class_names.pkl")
assert os.path.isfile(CLASSES_PATH), f"{CLASSES_PATH} is not valid, check path"
with open(CLASSES_PATH, 'rb') as f:
    class_names = pickle.load(f)

def predictor(img_path, *, model=model, class_names=class_names):
    """
    Make prediction on file specified by img_path
    Args:
        img_path: path to image
        model: keras model
    Returns:
        prediction (DataFrame): 5 top choices. columns: ['name', 'probablity']
    """
    #get image as tensor
    img = jpeg_to_tensor(img_path, img_shape=224)

    #predict
    prediction = model.predict(expand_dims(img, axis=0)) #add batch axis to tensor

    #put prediction tensor into dataframe & format
    prediction = pd.DataFrame(np.round(prediction, 2), columns=class_names).transpose()
    prediction = prediction.reset_index()
    prediction.rename(columns={'index': 'name', 0: 'probability'}, inplace=True)
    prediction = prediction.nlargest(n=5, columns='probability')

    return prediction
