from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from face.recognition import 
#from tensorflow.lite.python.lite import Interpreter


print("Loading face net model \n")
face_recognition = load_model('../face_recognition/model.h5', compile=False)
print("Face-net model loaded")

print("loading face-expression model \n")
face_expression  = load_model('../face_expression/model.h5', compile=False)
print("Face-expression model loaded")

if __name__=="main":
    
