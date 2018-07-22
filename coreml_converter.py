from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import load_model
import coremltools
import numpy as np
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="(required) The model to be converted.")
#ap.add_argument("-l", "--labels", type=str, required=True, help="(required) The labels of the model.")
args = vars(ap.parse_args())

model = load_model(args["model"])
#lb = pickle.load(open(args["labels"], "rb").read())
#num_classes = len(lb.classes_)
#labels = lb.classes_
#labels = ['background', 'bulbasaur', 'charizard', 'charmander', 'dragonite', 'gengar', 'gyrados', 'pikachu', 'snorlax', 'squirtle', 'zapdos']
labels = ['barckground', 'dragonite', 'gengar', 'gyrados', 'snorlax', 'zapdos']

coreml_mobilenet_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
    class_labels=labels,
	is_bgr=True)

print("[INFO] saving model as new.mlmodel")
coreml_mobilenet_model.save("new.mlmodel")
