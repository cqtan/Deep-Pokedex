from keras.applications import mobilenet, resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import coremltools

mobilenet_model = mobilenet.MobileNet(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
labels = []

with open('imagenet_classes_clean.txt', 'r') as infile:
    data = infile.read().splitlines()
    labels = list(data)

coreml_mobilenet_model = coremltools.converters.keras.convert(mobilenet_model,
	input_names="image",
	image_input_names="image",
    class_labels=labels,
	is_bgr=True)

print("[INFO] saving model as mobilenet.mlmodel")
coreml_mobilenet_model.save("mobilenet.mlmodel")

coreml_resnet_model = coremltools.converters.keras.convert(resnet_model,
	input_names="image",
	image_input_names="image",
    class_labels=labels,
	is_bgr=True)

print("[INFO] saving model as mobilenet.mlmodel")
coreml_mobilenet_model.save("resnet_model.mlmodel")