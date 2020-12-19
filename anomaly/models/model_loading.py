from keras.models import load_model
import numpy as np
#loading the model

model = load_model('anomaly_in_4_sensors_model.h5')

# 1st input data
samples = np.array([[70,82,25,24]])
print("For 1st input data: \n")
prediction = model.predict_classes(samples)
print("Predicted class: ",prediction)
# pred = model.predict(samples)
# print("Predicted prob: ",pred)
print("\n")

# 2nd input data
samples2 = np.array([[100,110,22,23]])
print("For 2nd input data: \n")
prediction2 = model.predict_classes(samples2)
print("Predicted class: ",prediction2)
# pred2 = model.predict(samples2)
# print("Predicted prob: ",pred2)