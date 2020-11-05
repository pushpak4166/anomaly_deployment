from keras.models import load_model
import numpy as np
#loading the model

model = load_model('anomaly_detection_model_final.h5')

# 1st input data
samples = np.array([[100,83,1952,19,5.26,57,0.231]])
prediction = model.predict_classes(samples)
print("For 1st input data: \n")
print("Predicted class: ",prediction)
pred = model.predict(samples)
print("Predicted prob: ",pred)
print("\n")

# 2nd input data
samples2 = np.array([[96.11490583,109.8962564,3565.196403,66.58616576,57.66038318,12.05468241,0.761498981]])
print("For 2nd input data: \n")
prediction2 = model.predict_classes(samples2)
print("Predicted class: ",prediction2)
pred2 = model.predict(samples2)
print("Predicted prob: ",pred2)