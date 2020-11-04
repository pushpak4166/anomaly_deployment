from keras.models import load_model
import numpy as np
model = load_model('anomaly_model_new.h5')
samples = np.array([-97.72768685,30.55331562,27.240221,16,245,257,3.1477048,5.959532,6.654961,-0.015488088,86,1928.75,31,21.17647171,63,16.07843208])
prediction = model.predict_classes(samples)
print(prediction)
pred = model.predict(samples)
print(pred)