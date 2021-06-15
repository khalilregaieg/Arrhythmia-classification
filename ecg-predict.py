import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

classes=['N', 'S', 'V', 'F', 'Q']


input_file = sys.argv[1]
ecg = pd.read_csv(input_file, header=None)
X_ecg = ecg.iloc[:,:].values
X_ecg = X_ecg.reshape(len(X_ecg),X_ecg.shape[1],1)

best_model = load_model('best_model.hdf5')
predicted_number = int(np.argmax(best_model.predict(X_ecg)))
print(predicted_number)
prediction = "Predicted class : "+str(classes[predicted_number])
print(prediction)
plt.plot(ecg.iloc[0,:])
plt.title(prediction)
plt.show()
