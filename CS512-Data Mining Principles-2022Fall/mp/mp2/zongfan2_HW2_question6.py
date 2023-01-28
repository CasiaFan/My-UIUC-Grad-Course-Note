import numpy as np
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import f1_score

data_file = "HW2_source/ALOI.npz"

data = np.load(data_file)
x = data["X"]
y = data["y"]
print(x.shape, y.shape)

LOF 
clf = LOF()
clf.fit(x)

pred = clf.predict(x)
print(pred[:10])
f1 = f1_score(y, pred, average="micro")
print("F1 score of LOF: ", f1)

# autoencoder
clf = AutoEncoder(hidden_neurons=[27, 13, 13, 27])
clf.fit(x)

pred = clf.predict(x)
print(pred[:10])
f1 = f1_score(y, pred, average="micro")
print("F1 score of Autoencoder: ", f1) 
