from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from Network import Network
import pickle
#function to normalize images
def normalize_data(data): 
	return data/255

topics = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

#load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

with open('mnist.pkl', 'rb') as net: 
	model = pickle.load(net)


#basic preprocessing of data
x_train = x_train.reshape((len(x_train), 784))/255
x_test = x_test.reshape((len(x_test), 784))/255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

encodings = model.feed_forward(x_test)

pca = PCA(n_components=5)
pca_result = pca.fit_transform(encodings)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
##Variance PCA: 0.993621154832802

#Run T-SNE on the PCA features.
tsne = TSNE(n_components=2, verbose = 1)
tsne_results = tsne.fit_transform(pca_result[:10000])




color_map = np.argmax(y_test, axis=1)
plt.figure(figsize=(10,10))
for cl in range(10):
    indices = np.where(color_map==cl)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=topics[cl])
plt.legend()
plt.show()
