import numpy as np
from sklearn.manifold import TSNE

from bumblebeat.utils.model import load_model


X = np.random.random((100,800))


# Load model

def extract_features(model):
	pass


def learn_tsne(X):
	return TSNE(n_components=2).fit_transform(X)

def create_plot(X, y):
	pass