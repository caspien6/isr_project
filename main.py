import pickle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import prepare_data as prep_data


def plot_results(pred_y, principalComponents, ncluster):
	principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2'])

	finalDf = pd.concat([principalDf, pd.DataFrame(pred_y, columns=['label'])], axis = 1)

	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('String label clusters', fontsize = 20)
	targets = [k for k in range(ncluster)]
	colors = ['r', 'g', 'b', 'gray', 'orange', 'yellow', 'purple', 'black']
	colors = colors[:ncluster]
	for target, color in zip(targets,colors):
	    indicesToKeep = finalDf['label'] == target
	    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
	               , finalDf.loc[indicesToKeep, 'principal component 2']
	               , c = color
	               , s = 50)
	ax.legend(targets)
	plt.show()

def only_stringfeature():
	df = prep_data.open_data("dataset.pkl")
	# ## Prepare the string column
	df1 = prep_data.prepare_string_column(df, 3, 20, 16)
	res_1 = df1.iloc[:,1:]

	#pca = PCA(n_components=4)
	#res = pca.fit_transform(res_1)
	res = res_1


	ncluster = 5
	kmeans = KMeans(n_clusters=ncluster, init='k-means++', max_iter=400, n_init=10, random_state=0)
	pred_y = kmeans.fit_predict(res)

	clustering = DBSCAN(eps=16, min_samples=10)
	pred_y2 = clustering.fit_predict(res)

	spectral = SpectralClustering(ncluster, eigen_solver='arpack', affinity="nearest_neighbors")
	pred_y3 = spectral.fit_predict(res)

	print('Simple k-means k=', ncluster)
	prep_data.check_result(res, pred_y)
	print('DBSCAN')
	prep_data.check_result(res, pred_y2)
	print('SpectralClustering k=', ncluster)
	prep_data.check_result(res, pred_y3)

	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(res)
	plot_results(pred_y, principalComponents, ncluster)
	plot_results(pred_y2, principalComponents, ncluster)
	plot_results(pred_y3, principalComponents, ncluster)
	

only_stringfeature()