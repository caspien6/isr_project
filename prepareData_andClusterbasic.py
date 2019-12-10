import pickle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import prepare_data as prep_data


df = prep_data.open_data("dataset.pkl")


malware_families = { 'Empty': -1, 'Shohdi':0, 'Bublik':1, 'Mira':2, 'spyeye':3, 'Njrat':4, 'Shifu':5, 'sakula_v1_3':6,
 'Wabot':7, 'Warp':8, 'Nanocore_RAT_Gen_2':9, 'Wimmie':10, 'xRAT':11}
inv_malware_fam = {v: k for k, v in malware_families.items()}

tmp = df
tmp['tag'] = -1
for i, row in df.iterrows():
    for yara_rule in row['yara']:
        if yara_rule in malware_families:
            tmp['tag'][i] = malware_families[yara_rule]
            break



groups = {}
for i in range(-1,12):
    group_rows = tmp[tmp['tag'] == i]
    groups[inv_malware_fam[i]] =  group_rows.index
tmp['tag'] = tmp['tag'].apply(lambda x: inv_malware_fam[x])


# # Prepare the yara column


df_yara = pd.DataFrame(df['yara'])
'''su_list = prep_data.get_unique_values(df, 'yara')
for colname in su_list:
    df_yara[colname] = 0
df1 = prep_data.fill_up_the_relevance_table(df_yara, set(su_list), 'yara')'''


# # Prepare the strings column

df1 = prep_data.prepare_string_column(df, 2, 22, 20)


res_1 = df1.iloc[:,1:]

#pca = PCA(n_components=5)
#res = pca.fit_transform(res_1)
res = res_1



kmean_ncluster = 8
kmeans = KMeans(n_clusters=kmean_ncluster, init='k-means++', max_iter=400, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(res)

clustering = DBSCAN(eps=10, min_samples=2)
pred_y2 = clustering.fit_predict(res)

spectral_ncluster = 8
spectral = SpectralClustering(spectral_ncluster, eigen_solver='arpack', affinity="nearest_neighbors")
pred_y3 = spectral.fit_predict(res)

prep_data.check_result(res, pred_y)
prep_data.check_result(res, pred_y2)
prep_data.check_result(res, pred_y3)


view = pd.DataFrame(df_yara.loc[:,('yara')].copy())
view['kmeans'] = pd.DataFrame(pred_y)
view['dbscan'] = pd.DataFrame(pred_y2)
view['spectral_clustering'] = pd.DataFrame(pred_y3)
view['validation_groups'] = tmp['tag']


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(res)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2', 'principal component 3'])


finalDf = pd.concat([principalDf, pd.DataFrame(view['validation_groups'].apply(lambda x: malware_families[x]).values.tolist(), columns=['label'])], axis = 1)
finalDf1 = pd.concat([principalDf, pd.DataFrame(pred_y, columns=['label'])], axis = 1)
finalDf2 = pd.concat([principalDf, pd.DataFrame(pred_y2, columns=['label'])], axis = 1)
finalDf3 = pd.concat([principalDf, pd.DataFrame(pred_y3, columns=['label'])], axis = 1)

def visualize_result(mydf, ncluster):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title('Visualize clusters', fontsize = 20)
    targets = [k for k in range(-1,ncluster)]
    colors = ['r', 'g', 'b', 'gray', 'orange', 'yellow', 'purple', 'black', 'lightblue', 'lightgreen', 'pink', 'lightpurple']
    colors = colors[:ncluster]
    for target, color in zip(targets,colors):
        indicesToKeep = mydf['label'] == target
        ax.scatter3D(mydf.loc[indicesToKeep, 'principal component 1']
                   , mydf.loc[indicesToKeep, 'principal component 2']
                   , mydf.loc[indicesToKeep, 'principal component 3'] 
                   , s = 50)
        ax.legend(targets)
    plt.show()


visualize_result(finalDf, 12)
visualize_result(finalDf1, kmean_ncluster)
visualize_result(finalDf2, 12)
visualize_result(finalDf3, spectral_ncluster)

