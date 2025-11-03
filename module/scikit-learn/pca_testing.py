import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
#In general a good idea is to scale the data
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    

pca = PCA()
x_new = pca.fit_transform(X)

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

#Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()


#--------------------------------------------------
#https://www.jcchouinard.com/pca-loadings/#:~:text=In%20PCA%2C%20loadings%20indicate%20the,stronger%20contribution%20to%20principal%20components.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
 
# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

 
# Apply PCA with two components 
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_standardized)
x_pca = np.round(x_pca, 2)
x_pca


pca = PCA(n_components=2)
pca.fit(X_standardized)
x_pca2 = pca.transform(X_standardized)
x_pca2 = np.round(x_pca2, 2)

# Extract loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
 
 
# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], 
                           index=iris.feature_names)
loadings_df

explained_variance = pca.explained_variance_
print("Explained_variance:")
pd.DataFrame({
    'Explained Variance': explained_variance,
    'Explained Variance Ratio': pca.explained_variance_ratio_,
}, index=['PC1', 'PC2'])

print("\nLoadings:")
loadings = eigenvectors.T * np.sqrt(explained_variance)
pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=iris.feature_names)


#--------------------------------------------------
pca = PCA(n_components=len(fndf.columns)-1)
pca.fit(fndf.T)
gene2component = pd.DataFrame(pca.components_, columns = fndf.index)
pc2sample = pd.DataFrame(pca.transform(fndf.T), index = fndf.columns)

pc2sample = pd.DataFrame(np.round(x_pca2, 2))