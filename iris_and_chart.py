import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

iris = pd.DataFrame(iris)

iris['species'].value_counts()

sns.jointplot(x='sepal_length', y='sepal_width', data=iris)

sns.FacetGrid(iris, hue='species', size=5) \
.map(plt.scatter, 'sepal_length', 'sepal_width') \
.add_legend()

sns.boxplot(x='species', y='petal_length', data=iris)

sns.FacetGrid(iris, hue='species', size=5) \
.map(sns.kdeplot, 'petal_length') \
.add_legend()

sns.pairplot(iris, hue='species', size=3)

iris.boxplot(by='species', figsize=(12, 6))

from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris, 'species')

petal_l = iris.loc[:, ['petal_length', 'species']]
petal_l_versicolor = petal_l.loc[petal_l['species'] == 'versicolor', 'petal_length']
petal_l_virginica = petal_l.loc[petal_l['species'] == 'virginica', 'petal_length']
plt.hist([petal_l_versicolor, petal_l_virginica], stacked=True, bins=10, label=['versicolor', 'virginica'])
plt.legend()
plt.show()