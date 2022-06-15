import pandas as pd
import pathlib
import seaborn as sns

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# generation
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA  # faz o PCA
import numpy as np

import random as rd


def clean_data(PATH: str, label_column: str, drop_columns_list: list = [], drop_columns: bool = False, have_classes: bool = True, dropna: bool = True):
    PATH = pathlib.Path(PATH)
    df = pd.read_csv(PATH)
    if drop_columns:
        try:
            df = df.drop(columns=drop_columns_list)
        except:
            pass
    try:
        df = df.drop(columns='Id')
    except:
        pass

    if dropna:
        df = df.dropna().reset_index(drop=True)
    columns = df.columns

    if have_classes:
        y = df[label_column]
        classes = set(y)
        classes = list(classes)
        X = df.drop(columns=label_column)
        return df, fn_cat_onehot(df), fn_cat_onehot(X), y, classes, columns
    else:
        y = df[label_column]
        X = df.drop(columns=label_column)
        return df, fn_cat_onehot(df), fn_cat_onehot(X), y, None, columns

def pair_plot(df, hue=None, height=1.5, save=False):
    plt.figure()
    sns.pairplot(df, hue=hue, height=height)
    if save:
        plt.savefig('img.png', format='png')
    plt.show()

def PCA_(X, y, label_column, classes, columns):
    color = ['blue', 'orange', 'green', 'purple', 'brown', 'pink',
             'gray', 'olive', 'cyan', 'black', 'yellow', 'white']
    scaled_data = sklearn.preprocessing.scale(X)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    columns = list(columns)
    columns.remove(label_column)
    aux = pd.Series(pca.components_[0], index=columns)
    top_dimesoes = aux.abs().sort_values(ascending=False)
    print(f'Top dimensions\n--------------\n{top_dimesoes}')

    # quanta imformação cada pc tem em %, o ultimo é zero pq ne
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # labels para o grafico
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)  # grafico de barras
    
    plt.ylabel('Contribuição na variancia dos dados')
    plt.xlabel('Componente Principal')
    plt.title('Importancia de Cada componente')
    plt.show()

    df = pd.DataFrame(pca_data, columns=labels)
    df = pd.concat([df, y], axis=1)

    # Grafico com PC1 e PC2
    df = df[['PC1', 'PC2', 'PC3', 'Species']]

    if classes != None:
        for i in range(len(classes)):  # cada iteração printa um a cor
            aux = df.loc[df[label_column] == classes[i]]
            print('In {} the {} samples'.format(color[i], classes[i]))
            plt.scatter(aux.PC1, aux.PC2, color=color[i])  # color
    else:
        plt.scatter(df.PC1, df.PC2, color=color[rd.randrange(1, 10)])  # color

    plt.grid(True)
    plt.title('Grafico com PC1 e PC2')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.show()

    # Agora com PC1, PC2 e PC3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.title('Grafico com PC1, PC2 e PC3')
    ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
    ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
    ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))

    if classes != None:
        for i in range(len(classes)):  # cada iteração printa um a cor

            aux = df.loc[df[label_column] == classes[i]]
            print('In {} the {} samples'.format(color[i], classes[i]))
            # OBS: Em geral a escala de PC3 é bem menor que a dos outros eixos
            ax.scatter(aux.PC1, aux.PC2, aux.PC3, color=color[i])
    else:
        # OBS: Em geral a escala de PC3 é bem menor que a dos outros eixos
        ax.scatter(df.PC1, df.PC2, df.PC3, color=color[rd.randrange(1, 10)])
    plt.show()
    return df, pca

def fn_cat_onehot(df):
    """Generate onehoteencoded features for all categorical columns in df"""
    #print(f"df shape: {df.shape}")
    # NaN handing
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"NaN = **{nan_count}** will be categorized under feature_nan columns")

    model_oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
    for c in  list(df.select_dtypes("category").columns) + list(df.select_dtypes("object").columns):
        print(f"Encoding **{c}**")  # which column
        matrix = model_oh.fit_transform(
            df[[c]]
        )  # get a matrix of new features and values
        names = model_oh.get_feature_names_out()  # get names for these features
        df_oh = pd.DataFrame(
            data=matrix, columns=names, index=df.index
        )  # create df of these new features
        
        #display(df_oh.plot.hist())
        
        df = pd.concat([df, df_oh], axis=1)  # concat with existing df
        
        df.drop(
            c, axis=1, inplace=True
        )  # drop categorical column so that it is all numerical for modelling

    #print(f"#### New df shape: **{df.shape}**")
    return df
