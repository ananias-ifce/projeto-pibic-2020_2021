#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CÓDIGO: PRÉ-PROCESSAMENTO DO PROJETO
# PRÉ PROCESSAMENTO: drop, SimpleImputer; underSample; Z-Score;
# CLASSIFICADORES: RandomForest, Multilayer Perceptron
# PIBIC 2020.2-2021.1 - VERSÃO 1.0
# AUTOR: Ananias Caetano de Oliveira
# ORIENTADOR: Rhyan Ximenes de Brito
# INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DO CEARÁ
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler # responsavel pelo balanceamento reduzindo a base
from scipy import stats

def pre_processa(base, metodo):
    # removendo as linhas que não possuem resultado de chuva
    # apagar somente os registros sem resultados
    base.drop(base[pd.isnull(base['Chuva (mm)'])].index, inplace=True)
    # Tornando a ultima coluna como 0 e 1
    base.loc[base['Chuva (mm)'] > 0, 'Chuva (mm)'] = 1  # atualizando o campo para a média
    
    X = base.iloc[:, 2:18].values # X TRAIN
    Y = base.iloc[:, 18].values # Y TRAIN
    
    # retirando valores 'NaN' e substituindo por media (mean)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:17])
    X[:, 1:17] = imputer.transform(X[:, 1:17])
    print('Aplicação SimpleImputer')
    
    # metodo underSampler
    # transformação da base de dados
    undersample = RandomUnderSampler()
    X, Y = undersample.fit_sample(X, Y)
    print('Aplicação UnderSampler')
    
    # Normalização Z Score
    X = stats.zscore(X)
    
    return X, Y
        