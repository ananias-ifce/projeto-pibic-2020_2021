#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CÓDIGO: CLASSIFICAÇÃO DO PROJETO
# PRÉ PROCESSAMENTO: drop, SimpleImputer; underSample; Z-Score;
# CLASSIFICADORES: RandomForest, Multilayer Perceptron
# PIBIC 2020.2-2021.1 - VERSÃO 1.0
# AUTOR: Ananias Caetano de Oliveira
# ORIENTADOR: Rhyan Ximenes de Brito
# INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DO CEARÁ
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def CLASSIFICA(x_train, y_train, x_test, classe):
    
    if classe['name'] == 'Random_Forest':
        # random_state=0 faz com que resultado de precisão seja sempre o mesmo, sem ele o resultado vai variar de acordo com a base de treino que for gerada
        
        if classe['posfixo'] == '_1':
            classificador = RandomForestClassifier(n_estimators = 20, 
                                                   criterion='gini')
        elif classe['posfixo'] == '_2':
            classificador = RandomForestClassifier(n_estimators = 40, 
                                                   criterion='gini')
        elif classe['posfixo'] == '_3':
            classificador = RandomForestClassifier(n_estimators = 100, 
                                                   criterion='gini')
        elif classe['posfixo'] == '_4':
            classificador = RandomForestClassifier(n_estimators = 20, 
                                                   criterion='entropy')
        elif classe['posfixo'] == '_5':
            classificador = RandomForestClassifier(n_estimators = 40, 
                                                   criterion='entropy')
        elif classe['posfixo'] == '_6':
            classificador = RandomForestClassifier(n_estimators = 100, 
                                                   criterion='entropy')
    elif classe['name'] == 'MLPClassifier':
        
        if classe['posfixo'] == '_1':
            classificador = MLPClassifier(max_iter=1000,
                                      verbose=False,
                                      tol = 0.00001,
                                      hidden_layer_sizes=(8, 8),
                                      solver='adam',
                                      activation='relu')
        elif classe['posfixo'] == '_2':
            classificador = MLPClassifier(max_iter=2000,
                                      verbose=False,
                                      tol = 0.00001,
                                      hidden_layer_sizes=(8, 8),
                                      solver='sgd',
                                      activation='logistic')

    elif classe['name'] == 'SVM':
        
        if classe['posfixo'] == '_1':
            classificador = SVC(kernel='linear',
                                C=1.0)
        elif classe['posfixo'] == '_2':
            classificador = SVC(kernel='linear',
                                C=2.0)
        elif classe['posfixo'] == '_3':
            classificador = SVC(kernel='rbf',
                                C=1.0)
        elif classe['posfixo'] == '_4':
            classificador = SVC(kernel='rbf',
                                C=2.0)

    
    classificador.fit(x_train, y_train)    
    previsoes = classificador.predict(x_test)
    
    return classificador, previsoes
        