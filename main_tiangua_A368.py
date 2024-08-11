#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CÓDIGO: MAIN DO PROJETO
# PRÉ PROCESSAMENTO: labelencoder; onehotencoder; escalonamento;
# CLASSIFICADORES: RandomForest, Multilayer Perceptron
# PIBIC 2020.2-2021.1 - VERSÃO 1.0
# AUTOR: Ananias Caetano de Oliveira
# ORIENTADOR: Rhyan Ximenes de Brito
# INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DO CEARÁ
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Importando bibliotecas
import pandas as pd
import numpy as np # computação numérica
import preprocessamento as process # arquivo python com funções de pré-processamento de dados
import classificadores
from save_results import FileResult
from save_results import Status
#from sklearn.metrics import classification_report # responsavel por retornar as principais informacoes sobre os resultados
#import scikitplot as skplt # responsavel pelo plot

metodos = {
    1: {'name': 'simpleImputer_underSampler_zscore'}
}

classes = {
    1: {'name': 'Random_Forest', 'posfixo': '_1', 'parameter': 'n_estimators: 20 | criterion = gini'},
    2: {'name': 'Random_Forest', 'posfixo': '_2', 'parameter': 'n_estimators: 40 | criterion = gini'},
    3: {'name': 'Random_Forest', 'posfixo': '_3', 'parameter': 'n_estimators: 100 | criterion = gini'},
    4: {'name': 'Random_Forest', 'posfixo': '_4', 'parameter': 'n_estimators: 20 | criterion = entropy'},
    5: {'name': 'Random_Forest', 'posfixo': '_5', 'parameter': 'n_estimators: 40 | criterion = entropy'},
    6: {'name': 'Random_Forest', 'posfixo': '_6', 'parameter': 'n_estimators: 100 | criterion = entropy'},
    7: {'name': 'MLPClassifier', 'posfixo': '_1', 'parameter': 'max_iter= 1000 | tol = 0.00001 | hidden_layer_sizes=(8, 8) | solver = adam | activation = relu'},
    8: {'name': 'MLPClassifier', 'posfixo': '_2', 'parameter': 'max_iter= 2000 | tol = 0.00001 | hidden_layer_sizes=(8, 8) | solver = sgd | activation = logistic'},
    9: {'name': 'SVM',  'posfixo': '_1', 'parameter': 'kernel:linear | C=1.0'},
    10: {'name': 'SVM', 'posfixo': '_2', 'parameter': 'kernel:linear | C=2.0'},
    11: {'name': 'SVM', 'posfixo': '_3', 'parameter': 'kernel:rbf | C=1.0'},
    12: {'name': 'SVM', 'posfixo': '_4', 'parameter': 'kernel:rbf | C=2.0'},
}

# Quantidade de execuções de cada classificador, método e dataset
MAX_RODADAS = 10
TAM_TESTE = 0.25

base = pd.read_csv('dataset/tiangua(A368).csv', encoding = 'UTF-8', sep=';', decimal=",")

for kme in classes.keys():
    classe = classes[kme]
    
    for km in metodos.keys():
        metodo = metodos[km]
    
        # Instancia um objeto FileResult para salvar os resultados
        file = FileResult('resultados/' + classe['name'], 'resultados_' + classe['name'] + classe['posfixo'] + '_' + metodo['name']+ '.txt', MAX_RODADAS)
    
        if file.status == Status.NOVO:
            file.write("DATASET = TIANGUA (A368)" + " | CLASSIFICADOR = " + classe['name'] + " | METODO = " + metodo['name'])
            file.write("PARAMETROS: "+ classe['parameter'] + "\n")
    
##################################################################################################################################
        # FASE DE PRÉ-PROCESSAMENTO
        X_processed, Y = process.pre_processa(base, metodo['name'])

##################################################################################################################################    
        print('\nTipo X_processed: ',  type(X_processed))
    
        # Execução do treinamento e classificação
        if file.status == Status.NOVO or file.status == Status.FALTA_VALIDAR:
            medias_acertos = []
            matriz_confusao = []
        else:
            medias_acertos = file.get_medias()
        
        if file.status == Status.CONCLUIR_RODADAS:
           rodada = file.get_rodadas_finalizadas()
        elif file.status == Status.NOVO:
            rodada = 0
        else:
            rodada = MAX_RODADAS
    
        while rodada < MAX_RODADAS:
            file.write("RODADA = " + str(rodada + 1))
            
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(X_processed, Y, test_size=TAM_TESTE)
    
###################################################################################################################################
            
            # ETAPA DE CLASSIFICAÇÃO
            classificador, previsoes = classificadores.CLASSIFICA(x_train, y_train, x_test, classe)
            
###################################################################################################################################
            # DEFINICAÇÃO DE RESULTADOS
            from sklearn.metrics import confusion_matrix, accuracy_score
            medias_acertos.append(accuracy_score(y_test, previsoes, normalize=True)*100)
            matriz_confusao.append(confusion_matrix(y_test, previsoes))
            
            file.write("Média de acerto = " + str( medias_acertos[rodada] ) + "\n")
            rodada = rodada + 1
            
        if file.status != Status.FALTA_VALIDAR:
            #OPERAÇÃO DE MEDIA DE ACERTOS
            media_final = np.mean(medias_acertos)
            # OPERAÇÃO DE MEDIA DA MATRIZ DE CONFUSAO
            media_confusao_final = np.mean(matriz_confusao, axis = 0)
                
            desvio_padrao = np.std(medias_acertos)
                            
            file.write("Média Final = " + str( media_final ))
            file.write("Menor média de acerto = " + str( np.amin(medias_acertos)))
            file.write("Maior média de acerto = " + str( np.amax(medias_acertos)))
            file.write("Mediana média de acerto = " + str( np.median(medias_acertos)))
            file.write("Desvio Padrão = " + str( desvio_padrao ) + "\n")
            file.write("Matriz de Confusão:")
            file.write("    0        1")
            file.write("0  "+str(media_confusao_final[0][0])+"   "+str(media_confusao_final[0][1]))
            file.write("1  "+str(media_confusao_final[1][0])+"   "+str(media_confusao_final[1][1]))
        
        # imprimir relatório de classificação
        #print("\nRelatório de Classificação:\n", classification_report(y_test, previsoes, digits=4))
        # plotar a matrix de confusão
        #skplt.metrics.plot_confusion_matrix(y_test, previsoes, normalize=True)
