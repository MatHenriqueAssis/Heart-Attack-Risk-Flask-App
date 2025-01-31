#Importação das bibliotecas e módulos bases do projeto 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import classification_report, accuracy_score 
import pickle

#Criando um dataframe e retirando do arquivo CSV aquelas colunas que não poussuem grau de importancia. 

df = pd.read_csv("instances/heart_attack_risk_dataset.csv", 
                usecols= ['Age', 'Gender', 'Hypertension','Family_History','Heart_Attack_Risk'],
                nrows= 1000,
                sep=",",
                encoding="latin1"
                )

def categorize_age(Age):
    if Age < 60:
        return 'Adulto'
    else:
        return 'Idoso'

df['Age'] = df['Age'].apply(categorize_age)

def categorize_hypertension(Hypertension):
    if Hypertension == 0:
        return 'No'
    elif Hypertension == 1:
        return 'Yes'
df['Hypertension'] = df['Hypertension'].apply(categorize_hypertension)


'''
print(df.columns)
print(df.head())
print(df.isnull().sum()) 
print(df.duplicated().sum()) c
'''

'''
#Criando alguns gravicos para melhor visualização das correlações entre os dados.

sns.countplot(data = df, x = 'Heart_Attack_Risk', hue = 'Gender')
plt.show()

sns.countplot(data = df, x = 'Heart_Attack_Risk', hue = 'Hypertension')
plt.show()

sns.countplot(data = df, x = 'Heart_Attack_Risk', hue = 'Family_History')
plt.show()
'''


#Fazendo o mapeamento numerico para incializar o processo de treinamento do modelo.

df['Age'] = df['Age'].map({'Adulto' : 0, 'Idoso' : 1})
df['Hypertension'] = df['Hypertension'].map({'Não' : 0, 'Sim' : 1})
df['Gender'] = df['Gender'].map({'Famale' : 0, 'Male' : 1})
df['Heart_Attack_Risk'] = df['Heart_Attack_Risk'].map({'Low' : 0, 'Moderate' : 1, 'High' : 2})


print(df.dtypes)

#Declarando as variaveis que serão utilizadas para o treinamento do modelo.

'''
X= df.drop("Heart_Attack_Risk", axis= 1)
Y= df["Heart_Attack_Risk"]


#Inciando o Treinento do Modelo. Para isso, utizei os processos de DecisionTree e também o de RandomForest.

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=0)


DT = DecisionTreeClassifier() 
DT.fit(X_train, y_train) 
DT_pred = DT.predict(X_test) 



RF = RandomForestClassifier(n_estimators=100) 
RF.fit(X_train, y_train) 
RF_pred = RF.predict(X_test) 

#Retornando os resultados dos treinamentos, para analisar qual está retornando uma melhor porcentagem de acertos.
print("Árvore de Decisão:",accuracy_score(y_test, DT_pred)) 
print("Random Forest:",accuracy_score(y_test, RF_pred))

plt.figure(figsize=(12, 8))
caracteristicas_names = X.columns.tolist()
rotulo_name = Y.unique().astype(str).tolist()
plot_tree(DT, 
    feature_names=caracteristicas_names,  
    class_names=rotulo_name,    
    filled=True,                      
    rounded=True)                     
plt.show()

with open('Modelo_preditivo.pkl', 'wb') as f:
    pickle.dump(DT, f)
    '''