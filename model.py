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
                usecols= lambda x: x not in ["Thalassemia","Exercise_Induced_Angina", "Fasting_Blood_Sugar"],
                sep=",",
                encoding="latin1"
                )

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

df['Gender'] = df['Gender'].map({'Male' : 0, 'Famele' : 1})
df['Physical_Activity_Level'] = df['Physical_Activity_Level'].map({'Low' : 0, 'Moderate' : 1, 'High' : 2})
df['Stress_level'] = df['Gender'].map({'Low' : 0, 'Moderate' : 1, 'High' : 2})
df['Chest_Pain_Type'] = df['Chest_Pain_Type'].map({'Non-anginal' : 0, 'Asymptomatic' : 1, 'Typical' : 2 })
df['Heart_Attack_Risk'] = df['Heart_Attack_Risk'].map({'Low' : 0, 'Moderate' : 1, 'High' : 2})

#Declarando as variaveis que serão utilizadas para o treinamento do modelo.

X= df.drop("Heart_Attack_Risk", axis= 1)
Y= df["Heart_Attack_Risk"]


#Inciando o Treinento do Modelo. Para isso, utizei os processos de DecisionTree e também o de RandomForest.

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=0)

'''
DT = DecisionTreeClassifier() 
DT.fit(X_train, y_train) 
DT_pred = DT.predict(X_test) 


RF = RandomForestClassifier(n_estimators=100) 
RF.fit(X_train, y_train) 
RF_pred = RF.predict(X_test) 

#Retornando os resultados dos treinamentos, para analisar qual está retornando uma melhor porcentagem de acertos.
print("Árvore de Decisão:",accuracy_score(y_test, DT_pred)) 
#print("Random Forest:",accuracy_score(y_test, RF_pred))

'''