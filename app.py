#Importação do FrameWork Flask e algumas bibliotecas, para a estruturação da aplicação de visualização do funcionamento do modelo.

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

#Carga do modelo preditivo 
model= pickle.load(open('models/Modelo_preditivo.pkl', 'rb'))

@app.route('/', methods = ["GET", "POST"])
def index():
    
    if request.method == 'POST':
        Age= int(request.form.get('Age'))
        Gender= int(request.form.get('Gender'))
        Hypertension= int(request.form.get('Hypertension'))
        Family_History= int(request.form.get('Family_History'))
        caracteristicas= np.array([[Age, Gender, Hypertension, Family_History]])
        predicao= model.predict(caracteristicas)
        Heart_Attack_Risk = {0 : 'Low', 1 : 'Moderate', 2 : 'High'}
        resultado= Heart_Attack_Risk.get(predicao[0])

        return render_template('index.html', predicao = resultado)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug= True)