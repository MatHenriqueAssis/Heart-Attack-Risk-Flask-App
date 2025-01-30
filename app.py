#Importação do FrameWork Flask e algumas bibliotecas, para a estruturação da aplicação de visualização do funcionamento do modelo.

from flask import Flask, request, render_template
import pickle
import numpy as np