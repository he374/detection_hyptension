import numpy as np
import pandas as pd
from get_data import data_bis_extracted , data_map_extracted, indicateur
import matplotlib.pyplot as plt
from io import StringIO
import os
import csv

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

map = "Solar8000/ART_MBP"
bis = "BIS/BIS"



def nbr_samples(id):
    start, end = indicateur(id) 
    nbr = (end - start)/300
    return int(nbr)
def nbr_samples_total(id):
    nbr = nbr_samples(id)
    return int(5*(nbr-1))

def sampling_data_time(id, index):
    
    var1 = data_bis_extracted(id)  
    var2 = data_map_extracted(id)  
    start, end = indicateur(id)  
    
    
    var = index*60 + start

    
    varbis_list = []  
    varmap_list = [] 
    
    for i in range(var, var + 300):
        if i >= len(var1) or i >= len(var2):  
            break
        varbis_list.append(var1.loc[i, "Time"])  
        varmap_list.append(var2.loc[i, "Time"])

    
    return varbis_list, varmap_list

def sampling_data_data(id, index):
    
    var1 = data_bis_extracted(id)  
    var2 = data_map_extracted(id)  
    start, end = indicateur(id)  
    
    
    var = index*60 + start

    
    varbis_list = []  
    varmap_list = [] 
    
    for i in range(var, var + 300):
        if i >= len(var1) or i >= len(var2):  
            break
        varbis_list.append(var1.loc[i, bis])  
        varmap_list.append(var2.loc[i, map])

    
    return varbis_list, varmap_list

def ident_hypotension(datab,datam):

    label_hypotension = False
    threshold = 65

    

    if datab is None or datam is None:
        return None

    
    
    
    
    
    window_size = 30

    

    
    for i in range(len(datam) - window_size + 1):
        
        if all(value < threshold for value in datam[i:i + window_size]):
            
            label_hypotension = True
    
    return label_hypotension


    


def min_label_time(id, index):

    varbis_list = []  
    varmap_list = [] 
     
    var1 = data_bis_extracted(id)  
    var2 = data_map_extracted(id)
    start, end = indicateur(id)

    var = index*60 + start + 300 

    for i in range(var, var + 60):
        if i >= len(var1) or i >= len(var2):  
            break
        varbis_list.append(var1.loc[i, "Time"])  
        varmap_list.append(var2.loc[i, "Time"])

    return varbis_list,varmap_list


def min_label_data(id, index):

    varbis_list = []  
    varmap_list = [] 
     
    var1 = data_bis_extracted(id)  
    var2 = data_map_extracted(id)
    start, end = indicateur(id)

    var = index*60 + start + 300 

    for i in range(var, var + 60):
        if i >= len(var1) or i >= len(var2):  
            break
        varbis_list.append(var1.loc[i, bis])  
        varmap_list.append(var2.loc[i, map])

    return varbis_list,varmap_list




def gener_sdf_poly(datab_time, datab_data, datam_time, datam_data):
    
    datab_time = np.array(datab_time)
    datam_time = np.array(datam_time)
    datab_data = np.array(datab_data)
    datam_data = np.array(datam_data)

    
    if datab_time.size == 0 or datam_time.size == 0 or datab_data.size == 0 or datam_data.size == 0:
        print("Les données d'entrée sont vides. Ignorer cette itération.")
        return None

    
    data_time = np.c_[datab_time, datam_time]
    data = np.c_[datab_data, datam_data]

    
    if data_time.shape[0] == 0 or data.shape[0] == 0:
        print("Les données combinées sont vides. Ignorer cette itération.")
        return None

    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(data_time)

   
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, data)
    intercept = lin_reg.intercept_
    coef = lin_reg.coef_

    return intercept, coef


def glob(id, index):
    
    L = []
    
    
    datab_time, datam_time = sampling_data_time(id, index)
    datab_data, datam_data = sampling_data_data(id, index)
    
    
    features = gener_sdf_poly(datab_time, datab_data, datam_time, datam_data)
    if features is None:
        return None

    
    datab_min, datam_min = min_label_data(id, index)
    label_min = ident_hypotension(datab_min, datam_min)
    label_datat = ident_hypotension(datab_data, datam_data)
    
    
    L.append(features)
    L.append(label_datat)
    L.append(label_min)

    return L


def etude_patient(id):
    All = []
    n_index = nbr_samples_total(id)
    
    for index in range(0, n_index):
        L = glob(id, index)
        if L is None:
            continue  
        All.append(L)
        print(f"Succès d'enregistrement des données de l'index : {index} du patient : {id}")

    enregistrer(All)
    return All



def enregistrer(data):
    file_path = "data.csv"
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"La liste a été sauvegardée dans le fichier : {file_path}")
    


l = etude_patient('1')

