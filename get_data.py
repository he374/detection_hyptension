import os

os.environ['TCL_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import requests
from io import StringIO


eeg1 = "BIS/EEG1_WAV"
eeg2 = "BIS/EEG2_WAV"
map = "Solar8000/ART_MBP"
bis = "BIS/BIS"
dap = "Solar8000/ART_DBP"
file_pathlist = [r"C:\Users\Admin\Desktop\pression_sujet.txt",
                 r"C:\Users\Admin\Desktop\bis_données.txt"]


def general_data():
    data = []
    file_path = r"C:\Users\Admin\Desktop\RZ.txt"
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\d+),([^,]+),([0-9a-f]+)', line)
            if match:
                id_value = match.group(1)
                param_name = match.group(2)
                tid = match.group(3)
                data.append({'ID': id_value, 'Parameter': param_name, 'TID': tid})

    df = pd.DataFrame(data)
    return df


def select_data(parameter):
    data = general_data()
    filter_data = (data['Parameter'] == parameter)
    return data[filter_data]


def select_tid(identification, parameter):
    result = select_data(parameter)
    tid_voulu = result[result['ID'] == identification]['TID'].values
    return tid_voulu if len(tid_voulu) > 0 else None


def write_on_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
    return


def create_dataframe(file_path, parameter):
    df = pd.read_csv(file_path, sep=",", skiprows=1, names=["TIME", parameter])
    return df


def get_tid_data(valeur_tid):
    data_tid = None
    svaleur_tid = valeur_tid[0]
    api_url = f'https://api.vitaldb.net/{svaleur_tid}'
    response = requests.get(api_url)
    data_tid = response.text
    return data_tid


def data_map(id):
    tv = select_tid(id, map)
    if tv is None:
        return None
    df = get_tid_data(tv)
    datam = pd.read_csv(StringIO(df))
    return datam


def data_bis(id):
    tv = select_tid(id, bis)
    if tv is None:
        return None
    df = get_tid_data(tv)
    datab = pd.read_csv(StringIO(df))
    return datab


def hypotension(id):
    label_hypotension = False
    hypotension_threshold = 65

    datam = data_map(id)
    datab = data_bis(id)

    if datab is None:
        return None
    if datam is None:
        return None

    datab = datab[datab[bis] != 0].reset_index(drop=True)

    start_index = None
    end_index = None

    for i in range(1, len(datab)):
        if datab[bis].iloc[i - 1] > 60 >= datab[bis].iloc[i]:
            start_index = i
            break

    for i in range(len(datab) - 1, 0, -1):
        if datab[bis].iloc[i - 1] < 60 <= datab[bis].iloc[i]:
            end_index = i
            break

    if start_index is None and end_index is None:
        return None
    elif start_index is None:
        data = datam[datam.index <= end_index]
    elif end_index is None:
        data = datam[datam.index >= start_index]
    else:
        data = datam[(datam.index >= start_index) & (datam.index <= end_index)]

    datab = datab[(datab.index >= start_index) & (datab.index <= end_index)]

    data = data.copy()
    data['Hypotension'] = data[map] < hypotension_threshold

    episodes = []
    start = None
    for idx, row in data.iterrows():
        if row['Hypotension']:
            if start is None:
                start = idx
        elif start is not None:
            episodes.append((start, idx - 1))
            start = None

    if start is not None:
        episodes.append((start, len(data) - 1))

    for start, end in episodes:
        duration = end - start
        if duration > 60:
            label_hypotension = True

    return label_hypotension




def labelisation():
    data = []
    for i in range(1, 100):
        id = str(i)
        label = hypotension(id)
        data.append({"ID": id, "Label": label})
    df = pd.DataFrame(data)
    return df


def data_eeg1(id):
    tv = select_tid(id, eeg1)
    if tv is None:
        return None
    df = get_tid_data(tv)
    data_e1 = pd.read_csv(StringIO(df))
    return data_e1


def data_eeg2(id):
    tv = select_tid(id, eeg2)
    if tv is None:
        return None
    df = get_tid_data(tv)
    data_e2 = pd.read_csv(StringIO(df))
    return data_e2





def indicateur(id):
    datab = data_bis(id)

    if datab is None:
        return None

    datab = datab[datab[bis] != 0].reset_index(drop=True)

    start_index = None
    end_index = None

    for i in range(1, len(datab)):
        if datab[bis].iloc[i - 1] > 60 >= datab[bis].iloc[i]:
            start_index = i
            break

    for i in range(len(datab) - 1, 0, -1):
        if datab[bis].iloc[i - 1] < 60 <= datab[bis].iloc[i]:
            end_index = i
            break

    return start_index, end_index


def data_bis_extracted(id):
    datab = data_bis(id)

    start_index, end_index = indicateur(id)
    if start_index is None and end_index is None:
        return None
    elif start_index is None:
        data = datab[datab.index <= end_index]
    elif end_index is None:
        data = datab[datab.index >= start_index]
    else:
        data = datab[(datab.index >= start_index) & (datab.index <= end_index)]

    return data


def data_map_extracted(id):
    datam = data_map(id)
    if datam is None:
        return None

    start_index, end_index = indicateur(id)

    if start_index is None and end_index is None:
        return None
    elif start_index is None:
        data = datam[datam.index <= end_index]
    elif end_index is None:
        data = datam[datam.index >= start_index]
    else:
        data = datam[(datam.index >= start_index) & (datam.index <= end_index)]

    return data


def show_save_plt(id, parameter, seuil):
    if parameter == map:
        df = data_map_extracted(id)
        df[parameter] = df[parameter].apply(lambda x: max(x, 0))
        df = df[df[parameter] != 0].reset_index(drop=True)
    elif parameter == bis:
        df = data_bis_extracted(id)
    else:
        return None
    
    

    if df is not None:
        df['Hypotension'] = df[parameter] < seuil

        episodes = []
        start = None
        for idx, row in df.iterrows():
            if row['Hypotension']:
                if start is None:
                    start = idx
            elif start is not None:
                episodes.append((start, idx - 1))
                start = None

        if start is not None:
            episodes.append((start, len(df) - 1))

        plt.figure(figsize=(10, 6))

        for start, end in episodes:
            duration = end - start
            color = "red" if duration > 60 else "blue"
            plt.plot(
                df["Time"].iloc[start:end + 1],
                df[parameter].iloc[start:end + 1],
                marker="o",
                color=color
            )

        plt.plot(
            df.loc[~df['Hypotension'], "Time"],
            df.loc[~df['Hypotension'], parameter],
            marker="o",
            color="blue"
        )

        plt.title(f"Évolution de {parameter} au cours du Temps")
        plt.xlabel("Temps")
        plt.ylabel(parameter)
        plt.grid(True)
        plt.legend()

        chemin = f"plots\\plot_{id}_{parameter}.png"
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        plt.savefig(chemin)
        plt.show()


    else:
        return None
    

def save_all(parameter):
    if parameter == map:
        x = 65
    else:
        x = None
    for i in range(1, 100):
        id = str(i)
        l = show_save_plt(id, parameter, seuil=x)

    return True





