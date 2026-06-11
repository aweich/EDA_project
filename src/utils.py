#utils.py

import matplotlib.pyplot as plt
import neurokit2 as nk 
import numpy as np
import os
import pandas as pd
from scipy import stats
import seaborn as sns
import openpyxl
print(os.getcwd())

def extract_phases(path, samplename):
    """ 
    How to use:

    samples = {"s1": '../../EDA_workshop_exercise/Data/Subjects/subject_01/Minute_Segments_Data.xlsx'}
    phases = extract_phases(samples["s1"], "s1")
    #240 4x measurements per minute (in this case minute 1)

    minutes = list(phases["s1"].keys()) # all minutes
    print(minutes)
    """
    df_raw = pd.read_excel(path)

    df_part = df_raw[
        df_raw['Task'].isin(['r1', 't1', 'r2', 't2', 'r3', 't3', 'r4'])
    ]

    fs = 4
    results = {}

    for _, row in df_part.iterrows():

        minute_eda_signal = row.iloc[2:].tolist()

        clean_neurokit, _ = nk.eda_process(
            minute_eda_signal,
            sampling_rate=fs,
            method="neurokit"
        )

        highpass = nk.eda_phasic(
            clean_neurokit["EDA_Clean"],
            sampling_rate=fs,
            method="highpass"
        )

        minute = row.iloc[0]

        results[minute] = {
            "task": row.iloc[1],
            "tonic": highpass["EDA_Tonic"].tolist(),
            "phasic": highpass["EDA_Phasic"].tolist()
        }

    return {samplename: results}


""" 
How to use:

samples = {"s1": '../../EDA_workshop_exercise/Data/Subjects/subject_02/Minute_Segments_Data.xlsx'}
phases = extract_phases(samples["s1"], "s1")
 #240 4x measurements per minute (in this case minute 1)

minutes = list(phases["s1"].keys()) # all minutes
print(minutes)
"""

from scipy.signal import welch

def compute_power_spectrum(signal, fs, segment_label=""):
    """
    Calcula el espectro de potencias usando el método de Welch.
    
    Parameters:
        signal       : array con los datos (EDA filtrada, phasic, etc.)
        fs           : frecuencia de muestreo en Hz
        
        segment_label: etiqueta para el gráfico
    
    Returns:
        freqs : array de frecuencias (Hz)
        power : densidad espectral de potencia
    """
    # nperseg: ventana de 60s para buena resolución frecuencial
    nperseg = min(len(signal), fs * 60)
    
    freqs, power = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, power


def extract_segment(signal, fs, start_min, end_min):
    """
    Extrae un segmento de la señal entre start_min y end_min (en minutos).
    
    Parameters:
        signal    : array completo de la señal
        fs        : frecuencia de muestreo en Hz
        start_min : inicio del segmento en minutos
        end_min   : fin del segmento en minutos (el intervalo es de 2 minutos)
    
    Returns:
        segment : array con los datos del intervalo
    """
    start_idx = int(start_min * 60 * fs)
    end_idx   = int(end_min   * 60 * fs)
    end_idx   = min(end_idx, len(signal))  # ← esto faltaba

    if start_idx >= len(signal):
        print(f"⚠️ Segmento {start_min}-{end_min} min fuera del rango.")
        return None

    return signal[start_idx:end_idx]