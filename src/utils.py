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