import os
import urllib.request, json
import pandas as pd
from pandas.io.json import json_normalize

known_metrics = ['fof2', 'md', 'mufd', 'foes', 'foe', 'hmf2', 'tec']

def get_data(url=os.getenv("METRICS_URI"), default_confidence=80):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())
    df = json_normalize(data)

    for field in ['cs', 'station.longitude', 'station.latitude']:
        df[field] = df[field].apply(pd.to_numeric)

    for metric in known_metrics:
        if metric in df:
            df[metric] = df[metric].apply(pd.to_numeric)

    df.cs = df.cs.apply(pd.to_numeric)
    df.loc[df.cs == -1, 'cs'] = default_confidence
    df.cs = df.cs / 100.

    df.time = pd.to_datetime(df.time)

    df.loc[df['station.longitude'] > 180, 'station.longitude'] = df['station.longitude'] - 360
    df.sort_values(by=['station.longitude'], inplace=True)

    return df

def filter(df, max_age=None, required_metrics=[], min_confidence=None):
    if max_age is not None:
       df = df.drop(df[df.time < max_age].index)

    if min_confidence is not None:
        df = df.drop(df[df.cs < min_confidence].index)

    df = df.dropna(subset=required_metrics)

    for metric in required_metrics:
        df = df.drop(df[df[metric] == 0.].index)

    return df
