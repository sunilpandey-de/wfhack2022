#detect gender from api call
from json import loads

import numpy as np
import pandas as pd
import requests


def extract_gender_fromname(name, api_key):
    gender = None
    base_url = f"https://v2.namsor.com/NamSorAPIv2/api2/json/gender/{name}"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": f'{api_key}'
    }
    response = requests.request("GET", base_url, headers=headers)
    if response.status_code not in range(200, 299):
        return None, None
    try:
        gender = response.json()['likelyGender']
    except:
        pass
    return gender


def split_pandas_col(df, col_to_split, delimiter):
    df1 = pd.concat([df[x].str.split(delimiter, n=1, expand=True) for x in col_to_split], axis=1, keys=col_to_split)
    df1.columns = df1.columns.map(lambda x: '_'.join((x[0], str(x[1]))))
    df1 = df1.replace({'': np.nan, None: np.nan})
    df2 = pd.concat([df, df1], axis=1)
    return df2


def process_excel_file(file, start, end):
    data_excel = pd.read_excel(file)
    data_excel = data_excel.loc[int(float(start)):int(float(end)),:]
    col_to_split = [col for col in data_excel if col.startswith('executive')]
    name_comp_df = split_pandas_col(data_excel, col_to_split, "-")
    name_comp_df.columns = [x.strip().replace('_0', '_name').replace('_1', '_job') for x in name_comp_df.columns]

    name_col_split = [col for col in name_comp_df if col.endswith('_name')]
    fname_lanme_df = split_pandas_col(name_comp_df, name_col_split, " ")
    fname_lanme_df.columns = [x.strip().replace('name_0', 'fname').replace('name_1', 'lname') for x in fname_lanme_df.columns]
    return fname_lanme_df


def json_config_parser(file_path):
    try:
        with open(file_path, 'r') as config:
            config_json = config.read().replace('/n', '')
        config_dict = loads(config_json)
        return config_dict
    except FileNotFoundError:
        config_dict = None
        return config_dict


