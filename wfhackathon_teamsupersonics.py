# Databricks notebook source
# MAGIC %md
# MAGIC ## Required libraries pip commands

# COMMAND ----------

!pip install xlsxwriter
!pip install ethnicolr
!pip install requests
!pip install genderize
!pip install azure-storage-blob
!pip install openpyxl 
!pip install h5py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing all the libraries

# COMMAND ----------

import asyncio
import sys
import time
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from ethnicolr import pred_wiki_name, pred_fl_reg_ln_five_cat
from json import loads
import requests
from genderize import Genderize
from tensorflow.keras.models import load_model
import h5py
import asyncio
import numpy as np
import pandas as pd
from ethnicolr import pred_wiki_name, pred_fl_reg_ln_five_cat
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import h5py

# COMMAND ----------

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## common methods to read data from different source

# COMMAND ----------


def read_from_blob_excel(config):
    connectionstring = config.get("storage_connection")
    container = config.get("input_container")
    blob = config.get("input_file")
    
    blob_service_client_instance = BlobServiceClient.from_connection_string(connectionstring)

    blob_client_instance = blob_service_client_instance.get_blob_client(container, blob, snapshot=None)
    with open(blob, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)

        
        
async def extract_gender_fromname(name, api_key):
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


async def extract_origin_fromname(name, api_key):
    region = None
    base_url = f"https://v2.namsor.com/NamSorAPIv2/api2/json/country/{name}"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": f'{api_key}'
    }
    response = requests.request("GET", base_url, headers=headers)
    if response.status_code not in range(200, 299):
        return None, None
    try:
        region = response.json()['subRegion']
    except:
        pass
    return region


async def genderize(name):
    return Genderize().get([name])[0]['gender']





# COMMAND ----------

# MAGIC %md
# MAGIC ## Gender detection and ethnicity detection common methods

# COMMAND ----------



def get_gender(df, key):
    logger.info("<<<---- Gender prediction is started ---->>>")
    df['executive1_gender'] = df.apply(lambda row: asyncio.run(extract_gender_fromname(row['executivecontact1_name'], key)), axis=1)
    df['executive1_gender'].mask(df['executivecontact1_name'].isnull(), np.nan, inplace=True)
    logger.info("<<<---- Gender prediction for executive1 completed ---->>>")

    df['executive2_gender'] = df.apply(lambda row: asyncio.run(extract_gender_fromname(row['executivecontact2_name'], key)), axis=1)
    df['executive2_gender'].mask(df['executivecontact2_name'].isnull(), np.nan, inplace=True)
    logger.info("<<<---- Gender prediction for executive2 completed ---->>>")

    cols = [col for col in df if col.endswith('gender')]
    cols.append("dunsnum")
    gender_df = df.loc[:, cols]
    gender_df['gender'] = gender_df.apply(flag_gender, axis = 1)
    logger.info("<<<---- Gender prediction is complete ---->>>")
    return gender_df


def flag_gender(df):
    if (df['executive1_gender'] == 'Female') or (df['executive2_gender'] == 'Female'):
        return 1
    elif (df['executive1_gender'] == 'Male') or (df['executive2_gender'] == 'Male'):
        return 0
    else:
        return np.nan


async def get_minor_detect(df):
    # processing data for minority-owned
    logger.info(f"<<<-- minority prediction  started ------->>>")
    diverse_df1 = pred_wiki_name(df, lname_col='executivecontact1_lname', fname_col='executivecontact1_fname', num_iter=50)
    logger.info(f"<<<-- minority prediction for executive1 completed ------->>>")

    diverse_df2 = pred_wiki_name(df, lname_col='executivecontact2_lname', fname_col='executivecontact2_fname', num_iter=50)
    logger.info(f"<<<-- minority prediction for executive2 completed ------->>>")

    diverse_df1["executive1_race"] = diverse_df1["race"].str.split(',').str[-1]
    diverse_df1['executive1_race'].mask(diverse_df1['executivecontact1_name'].isnull(), np.nan, inplace=True)
    diverse_df1 = diverse_df1.loc[:, ["dunsnum", "executive1_race"]]

    diverse_df2["executive2_race"] = diverse_df2["race"].str.split(',').str[-1]
    diverse_df2['executive2_race'].mask(diverse_df2['executivecontact2_name'].isnull(), np.nan, inplace=True)
    diverse_df2 = diverse_df2.loc[:, ["dunsnum", "executive2_race"]]

    diverse_df = pd.merge(diverse_df1, diverse_df2, how='inner')
    diverse_df[["executive1_race", "executive2_race"]] = diverse_df[["executive1_race", "executive2_race"]].replace(to_replace=["British", "French", 'Italian', 'Germanic', 'Nordic','EastEuropean','Jewish'], value='')
    diverse_df["race"] = diverse_df[['executive1_race', 'executive2_race']].apply(lambda x: ','.join(x.replace(np.nan,'')), axis=1)
    logger.info(f"<<<-- minority prediction for executive completed ------->>>")
    return diverse_df



def get_region(df, config):
    logger.info("<<<---- region prediction is started ---->>>")
    df['executive1_origin'] = df.apply(lambda row: asyncio.run(extract_origin_fromname(row['executivecontact1_name'], config.get("region_key"))), axis=1)
    df['executive1_origin'].mask(df['executivecontact1_name'].isnull(), np.nan, inplace=True)
    logger.info("<<<---- origin prediction for executive1 completed ---->>>")
    df['executive2_origin'] = df.apply(lambda row: asyncio.run(extract_origin_fromname(row['executivecontact2_name'], config.get("region_key"))), axis=1)
    df['executive2_origin'].mask(df['executivecontact2_name'].isnull(), np.nan, inplace=True)
    logger.info("<<<---- origin prediction for executive2 completed ---->>>")
    cols = [col for col in df if col.endswith('origin')]
    cols.append("dunsnum")
    origin_df = df.loc[:, cols]
    origin_df[["executive1_origin", "executive2_origin"]] = origin_df[["executive1_origin", "executive2_origin"]].replace(to_replace=str(config.get("region_list")).split(","), value='')
    origin_df['executive2_origin'].mask(origin_df['executive2_origin'].isnull(), '', inplace=True)
    origin_df["origin"] = origin_df[['executive1_origin', 'executive2_origin']].apply(lambda x: ','.join(x.replace(np.nan,'')), axis=1)
    logger.info("<<<---- Gender prediction is complete ---->>>")
    return origin_df


def get_minor_detect_cat(df):
    # processing data for minority-owned
    logger.info(f"<<<-- minority prediction  started ------->>>")
    diverse_df1 = pred_fl_reg_ln_five_cat(df, 'executivecontact1_lname', num_iter=50)
    logger.info(f"<<<-- minority prediction for executive1 completed ------->>>")

    diverse_df2 = pred_fl_reg_ln_five_cat(df, 'executivecontact2_lname', num_iter=50)
    logger.info(f"<<<-- minority prediction for executive2 completed ------->>>")

    diverse_df1["executive1_race"] = diverse_df1["race"].str.split('_').str[-1]
    diverse_df1['executive1_race'].mask(diverse_df1['executivecontact1_name'].isnull(), np.nan, inplace=True)
    diverse_df1 = diverse_df1.loc[:, ["dunsnum", "executive1_race"]]

    diverse_df2["executive2_race"] = diverse_df2["race"].str.split('_').str[-1]
    diverse_df2['executive2_race'].mask(diverse_df2['executivecontact2_name'].isnull(), np.nan, inplace=True)
    diverse_df2 = diverse_df2.loc[:, ["dunsnum", "executive2_race"]]

    diverse_df = pd.merge(diverse_df1, diverse_df2, how='inner')
    diverse_df[["executive1_race", "executive2_race"]] = diverse_df[["executive1_race", "executive2_race"]].replace(to_replace=["white","other"], value='')
    diverse_df["race"] = diverse_df[['executive1_race', 'executive2_race']].apply(lambda x: ','.join(x.replace(np.nan,'')), axis=1)
    logger.info(f"<<<-- minority prediction for executive completed ------->>>")
    return diverse_df


def detect_gender(df, model):
    logger.info("<<<---- Gender prediction is started ---->>>")
    gender_df1 = get_gender_diversity(df, 'executivecontact1_name', model)
    gender_df1.rename(columns={'gender': 'executive1_gender'}, inplace=True)
    gender_df1['executive1_gender'].mask(gender_df1['executivecontact1_name'] == 'nan', np.nan, inplace=True)
    logger.info("<<<---- Gender prediction for executive1 completed ---->>>")

    gender_df2 = get_gender_diversity(df, 'executivecontact2_name', model)
    gender_df2.rename(columns={'gender': 'executive2_gender'}, inplace=True)
    gender_df2['executive2_gender'].mask(gender_df2['executivecontact2_name'] == 'nan', np.nan, inplace=True)
    logger.info("<<<---- Gender prediction for executive2 completed ---->>>")

    gender_df = pd.merge(gender_df1, gender_df2, how='inner')

    cols = [col for col in df if col.endswith('gender')]
    cols.append("dunsnum")
    cols.append("probability")
    gender_df = gender_df.loc[:, cols]
    gender_df['gender'] = gender_df.apply(flag_gender, axis = 1)
    logger.info("<<<---- Gender prediction is complete ---->>>")
    return gender_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM model for gender detection

# COMMAND ----------



def preprocess(names_df, train=True):
    names_df['name'] = names_df['name'].str.lower()
    #names_df['name'] = [list(name) for name in names_df['name']]

    names_df['name'] = (names_df['name'].dropna().apply(lambda x: [item for item in x]))
    logger.info("<<<--- Split individual characters  ---->>>>")

    name_length = 50
    names_df['name'] = [(name + [' ']*name_length)[:name_length]
        for name in names_df['name']
                        ]
    logger.info("<<<--- Pad names with spaces to make all names same length  ---->>>>")

    names_df['name'] = [
        [
            max(0.0, ord(char)-96.0)
            for char in name
        ]
        for name in names_df['name']
    ]
    logger.info("<<<--- Encode Characters to Numbers  ---->>>>")

    if train:
        logger.info("<<<--- Encode Gender to Numbers  ---->>>>")
        names_df['gender'] = [
            0.0 if gender=='F' else 1.0
            for gender in names_df['gender']
        ]

    return names_df


def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def get_gender_diversity(df, name, model):

    df.rename(columns={f'{name}': 'name'}, inplace=True)
    logger.info("<<<--- Rename columns  ---->>>>")

    # Convert to dataframe
    names = df['name'].values.tolist()
    # Preprocess
    pred_df = preprocess(df, train=False)

    # Predictions
    result = model.predict(np.asarray(
        pred_df['name'].values.tolist())).squeeze(axis=1)

    logger.info("<<<--- predictions based on the LSTM model  ---->>>>")
    pred_df['gender'] = [
        'Male' if logit > 0.5 else 'Female' for logit in result
    ]
    logger.info("<<<--- deriving the gender based on model prediction   ---->>>>")

    pred_df['probability'] = [
        logit if logit > 0.5 else 1.0 - logit for logit in result
    ]
    logger.info("<<<--- filtering based on the probability  ---->>>>")

    # Format the output
    pred_df['name'] = names
    pred_df.rename(columns={'name': f'{name}'}, inplace=True)
    pred_df['probability'] = pred_df['probability'].round(2)
    pred_df.drop_duplicates(inplace=True)
    return pred_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Invoker

# COMMAND ----------

start_time = time.time()

blob_service_client_instance = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=codesanddoc;AccountKey=ydz3XndrMKSX4aYFOc/TQfr8C6gj2/jQddc75+7FNshlqIGf5bieFfJvDg0M6YFCvGHYwOcU/CKm+AStywykNA==;EndpointSuffix=core.windows.net")
blob_client_instance = blob_service_client_instance.get_blob_client('input', 'connection_config.json', snapshot=None)
blob_data = blob_client_instance.download_blob()

blob_client_h5 = blob_service_client_instance.get_blob_client('input', 'boyorgirl.h5', snapshot=None)
blob_datah5 = blob_client_h5.download_blob()

import json
config = json.loads(blob_data.readall())

read_from_blob_excel(config)

connectionstring = config.get("storage_connection")
input_container = config.get("input_container")
input_blob = config.get("input_file")
output_container = config.get("ouput_container")
output_blob = config.get("output_file")
start = config.get("start_val")
end = config.get("end_val")


pred_model = load_model(h5py.File("/dbfs/FileStore/shared_uploads/sunilpandey.de@wfhackathon2022.onmicrosoft.com/boyorgirl.h5"))


df = process_excel_file(input_blob, int(start), int(end))
df.columns= df.columns.str.lower()
df=df.apply(lambda x: x.astype(str).str.lower())
logger.info("<<<------ Reading the source file  ------->>>")


#gender_df = get_gender(df, config.get("api_key"))
gender_df = detect_gender(df, pred_model)
logger.info("<<<------ Gender detection completed  ------->>>")


#minor_df = asyncio.run(get_minor_detect(df))
#minor_df = get_region(df, config)
df = df.replace('nan', np.nan, regex=True)
minor_df = get_minor_detect_cat(df)
logger.info("<<<------ Minority detection completed  ------->>>")


final_df = pd.merge(pd.merge(df, gender_df, on='dunsnum'), minor_df, on='dunsnum', suffixes=('', '_y'))
logger.info("<<<------ Intermediate data joined ------->>>")
final_df.drop(final_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
final_df = final_df.loc[:, ]

xl_writer = pd.ExcelWriter(output_blob, engine='xlsxwriter')
final_df.to_excel(xl_writer, index=False)
xl_writer.save()

service = BlobServiceClient.from_connection_string(connectionstring)
container_client = service.get_container_client(output_container)

with open(output_blob, "rb") as data:
    container_client.upload_blob(name=output_blob+str(start)+"_"+str(end)+".xlsx", data=data, overwrite=True)

logger.info("<<<------ Final dataset written to excel ------->>>")
logger.info("<<<--- %s seconds --->>>" % (time.time() - start_time))

