import asyncio
import logging
import sys

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

from data_processing.baselogger import logger
from data_processing.common_methods import process_excel_file, json_config_parser
from data_processing.detect_gender_diversity import  detect_gender, get_minor_detect_cat


def main():
    import time
    start_time = time.time()
    from tensorflow.keras.models import load_model
    pred_model = load_model('boyorgirl.h5')

    config = json_config_parser("connection_config.json")
    df = process_excel_file(config.get("input_file"), int(config.get("start_val")), int(config.get("end_val")))
    df.columns = df.columns.str.lower()
    df = df.apply(lambda x: x.astype(str).str.lower())
    logger.info("<<<------ Reading the source file  ------->>>")

    gender_df = detect_gender(df, pred_model)
    logger.info("<<<------ Gender detection completed  ------->>>")

    df = df.replace('nan', np.nan, regex=True)
    minor_df = get_minor_detect_cat(df, config)
    logger.info("<<<------ Minority detection completed  ------->>>")

    final_df = pd.merge(pd.merge(df, gender_df, on='dunsnum'), minor_df, on='dunsnum', suffixes=('', '_y'))
    logger.info("<<<------ Intermediate data joined ------->>>")
    final_df.drop(final_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    final_df = final_df.loc[:, ]

    xl_writer = pd.ExcelWriter(config.get("output_file"), engine='xlsxwriter')
    final_df.to_excel(xl_writer, index=False)
    xl_writer.save()

    service = BlobServiceClient.from_connection_string(config.get("storage_connection"))
    container_client = service.get_container_client(config.get("ouput_container"))

    with open(config.get("output_file"), "rb") as data:
        container_client.upload_blob(
            name="Hackathon_Data_MinorityWomenOwned_2022_final_" + str(config.get("start_val")) + "_" + str(config.get("end_val")) + ".xlsx", data=data, overwrite=True)

    logger.info("<<<------ Final dataset written to excel ------->>>")
    logger.info("<<<--- %s seconds --->>>" % (time.time() - start_time))


if __name__ == '__main__':
    main()
