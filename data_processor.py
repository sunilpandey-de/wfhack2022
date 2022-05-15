import asyncio
import sys

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

from data_processing.baselogger import logger
from data_processing.common_methods import process_excel_file, json_config_parser
from data_processing.detect_gender_diversity import get_gender, get_minor_detect, get_region, detect_gender, \
    get_minor_detect_cat


def main():
    import time
    start_time = time.time()
    from tensorflow.keras.models import load_model
    pred_model = load_model('boyorgirl.h5')

    config = json_config_parser("connection_config.json")
    df = process_excel_file('Hackathon_Data_MinorityWomenOwned_2022 v1.xlsx', sys.argv[1], sys.argv[2])
    df.columns= df.columns.str.lower()
    df=df.apply(lambda x: x.astype(str).str.lower())
    # print("<<<------ Reading the source file  ------->>>")
    logger.info("<<<------ Reading the source file  ------->>>")

    #gender_df = get_gender(df, config.get("api_key"))
    gender_df = detect_gender(df, pred_model)
    # print("<<<------ gender detection completed  ------->>>")
    logger.info("<<<------ Gender detection completed  ------->>>")

    #minor_df = asyncio.run(get_minor_detect(df))
    #minor_df = get_region(df, config)
    df = df.replace('nan', np.nan, regex=True)
    minor_df = get_minor_detect_cat(df)
    # print("<<<------ minority detection completed  ------->>>")
    logger.info("<<<------ Minority detection completed  ------->>>")


    final_df = pd.merge(pd.merge(df, gender_df, on='dunsnum'), minor_df, on='dunsnum', suffixes=('', '_y'))
    logger.info("<<<------ Intermediate data joined ------->>>")
    final_df.drop(final_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    final_df = final_df.loc[:, ]

    xl_writer = pd.ExcelWriter('Hackathon_Data_MinorityWomenOwned_2022_updated.xlsx', engine='xlsxwriter')
    final_df.to_excel(xl_writer, index=False)
    xl_writer.save()

    service = BlobServiceClient.from_connection_string(config.get("storage_connection"))
    container_client = service.get_container_client(config.get("ouput_container"))

    with open("Hackathon_Data_MinorityWomenOwned_2022_updated.xlsx", "rb") as data:
        container_client.upload_blob(name="Hackathon_Data_MinorityWomenOwned_2022_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".xlsx", data=data, overwrite=True)

    # print("<<<------ final dataset written to excel ------->>>")
    logger.info("<<<------ Final dataset written to excel ------->>>")
    # print("<<<--- %s seconds --->>>" % (time.time() - start_time))
    logger.info("<<<--- %s seconds --->>>" % (time.time() - start_time))



if __name__=='__main__':
    main()
