import asyncio
import sys
import pandas as pd
from azure.storage.blob import BlobServiceClient

from data_processing.common_methods import process_excel_file, json_config_parser
from data_processing.detect_gender import get_gender, get_minor_detect


def main():
    import time
    start_time = time.time()

    config = json_config_parser("connection_config.json")
    df = process_excel_file('Hackathon_Data_MinorityWomenOwned_2022 v1.xlsx', sys.argv[1], sys.argv[2])
    print("<<<------ Reading the source file  ------->>>")

    gender_df = get_gender(df, config.get("api_key"))
    print("<<<------ gender detection completed  ------->>>")

    minor_df = asyncio.run(get_minor_detect(df))
    print("<<<------ minority detection completed  ------->>>")

    print("<<<------ Intermediate data joined ------->>>")
    final_df = pd.merge(pd.merge(df, gender_df, on='dunsNum'), minor_df, on='dunsNum', suffixes=('', '_y'))
    final_df.drop(final_df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    final_df = final_df.loc[:, ]

    xl_writer = pd.ExcelWriter('Hackathon_Data_MinorityWomenOwned_2022_updated.xlsx', engine='xlsxwriter')
    final_df.to_excel(xl_writer, index=False)
    xl_writer.save()

    service = BlobServiceClient.from_connection_string(config.get("storage_connection"))
    container_client = service.get_container_client(config.get("ouput_container"))

    with open("Hackathon_Data_MinorityWomenOwned_2022_updated.xlsx", "rb") as data:
        container_client.upload_blob(name="Hackathon_Data_MinorityWomenOwned_2022_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".xlsx", data=data)

    print("<<<------ final dataset written to excel ------->>>")
    print("<<<--- %s seconds --->>>" % (time.time() - start_time))


if __name__=='__main__':
    main()
