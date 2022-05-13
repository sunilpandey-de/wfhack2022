import asyncio
import numpy as np
import pandas as pd
from ethnicolr import pred_wiki_name

from data_processing.common_methods import extract_gender_fromname


def get_gender(df, key):
    print("<<<---- Gender prediction is started ---->>>")
    df['executive1_gender'] = df.apply(lambda row: extract_gender_fromname(row['executiveContact1_name'], key), axis=1)
    print("<<<---- Gender prediction for executive1 completed ---->>>")
    df['executive2_gender'] = df.apply(lambda row: extract_gender_fromname(row['executiveContact2_name'], key), axis=1)
    print("<<<---- Gender prediction for executive2 completed ---->>>")
    df["gender"] = df.apply(lambda x: np.nan if (x['executiveContact1_name'] == np.nan and x['executive1_gender'] != np.nan) or (x['executiveContact2_name'] == np.nan and x['executive2_gender'] != np.nan) else x['executive1_gender'], axis=1)
    cols = [col for col in df if col.endswith('gender')]
    cols.append("dunsNum")
    gender_df = df.loc[:, cols]
    gender_df["gender"] = np.where(gender_df["gender"] == "female", 1, 0)
    print("<<<---- Gender prediction is complete ---->>>")
    return gender_df


async def get_minor_detect(df):
    # processing data for minority-owned
    print(f"<<<-- minority prediction  started ------->>>")
    diverse_df1 = pred_wiki_name(df, lname_col='executiveContact1_lname', fname_col='executiveContact1_fname', num_iter=20)
    print(f"<<<-- minority prediction for executive1 completed ------->>>")

    diverse_df2 = pred_wiki_name(df, lname_col='executiveContact2_lname', fname_col='executiveContact2_fname', num_iter=20)
    print(f"<<<-- minority prediction for executive2 completed ------->>>")

    diverse_df1["executive1_race"] = diverse_df1["race"].str.split(',').str[-1]
    diverse_df1 = diverse_df1.loc[:, ["dunsNum", "executive1_race"]]

    diverse_df2["executive2_race"] = diverse_df2["race"].str.split(',').str[-1]
    diverse_df2 = diverse_df2.loc[:, ["dunsNum", "executive2_race"]]

    diverse_df = pd.merge(diverse_df1, diverse_df2, how='inner')
    diverse_df[["executive1_race", "executive2_race"]] = diverse_df[["executive1_race", "executive2_race"]].replace(to_replace=["British", "French", 'Italian', 'Germanic', 'Nordic'], value='')
    diverse_df["race"] = diverse_df["executive1_race"].astype(str).str.cat(diverse_df["executive2_race"], sep = ",")
    print(f"<<<-- minority prediction for executive completed ------->>>")
    return diverse_df
