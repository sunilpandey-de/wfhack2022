import asyncio
import numpy as np
import pandas as pd
from ethnicolr import pred_wiki_name

from data_processing.common_methods import extract_gender_fromname


def get_gender(df, key):
    print("<<<---- Gender prediction is started ---->>>")
    df['executive1_gender'] = df.apply(lambda row: asyncio.run(extract_gender_fromname(row['executiveContact1_name'], key)), axis=1)
    df['executive1_gender'].mask(df['executiveContact1_name'].isnull(), np.nan, inplace=True)
    print("<<<---- Gender prediction for executive1 completed ---->>>")

    df['executive2_gender'] = df.apply(lambda row: asyncio.run(extract_gender_fromname(row['executiveContact2_name'], key)), axis=1)
    df['executive2_gender'].mask(df['executiveContact2_name'].isnull(), np.nan, inplace=True)
    print("<<<---- Gender prediction for executive2 completed ---->>>")

    cols = [col for col in df if col.endswith('gender')]
    cols.append("dunsNum")
    gender_df = df.loc[:, cols]
    gender_df['gender'] = gender_df.apply(flag_gender, axis = 1)
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
    diverse_df["race"] = diverse_df["executive1_race"].astype(str).str.cat(diverse_df["executive2_race"], sep=",")
    print(f"<<<-- minority prediction for executive completed ------->>>")
    return diverse_df


def flag_gender(df):
    if (df['executive1_gender'] == 'female') or (df['executive2_gender'] == 'female'):
        return 1
    elif (df['executive1_gender'] == 'male') or (df['executive2_gender'] == 'male'):
        return 0
    else:
        return np.nan
