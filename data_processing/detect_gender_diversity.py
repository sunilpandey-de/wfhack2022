import asyncio
import numpy as np
import pandas as pd
from ethnicolr import pred_wiki_name, pred_fl_reg_ln_five_cat

from data_processing.baselogger import logger
from data_processing.common_methods import extract_gender_fromname, extract_origin_fromname
from predictionmodel.model import preprocess, get_gender_diversity


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

