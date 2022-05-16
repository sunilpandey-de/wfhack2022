import numpy as np
import pandas as pd
from ethnicolr import pred_fl_reg_ln_five_cat

from data_processing.baselogger import logger
from predictionmodel.model import get_gender_diversity


def flag_gender(df):
    if (df['executive1_gender'] == 'Female') or (df['executive2_gender'] == 'Female'):
        return 1
    elif (df['executive1_gender'] == 'Male') or (df['executive2_gender'] == 'Male'):
        return 0
    else:
        return np.nan


def get_minor_detect_cat(df, config):
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
    diverse_df[["executive1_race", "executive2_race"]] = diverse_df[["executive1_race", "executive2_race"]].replace(
        to_replace=str(config.get("region_list")).split(","), value='')
    diverse_df["race"] = diverse_df[['executive1_race', 'executive2_race']].apply(
        lambda x: ','.join(x.replace(np.nan, '')), axis=1)
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
    gender_df['gender'] = gender_df.apply(flag_gender, axis=1)
    logger.info("<<<---- Gender prediction is complete ---->>>")
    return gender_df
