import numpy as np
import pandas as pd
import pytest

from data_processing.detect_gender_diversity import get_gender, flag_gender


def test_get_gender():
    new_data = pd.DataFrame([[123,'sunil', 'maria']],
                        columns=['dunsnum', 'executivecontact1_name', 'executivecontact2_name'])
    df = get_gender(new_data, '746941dd70d18d9e21147566090c8035')
    assert df['executive1_gender'].iat[0] == 'male'
    assert df['executive2_gender'].iat[0] == 'female'


def test_flag_gender():
    data1 = pd.DataFrame([['Male', 'Female'],['Male', 'Male'],[np.nan,np.nan]],columns=['executive1_gender', 'executive2_gender'])
    data1['gender'] = data1.apply(flag_gender, axis=1)
    assert data1['gender'].iat[0] == 1
    assert data1['gender'].iat[1] == 0






