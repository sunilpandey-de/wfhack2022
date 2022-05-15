import numpy as np
import pandas as pd
import pytest

from data_processing.detect_gender import get_gender, flag_gender


def test_get_gender():
    new_data = pd.DataFrame([[123,'sunil', 'maria']],
                        columns=['dunsnum', 'executivecontact1_name', 'executivecontact2_name'])
    df = get_gender(new_data, 'ccd498f48fe9da653392772756499b63')
    assert df['executive1_gender'].iat[0] == 'male'
    assert df['executive2_gender'].iat[0] == 'female'


def test_flag_gender():
    data1 = pd.DataFrame([['male', 'female'],['male', 'male'],[np.nan, np.nan]],columns=['executive1_gender', 'executive2_gender'])
    data1['gender'] = data1.apply(flag_gender, axis=1)
    assert data1['gender'].iat[0] == 1
    assert data1['gender'].iat[1] == 0






