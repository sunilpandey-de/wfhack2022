import numpy as np
import pandas as pd
import pytest

from data_processing.detect_gender_diversity import flag_gender


def test_flag_gender():
    data1 = pd.DataFrame([['Male', 'Female'],['Male', 'Male'],[np.nan,np.nan]],columns=['executive1_gender', 'executive2_gender'])
    data1['gender'] = data1.apply(flag_gender, axis=1)
    assert data1['gender'].iat[0] == 1
    assert data1['gender'].iat[1] == 0






