import collections

import numpy as np
import pandas as pd
import pytest
from data_processing.common_methods import json_config_parser, split_pandas_col, \
    process_excel_file



def test_json_config_parser():
    config = json_config_parser('connection_config.json')
    assert str(config.get("ouput_container")) == "output"


def test_split_pandas_col():
    expected = ['persondetail', 'persondetail_0', 'persondetail_1']
    new_data = pd.DataFrame([['sunil-bangalore'],
        ['siva-hyderabad'],
        ['sumit-calcutta']],
        columns=['persondetail'])
    df = split_pandas_col(new_data, ['persondetail'], '-')
    print(df.columns)
    assert all([a==b for a,b in zip(df.columns, expected)])


def test_process_excel_file():
    file = "test.xlsx"
    start = 0
    end = 5
    df = process_excel_file(file, start, end)
    assert df.columns[0] == 'executivecontact1'
    assert df.columns[2] == 'executivecontact1_name'
    assert df.columns[5] == 'executivecontact2_job'

