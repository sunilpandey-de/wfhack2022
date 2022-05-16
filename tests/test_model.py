import pandas as pd

from predictionmodel.model import preprocess, get_gender_diversity

names = [{"name": "lorrie"},
         {"name": "george"},
         {"name": "suju"},
         {"name": "jay"},
         {"name": "doug"}]

df = pd.DataFrame(names)


def test_preprocess():
    df1 = preprocess(df, False)
    assert (df1['name'].iat[0])[0] == 12.0


def test_gender_diversity():
    from tensorflow.keras.models import load_model
    import h5py
    pred_model = load_model('boyorgirl.h5')
    df2 = get_gender_diversity(df, 'name', pred_model)
    assert df2['name'].iat[0] == 'lorrie'
    assert df2['gender'].iat[0] == 'Female'
