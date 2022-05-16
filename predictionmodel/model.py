import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

from data_processing.baselogger import logger


def preprocess(names_df, train=True):
    names_df['name'] = names_df['name'].str.lower()
    #names_df['name'] = [list(name) for name in names_df['name']]

    names_df['name'] = (names_df['name'].dropna().apply(lambda x: [item for item in x]))
    logger.info("<<<--- Split individual characters  ---->>>>")

    name_length = 50
    names_df['name'] = [(name + [' ']*name_length)[:name_length]
                        for name in names_df['name']
                        ]
    logger.info("<<<--- Pad names with spaces to make all names same length  ---->>>>")

    names_df['name'] = [
        [
            max(0.0, ord(char)-96.0)
            for char in name
        ]
        for name in names_df['name']
    ]
    logger.info("<<<--- Encode Characters to Numbers  ---->>>>")

    if train:
        logger.info("<<<--- Encode Gender to Numbers  ---->>>>")
        names_df['gender'] = [
            0.0 if gender=='F' else 1.0
            for gender in names_df['gender']
        ]

    return names_df


def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def get_gender_diversity(df, name, model):

    df.rename(columns={f'{name}': 'name'}, inplace=True)
    logger.info("<<<--- Rename columns  ---->>>>")

    # Convert to dataframe
    names = df['name'].values.tolist()
    # Preprocess
    pred_df = preprocess(df, train=False)

    # Predictions
    result = model.predict(np.asarray(
        pred_df['name'].values.tolist())).squeeze(axis=1)

    logger.info("<<<--- predictions based on the LSTM model  ---->>>>")
    pred_df['gender'] = [
        'Male' if logit > 0.5 else 'Female' for logit in result
    ]
    logger.info("<<<--- deriving the gender based on model prediction   ---->>>>")

    pred_df['probability'] = [
        logit if logit > 0.5 else 1.0 - logit for logit in result
    ]
    logger.info("<<<--- filtering based on the probability  ---->>>>")

    # Format the output
    pred_df['name'] = names
    pred_df.rename(columns={'name': f'{name}'}, inplace=True)
    pred_df['probability'] = pred_df['probability'].round(2)
    pred_df.drop_duplicates(inplace=True)
    return pred_df
