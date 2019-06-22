from keras import Input, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, concatenate, Reshape

from FinalProject import load_data

DATA_PATH = "data/user_seq.json"
TRAIN_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"
MODEL_PATH = "Time_Sequence_model.h5"
OUTPUT_PATH = "data/LSTM_output.csv"
MAX_WORDS_NUM = 20000
BATCH_SIZE = 32
DROPOUT_VALUE = 0.2
MAX_LEN = 20


def get_model():
    text = load_data.txt_input
    user = load_data.user_input
    time = load_data.time_input
    score = load_data.score_input

    test_text = load_data.test_text
    test_user = load_data.test_user
    test_time = load_data.test_time
    test_score = load_data.test_score

    text_input = Input(shape=(20,), name="text_input")
    time_input = Input(shape=(20,), name="time_input")

    # print(text.shape)
    # print(time.shape)

    x = Embedding(input_dim=MAX_WORDS_NUM, output_dim=128)(text_input)
    y = Embedding(input_dim=MAX_WORDS_NUM, output_dim=128)(time_input)

    x = LSTM(32, return_sequences=True)(x)
    y = LSTM(32, return_sequences=True)(y)

    z = concatenate([x, y], axis=-1)

    z = Dropout(DROPOUT_VALUE)(z)
    output = Dense(1, activation="relu", name="output")(z)
    print(output.shape)

    model = Model(inputs=[text_input, time_input], output=output)
    model.summary()

    model.compile(optimizer="sgd", loss="mse")

    model.fit([text, time], score, batch_size=BATCH_SIZE, epochs=3)

    model.save(MODEL_PATH)

    score = model.evaluate([test_text, test_time], test_score, batch_size=BATCH_SIZE)

    print(score)


if __name__ == "__main__":
    get_model()