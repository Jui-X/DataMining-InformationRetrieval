import csv

from keras.engine.saving import load_model

from FinalProject import load_data

MODEL_PATH = "BGRU_model.h5"
OUTPUT_PATH = "data/GRU_output.csv"
BATCH_SIZE = 16


if __name__ == "__main__":
    x_test = load_data.test_txt
    y_test = load_data.test_score
    model = load_model(MODEL_PATH)

    # score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    # print(score)

    predictions = model.predict(x_test, batch_size=BATCH_SIZE)
    # for prediction in predictions:
    #     print(prediction)

    with open(OUTPUT_PATH, "w") as f:
        writer = csv.writer(f)

        writer.writerows(predictions)