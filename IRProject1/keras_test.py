from keras.datasets import boston_housing
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


def load_data():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = Sequential()
    model.add(Dense(units=64, input_dim=13, activation="relu"))
    model.add(Dense(units=64, input_dim=13))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer=RMSprop(), metrics=["mae"])

    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=False)
    test_mse, test_mae = model.evaluate(x_test, y_test)
    print(test_mae)
    print(test_mse)