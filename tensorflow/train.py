import tensorflow as tf


def nuraleNet():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(150, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    # because there are 10 classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    model.add(tf.keras.layers.Dense(10))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def evalModel(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Validation Loss: {}, Validation Accuracy: {}".format(loss, accuracy))


if __name__ == "__main__":
    msint = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = msint.load_data()
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    model = nuraleNet()

    model.fit(X_train, y_train, epochs=10)

    evalModel(model, X_test, y_test)

    model.save("./tensorflow/trained_model.tnf")
