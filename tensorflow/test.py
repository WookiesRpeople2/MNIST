import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def drawplt(model, x, y, num_plots=30):
    preds = np.argmax(model.predict(x), axis=1)

    fig, axes = plt.subplots(num_plots//10, 10, figsize=(20, 15))
    fig.suptitle("Predictions")
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i])
        ax.set_title(f'Prediction: {preds[i]}')
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    msint = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = msint.load_data()
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.load_model("./tensorflow/trained_model.tnf")

    drawplt(model, X_test, y_test, 40)
