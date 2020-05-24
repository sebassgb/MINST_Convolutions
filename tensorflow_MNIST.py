import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
  
    class myCallback(tf.keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs={}):
         if(logs.get('loss')<0.02):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True
 
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)

    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    callbacks = myCallback()

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # model fitting
    history = model.fit(
        model.fit(training_images, training_labels, epochs=20)
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
    test_loss = model.evaluate(test_images, test_labels)
    
    _, _ = train_mnist_conv()
