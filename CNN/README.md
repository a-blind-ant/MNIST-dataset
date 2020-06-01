# MNIST_CNN
Implemented CNN using Keras library on MNIST dataset

I am really thankful to [this](https://pdfs.semanticscholar.org/450c/a19932fcef1ca6d0442cbf52fec38fb9d1e5.pdf) tutorial on Convolutional Neural Networks by Prof. Jianxin Wu, LAMDA group, Nanjing University.

Used several references for the preferred architectures on MNIST dataset.  

DATASET: MNIST (28 x 28) grayscale images for digit recognition

RESULTS:

    x_train.shape : (60000, 28, 28, 1)

    number of training samples : 60000
    
    number of test samples : 10000

    Train on 60000 samples, validate on 10000 samples
    
    Epoch 1/12
    60000/60000 [==============================] - 143s 2ms/step - loss: 0.2597 - accuracy: 0.9212 - val_loss: 0.0696 - val_accuracy: 0.9782

    Epoch 2/12
    60000/60000 [==============================] - 134s 2ms/step - loss: 0.0863 - accuracy: 0.9745 - val_loss: 0.0420 - val_accuracy: 0.9861

    Epoch 3/12
    60000/60000 [==============================] - 137s 2ms/step - loss: 0.0654 - accuracy: 0.9804 - val_loss: 0.0380 - val_accuracy: 0.9867

    Epoch 4/12
    60000/60000 [==============================] - 136s 2ms/step - loss: 0.0550 - accuracy: 0.9830 - val_loss: 0.0360 - val_accuracy: 0.9873

    Epoch 5/12
    60000/60000 [==============================] - 136s 2ms/step - loss: 0.0471 - accuracy: 0.9859 - val_loss: 0.0303 - val_accuracy: 0.9902

    Epoch 6/12
    60000/60000 [==============================] - 136s 2ms/step - loss: 0.0414 - accuracy: 0.9873 - val_loss: 0.0289 - val_accuracy: 0.9910

    Epoch 7/12
    60000/60000 [==============================] - 134s 2ms/step - loss: 0.0382 - accuracy: 0.9883 - val_loss: 0.0292 - val_accuracy: 0.9897

    Epoch 8/12
    60000/60000 [==============================] - 124s 2ms/step - loss: 0.0335 - accuracy: 0.9897 - val_loss: 0.0298 - val_accuracy: 0.9903

    Epoch 9/12
    60000/60000 [==============================] - 139s 2ms/step - loss: 0.0319 - accuracy: 0.9900 - val_loss: 0.0267 - val_accuracy: 0.9913

    Epoch 10/12
    60000/60000 [==============================] - 138s 2ms/step - loss: 0.0297 - accuracy: 0.9909 - val_loss: 0.0281 - val_accuracy: 0.9913

    Epoch 11/12
    60000/60000 [==============================] - 135s 2ms/step - loss: 0.0273 - accuracy: 0.9914 - val_loss: 0.0281 - val_accuracy: 0.9914

    Epoch 12/12
    60000/60000 [==============================] - 137s 2ms/step - loss: 0.0264 - accuracy: 0.9918 - val_loss: 0.0304 - val_accuracy: 0.9899

**Test loss:  0.03044554551005058**

**Best val_accuracy = 0.9914**
