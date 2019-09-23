import reader as r
import tensorflow as tf 
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def load_images(train):
    images, hdr1 = r.read_img('images/0/pre/T1.nii.gz')
    
    if train:
        labels, hdr2 = r.read_img('images/0/wmh.nii.gz')
    return images, labels


def define_model(images):
    model = tf.keras.models.Sequential()
    
    # frist layer
    # Convolution
    model.add(tf.keras.layers.Conv2D(64 , (3,3) , input_shape = (240,240,1) ))
    # Activation funtion
    model.add(tf.keras.layers.Activation('relu'))
    # Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

    # second layer
    # Convolution
    model.add(tf.keras.layers.Conv2D(64 , (3,3) ))
    # Activation funtion
    model.add(tf.keras.layers.Activation('relu'))
    # Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))
    
    # Output Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    return model

def train_model(model, images, labels):
    model.compile(loss = 'binary_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    
    model.fit(images, labels, batch_size = 32, epochs = 3, validation_split = 0.1)
    return model


