Got Food101 dataset from tensorflow_datasets in the following manner.
```py
import tensorflow_datasets as tfds

# Load in the data
(train_data, test_data), ds_info = tfds.load(name='food101', # target dataset to get from TFDS
                                             split=['train', 'validation'], # what splits of data should we get? note: not all datasets have train, valid, test
                                             shuffle_files=True, # shuffle files on download?
                                             as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                             with_info=True) # include dataset metadata? is so, tfds.load returns tuple (data, ds_info)
```

Goal with this project was to beat [DeepFood](https://www.google.com/url?q=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment), a 2016 paper which used a CNN trained for 2-3 days to achiever 77,4% top-1% accuracy.

Train set has 75,750 images and test set has 25,250 images. val set has 3,788 images (15% of total test data)
There are a total of 101 classes.
Training set has 750 images per class, testing set has 20 images per clsss.

Total dataset size is 4.65 GiB

Resized all images to 224x224 resolution using this method,
```py
# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple
```

Then created batches and mapped the preprocessing function to the data like this
```py
# mapping preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img)

# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map preprocessing function onto test data
test_data = test_data.map(preprocess_img)

# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
```

Used mized precision while training to increase my efficiency
```py
# turning on mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy='mixed_float16') # set global policy to mixed precision

mixed_precision.global_policy() # should output "mixed_float16" (if your GPU is compatible with mixed precision)
```

Then built a feature extraction model
```py
from tensorflow.keras import layers

# creating base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False # freezing base model layers

# creating functional model
inputs = layers.Input(shape=input_shape, name='input_layer')

# data augmentation layers
x = layers.RandomFlip('horizontal', name='data_aug_flip')(inputs)
x = layers.RandomRotation(0.15, name='data_aug_rotation')(x)
x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1, name='data_aug_translation')(x)
x = layers.RandomContrast(factor=0.2, name='data_aug_contrast')(x)

# note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
# x = layers.Rescaling(1./255)(x)

x = base_model(inputs, training=False) # setting base model to inference mode only

x = layers.GlobalAveragePooling2D(name='pooling_layer')(x)
x = layers.Dropout(0.3, name='head_dropout')(x)
x = layers.Dense(len(class_names))(x) # want one output return per class

# separate activation of output layer so we can output float32 activations
outputs = layers.Activation('softmax', dtype=tf.float32, name='softmax_float32')(x)

model = tf.keras.Model(inputs, outputs)

# compiling the model
model.compile(loss='sparse_categorical_crossentropy', # use spares_categoical_crossentropy when labels are *not* one-hot encoded
              optimizer='adam',
              metrics=['accuracy'])
```

fitted our feature extraction model for 5 epochs, with a checkpoint callback and a custom tnsorboard callback to create tensorboard logs
```py
# creating TensorBoard callback (already have "create_tensorbard_callback()" from a previous notebook)
from helper_functions import create_tensorboard_callback

# creating ModelCheckpoint callback to save model's progress
checkpoint_path = 'model_checkpoint/cp.weights.h5' # saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor='val_accuracy', # save the model weights with the best validation accuracy
                                                      save_best_only=True, # only saves the best weights
                                                      save_weights_only=True, # only saves model weights (not the entire model)
                                                      verbose=0) # don't print our whether or not model is being saved

# turning off all warning except for errors
tf.get_logger().setLevel('ERROR')

# fitting the model with callbacks
history_101_food_classes_feature_Extract = model.fit(train_data,
                                                     epochs=5, # since data augmentation is added, training for longer (5 instead of 3 epochs)
                                                     steps_per_epoch=len(train_data),
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)),
                                                     callbacks=[create_tensorboard_callback('training_logs',
                                                                                            'efficeintnetb0_101_classes_all_data_feature_extract'),
                                                                model_checkpoint])
```
this feature extraction model has the following metrics on test data (including 15% of validation data)
`accuracy: 0.7278 - loss: 0.9974`

Moved on with fine tuning with some callbacks
```py
# setup EarlyStopping callback to stop training if model's val_loss doesn;t improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', # watch the val loss metric
                                                  patience=3) # if val loss decreaser for 3 epochs in a row, stop training

# create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = 'fine_tune_checkpoints/07_efficientnetb0_101_classes_all_data_fine_tune_model.keras'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor='val_loss')

# creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down
                                                 min_lr=1e-7)
```

Started fine tuning it with these callbacks at a max epoch liit of 100
```py
# start to fine-tune all layers
history_101_food_classes_all_data_fine_tune = loaded_saved_model.fit(train_data,
                                                                     epochs=100, # fine-tune for a maximum of 100 epochs
                                                                     steps_per_epoch=len(train_data),
                                                                     validation_data=test_data,
                                                                     validation_steps=int(0.15 * len(test_data)), # validation during training on 15% of test data
                                                                     callbacks=[create_tensorboard_callback('training_logs', 'efficientb0_101_classes_all_data_fine_tuning'), # track the model training logs
                                                                                model_checkpoint, # save only the bset model during training
                                                                                early_stopping, # stop model training after X epochs of no improvements
                                                                                reduce_lr]) # reduce the learning rate after X epochs of not improvementss
```

It trained for 26 epochs, these are the test data results,
```
Accuracy: 0.8268
Precision: 0.8259
Recall: 0.8268
F1: 0.8259
```

The reason of making this project is that SILC was a relatively small scaled model, the eurosat rgb band dataset was not as massive has the food101. I wanted to experience some larger scaled model development, hence made this