import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path

def preprocess_image(pil_image) -> tf.Tensor:
    """
    Converts a PIL image into a 4D float32 tensor of shape (1, 224, 224, 3).
    """
    # ensuring RGB mode (to handle PNG transparency/grayscale)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # convert PIL image to tensorflow tensor
    img_tensor = tf.constant(np.array(pil_image))

    # resizing to target dimensions (224, 224) to match our model input shape
    resized_image = tf.image.resize(img_tensor, [224, 224])
    
    # convert datatype to float32
    float32_image = tf.cast(resized_image, tf.float32)

    # expanding dimensions to create a 4D batch tensor: (1, 224, 224, 3)
    tensor_4d = tf.expand_dims(float32_image, axis=0)
    return tensor_4d


def top_k_preds(preds, k=5):
    class_names_path = Path(__file__).with_name('class_names.txt')
    
    with class_names_path.open('r', encoding='utf-8') as file:
        class_names = file.read().splitlines()

    # getting probabilits and indices of top k preds
    top_k_values, top_k_indices = tf.math.top_k(preds[0], k=k)

    # converting to numpy for easy looping
    predictions = top_k_values.numpy()
    indices = top_k_indices.numpy()

    # dictionary to store predictions class and probability percentages
    top_k = []
    for i in range(k):
        class_name = class_names[indices[i]]
        confidence = float(predictions[i] * 100)

        # appending as a dictionary to the list
        top_k.append({
            'label': class_name,
            'confidence': round(confidence, 2)
        })

    return top_k