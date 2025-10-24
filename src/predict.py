import tensorflow as tf
import numpy as np
import difflib
import pandas as pd
from src.model import CTCLayer
from src.preprocess import encode_single_image, num_to_char

model = tf.keras.models.load_model('models/improved_final_model.keras', custom_objects={'CTCLayer': CTCLayer})
dense_layer = None
for layer in reversed(model.layers):
    if 'dense' in layer.name.lower() and 'ctc' not in layer.name.lower():
        dense_layer = layer
        break
prediction_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=dense_layer.output if dense_layer else model.layers[-2].output)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred,input_length=input_len,greedy=True)[0][0]
    output_text=[]
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res,-1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res[:5])
    return output_text

val_predictions, val_true_labels = [], []
for batch in validation_dataset:
    batch_images = batch["image"]
    batch_labels = batch["label"]
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    val_predictions.extend(pred_texts)
