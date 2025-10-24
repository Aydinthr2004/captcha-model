import tensorflow as tf
from model import CTCLayer, build_ctc_model
from preprocess import encode_single_image, char_to_num, num_to_char
import numpy as np
import os

model_path = r"C:\Users\aydin\OneDrive\Desktop\captcha-model\models\improved_final_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})

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

def predict_single(img_path):
    img = encode_single_image(img_path)
    img = tf.expand_dims(img, axis=0)
    pred = prediction_model.predict(img)
    return decode_batch_predictions(pred)[0]

if __name__=="__main__":
    img_name = input("Enter image filename (in 'example/' folder): ")
    img_path = os.path.join(r"C:\Users\aydin\OneDrive\Desktop\captcha-model\example", img_name)
    print(predict_single(img_path))
