import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
    def get_config(self):
        return super().get_config()
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_ctc_model(img_width, img_height, vocab_size):
    input_img = layers.Input(shape=(img_width,img_height,1), name="image")
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(96,(3,3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    new_shape = (img_width//8,(img_height//8)*96)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(96,return_sequences=True))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(vocab_size+1,activation="softmax")(x)
    labels = layers.Input(name="label",shape=(None,),dtype="int32")
    input_length = layers.Input(name="input_length",shape=(1,),dtype="int32")
    label_length = layers.Input(name="label_length",shape=(1,),dtype="int32")
    ctc_output = CTCLayer(name="ctc_loss")(labels,x,input_length,label_length)
    model = keras.models.Model(inputs=[input_img,labels,input_length,label_length], outputs=ctc_output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model
