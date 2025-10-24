from src.model import build_ctc_model
import tensorflow as tf


model = build_ctc_model(200,50,vocab_size=62)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True,verbose=1,min_delta=0.005),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=4,verbose=1,min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint('models/improved_final_model.keras',monitor='val_loss',save_best_only=True,verbose=1)
]
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=80, callbacks=callbacks, verbose=1)
model.save('models/improved_final_model.h5')
