import tensorflow as tf

print("Loading your model...")
model = tf.keras.models.load_model('deepfake_model.h5')

print("Converting to small format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Saving new model...")
with open('deepfake_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("DONE! File: deepfake_model.tflite")