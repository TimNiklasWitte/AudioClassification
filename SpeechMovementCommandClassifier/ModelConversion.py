import tensorflow as tf
import os





def convert_model_to_tflite(model=None, model_path=None, target_path=None, model_name=None):
  if not model_name:
    model_name= "model"
  if target_path:
    export_path = os.path.join(target_path, model_name)
  else:
    export_path= './model'
  if model is None and model_path is None:
        raise ValueError("You need to pass a model or model_path")

  if (isinstance(model, tf.keras.Model)):
    model.export(export_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(export_path) # path to the SavedModel directory
    tflite_model = converter.convert()
  else:
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
    tflite_model = converter.convert()
  # Save the model.
  with open((export_path + '.tflite'), 'wb') as f:
    f.write(tflite_model)