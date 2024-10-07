import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from data_preprocessing import train_generator

# Load the model
model = load_model('trmodel.h5')


# Function to preprocess and classify an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    class_labels = train_generator.class_indices  # Get class indices from the training generator
    class_labels = dict((v, k) for k, v in class_labels.items())
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

