from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Load the model and class labels once to avoid repeated loading
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Define a function to make predictions
def predict_image(image_path):
    # Prepare input array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Make predictions
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Example usage
if __name__ == "__main__":
    image_path = "<IMAGE_PATH>"  # Replace with actual image path
    class_name, confidence_score = predict_image(image_path)
    print(f"Class: {class_name}")
    print(f"Confidence Score: {confidence_score:.2f}")
