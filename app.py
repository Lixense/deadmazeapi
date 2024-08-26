from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

# Import the YOLOv8 model from Ultralytics (assuming you have the ultralytics package)
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

def predict_zombies(image):
    # Convert the image to a format YOLOv8 expects
    # Run inference
    results = model(image)

    # Filter out detections with the class "Zombie"
    # Assuming that 'Zombie' is class 0, otherwise adjust the index.
    zombie_detections = [d for d in results[0].boxes if d.cls == 0]
    
    # Return the number of zombies detected
    return len(zombie_detections)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Get the number of zombies
    num_zombies = predict_zombies(image)
    
    # Return the result
    return jsonify({"zombies_detected": num_zombies})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
