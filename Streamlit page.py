import torch
from torchvision import models
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# Function to load the model with state_dict adjustments
@st.cache_resource
def load_model():
    model_path = r"F:\Path here\faster_rcnn_model.pth"
    
    # Load the Faster R-CNN model (without pretrained weights)
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    
    # Get the number of input features for the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # Update this to match the number of your classes (e.g., background and fish)
    
    # Update the box predictor to match the number of classes
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Load the checkpoint directly without 'model' key
    checkpoint = torch.load(model_path)
    
    # Remove the box predictor weights from the state_dict
    checkpoint.pop('roi_heads.box_predictor.cls_score.weight', None)
    checkpoint.pop('roi_heads.box_predictor.cls_score.bias', None)
    checkpoint.pop('roi_heads.box_predictor.bbox_pred.weight', None)
    checkpoint.pop('roi_heads.box_predictor.bbox_pred.bias', None)

    # Load the state_dict
    model.load_state_dict(checkpoint, strict=False)
    
    model.eval()  # Set to evaluation mode
    return model

# Load the model at the start of the app
model = load_model()

# Class names (as per your training)
# Replace with actual class names (e.g., marine species classes) from your model's training dataset
class_names = [
    "Background",  # Class 0 for background
    "Marine Species"  # Class 1 for the marine species detected
]

# Determine the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the selected device
model = model.to(device)

# Streamlit app UI
st.title("Faster R-CNN Object Detection")
st.write("Upload an image to run the object detection model and display the results.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Preprocess the image (transform to tensor)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move the image tensor to the selected device
    image_tensor = image_tensor.to(device)
    
    # Run the model on the image
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Apply NMS to remove overlapping bounding boxes
    keep = torch.ops.torchvision.nms(predictions[0]['boxes'], predictions[0]['scores'], 0.5)
    
    # Get boxes, labels, and scores after NMS
    boxes = predictions[0]['boxes'][keep].cpu().numpy()
    labels = predictions[0]['labels'][keep].cpu().numpy()
    scores = predictions[0]['scores'][keep].cpu().numpy()
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write(f"Detected {len(boxes)} objects.")
    
    # Loop through the boxes and display them on the image
    img_array = np.array(image)
    
    # Loop through the detected objects
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Only show detections with confidence above 0.5
            x1, y1, x2, y2 = boxes[i]
            
            # Draw bounding box
            cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Get the label and corresponding class name
            label_id = labels[i]
            label_name = class_names[label_id]  # Map label to class name
            
            # Label text with confidence score
            label_text = f"{label_name}: {scores[i]:0.2f}"
            
            # Add label and score to the image
            cv2.putText(img_array, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the detection results
    st.image(img_array, caption='Detection Results', use_container_width=True)
