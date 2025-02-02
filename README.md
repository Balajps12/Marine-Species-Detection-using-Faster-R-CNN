# **Marine Species Detection using Faster R-CNN** 🌊🐠  

## **Overview**  
This project implements a deep learning-based object detection system for identifying marine species in images. Using a **Faster R-CNN** model trained on the [Aquarium Object Detection Dataset](https://public.roboflow.com/object-detection/aquarium/2), the system detects various underwater creatures such as fish, jellyfish, penguins, puffins, sharks, starfish, and stingrays.  

## **Features**  
✅ Upload an image for marine species detection  
✅ Detect and label multiple species in the image  
✅ Display bounding boxes with confidence scores  
✅ Simple and interactive **Streamlit** UI  

## **Dataset**  
- **Source:** [Roboflow Aquarium Dataset](https://public.roboflow.com/object-detection/aquarium/2)  
- **Classes:**  
  - Creatures  
  - Fish  
  - Jellyfish  
  - Penguin  
  - Puffin  
  - Shark  
  - Starfish  
  - Stingray  

## **Installation**  
### Clone the Repository  

git clone https://github.com/Balaj-PS/Marine-Species-Detection.git
cd Marine-Species-Detection

### Run the Application

streamlit run app.py

### Model Details

- **Architecture:** Faster R-CNN
- **Framework:** PyTorch
- **Trained Weights:** faster_rcnn_model.pth

### Usage

- Open the Streamlit UI
- Upload an image containing marine species
- The model detects objects and displays predictions with bounding boxes

Future Enhancements
🔹 Improve accuracy with fine-tuning
🔹 Extend detection to more marine species
🔹 Deploy as a web service
