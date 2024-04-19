import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the model weights
model.load_weights("emotiondetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Define labels for emotion classes
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read a frame from the webcam
    i, im = webcam.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try: 
        # Iterate over detected faces
        for (p, q, r, s) in faces:
            # Extract the face region
            image = gray[q:q+s, p:p+r]
            
            # Draw a rectangle around the detected face
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            # Resize the face region to match model input size
            image = cv2.resize(image, (48, 48))
            
            # Extract features from the face image
            img = extract_features(image)
            
            # Make prediction using the loaded model
            pred = model.predict(img)
            
            # Get the predicted emotion label
            prediction_label = labels[pred.argmax()]
            
            # Display the predicted emotion label on the frame
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display the output frame
        cv2.imshow("Output", im)
        
        # Wait for key press to exit
        cv2.waitKey(27)
    
    except cv2.error:
        pass
