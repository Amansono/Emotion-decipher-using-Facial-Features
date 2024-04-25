import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model (assuming it's saved as 'best.h5')
model = load_model('best.h5')

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for single image input
    return feature / 255.0

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Define labels for emotion classes (modify based on your model)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read a frame from the webcam
    ret, im = webcam.read()
    if not ret:
        break  # Exit if frame reading fails

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

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
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Display the output frame
        cv2.imshow("Output", im)

        # Wait for key press to exit (ESC key)
        if cv2.waitKey(1) == ord('q'):
            break

    except cv2.error:
        pass

# Release the webcam resources
webcam.release()
cv2.destroyAllWindows()
