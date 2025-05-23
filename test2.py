import cv2
import cvzone
import math
import pyttsx3  # Importing pyttsx3 for text-to-speech
from ultralytics import YOLO

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)

# Load the custom-trained YOLO model
model = YOLO("best (1).pt")

# List of class names corresponding to the trained model
classNames = ['-Road narrows on right', '50 mph speed limit', 'Attention Please-', 'Beware of children',
              'CYCLE ROUTE AHEAD WARNING', 'Dangerous Left Curve Ahead', 'Dangerous Right Curve Ahead',
              'End of all speed and passing limits', 'Give Way', 'Go Straight or Turn Right',
              'Go straight or turn left', 'Keep-Left', 'Keep-Right', 'Left Zig Zag Traffic', 'No Entry',
              'No_Over_Taking', 'Overtaking by trucks is prohibited', 'Pedestrian Crossing', 'Round-About',
              'Slippery Road Ahead', 'Speed Limit 20 KMPh', 'Speed Limit 30 KMPh', 'Stop_Sign', 'Straight Ahead Only',
              'Traffic_signal', 'Truck traffic is prohibited', 'Turn left ahead', 'Turn right ahead', 'Uneven Road']

# Custom messages for each class
custom_messages = {
    0: "Caution! The road narrows on the right.",
    1: "Speed limit is 50 miles per hour.",
    2: "Attention please! Follow the road rules.",
    3: "Beware of children crossing ahead.",
    4: "Cyclists ahead. Be careful on the route.",
    5: "Dangerous left curve ahead. Drive carefully.",
    6: "Dangerous right curve ahead. Slow down.",
    7: "End of all speed and passing limits.",
    8: "Give way to oncoming traffic.",
    9: "Go straight or turn right.",
    10: "Go straight or turn left.",
    11: "Keep to the left.",
    12: "Keep to the right.",
    13: "Zigzag traffic to the left ahead.",
    14: "No entry! Do not proceed.",
    15: "No overtaking is allowed here.",
    16: "Overtaking by trucks is prohibited here.",
    17: "Pedestrian crossing ahead. Watch out.",
    18: "Approaching a roundabout. Slow down.",
    19: "Slippery road ahead. Drive carefully.",
    20: "Speed limit is 20 kilometers per hour.",
    21: "Speed limit is 30 kilometers per hour.",
    22: "Stop sign ahead. Be prepared to stop.",
    23: "Go straight ahead only.",
    24: "Traffic signal ahead. Obey the lights.",
    25: "Truck traffic is prohibited in this area.",
    26: "Turn left ahead.",
    27: "Turn right ahead.",
    28: "Uneven road ahead. Drive slowly."
}

# Initialize text-to-speech engine
engine = pyttsx3.init()


# Function to speak out detected traffic signs
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Variable to keep track of the last detected sign to avoid repeated speech
last_detected_sign = None

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw a corner rectangle around the detection
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Get the confidence and class index for the detection
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Display the class name and confidence on the image
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=2)

            # Check if the detected sign is different from the last one spoken
            detected_sign = classNames[cls]
            if detected_sign != last_detected_sign and conf > 0.6:  # Only speak if confidence is high enough
                custom_message = custom_messages.get(cls, "Unknown sign detected.")
                speak(custom_message)  # Speak out the custom message
                last_detected_sign = detected_sign  # Update last detected sign

    # Show the image in a window
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
