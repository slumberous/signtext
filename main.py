from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(1)
#change this according to the video input you need

classifier = Classifier("models/keras_model.h5", "models/labels.txt")

labels = ["0", "1", "2", "3"]

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    # Get image frame
    success, img = cap.read()
    imgPros = img.copy()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        x, y, w, h  = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right



        prediction, index = classifier.getPrediction(img)
        cv2.putText(imgPros, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        fingers1 = detector.fingersUp(hand1)

    # Display
    cv2.imshow("Image", imgPros)

    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()