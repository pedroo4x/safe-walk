import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()