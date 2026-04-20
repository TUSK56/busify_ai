import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # أو بدون CAP_DSHOW أولاً

if not cap.isOpened():
    print("Cannot open camera")
    exit()

for i in range(100):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")

cap.release()
cv2.destroyAllWindows()