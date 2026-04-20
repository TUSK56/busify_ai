import cv2
import dlib

# تحميل الـ detector الافتراضي (HOG-based)
detector = dlib.get_frontal_face_detector()

# فتح الكاميرا
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("مش قادر يفتح الكاميرا، جربي رقم تاني زي 1 أو 2")
    exit()

print("الكاميرا شغالة – اضغطي q عشان تقفلي")

while True:
    ret, frame = cap.read()
    if not ret:
        print("مش قادر يقرأ الإطار")
        break

    # تحويل لـ grayscale عشان الـ detector يشتغل أحسن
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # كشف الوجوه (الـ 1 = upsample مرة واحدة عشان دقة أعلى شوية)
    faces = detector(gray, 1)

    # رسم مستطيل على كل وجه
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # رسم المستطيل الأخضر
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # كتابة "وجه" فوق المستطيل (اختياري)
        cv2.putText(frame, "وجه", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # عرض الإطار
    cv2.imshow("dlib HOG Face Detection - اضغط q للخروج", frame)

    # خروج بالضغط على q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تنظيف
cap.release()
cv2.destroyAllWindows()
print("تم إغلاق الكاميرا")