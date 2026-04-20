# busvision_attendance.py
# نظام تسجيل حضور الطلاب بالوجه داخل حافلة BusVision 360
# مناسب للمشرف داخل الباص

import cv2
import dlib
import face_recognition
import pickle
import numpy as np
import csv
import os
from datetime import datetime

# ────────────────────────────────────────────────
# إعدادات أساسية
# ────────────────────────────────────────────────
ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance_bus.csv"
RESIZE_FACTOR = 0.5          # للسرعة (0.5 = نصف الحجم)
UPSAMPLE = 1                 # لـ HOG detector
TOLERANCE = 0.52             # يمكن تعديله (أقل = أدق، أكثر = أسرع)
USE_CNN = False              # غيري لـ True بعد حل مشكلة CNN

# ────────────────────────────────────────────────
# تحميل الـ encodings (وجوه الطلاب المسجلة)
# ────────────────────────────────────────────────
print("جاري تحميل بيانات الطلاب...")
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_encodings = data.get("encodings", []) or []
    known_names = data.get("names", []) or []
    if len(known_encodings) != len(known_names) or not known_encodings:
        raise ValueError("Invalid encodings database.")
    print(f"تم تحميل {len(known_names)} وجه طالب بنجاح")
except Exception as e:
    print("خطأ في تحميل encodings.pickle →", e)
    print("تأكدي من تشغيل generate_encodings.py أولاً")
    exit(1)

# ────────────────────────────────────────────────
# إعداد الـ face detector
# ────────────────────────────────────────────────
if USE_CNN:
    print("استخدام CNN face detector (أبطأ لكن أدق)")
    model_path = "mmod_human_face_detector.dat"
    if not os.path.exists(model_path):
        print(f"❌ ملف CNN غير موجود: {os.path.abspath(model_path)}")
        print("غيّري USE_CNN إلى False أو ضعي ملف model بجانب السكربت.")
        exit(1)
    detector = dlib.cnn_face_detection_model_v1(model_path)
else:
    print("استخدام HOG face detector (أسرع وأكثر استقرارًا)")
    detector = dlib.get_frontal_face_detector()

# ────────────────────────────────────────────────
# فتح الكاميرا (0 = الكاميرا الافتراضية)
# ────────────────────────────────────────────────
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("خطأ: لا يمكن فتح الكاميرا")
    exit(1)

# لتسجيل الحضور مرة واحدة فقط لكل طالب في الجلسة
already_marked = set()

# إنشاء ملف الحضور إذا مش موجود + كتابة الهيدر
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["الاسم", "التاريخ", "الوقت", "الحالة"])

print("نظام الحضور جاهز... اضغطي 'q' للخروج")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("خطأ في قراءة الإطار")
        break

    # تصغير الصورة للسرعة
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    # تحويل إلى RGB (مهم جدًا)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # كشف الوجوه
    if USE_CNN:
        dets = detector(rgb_small, UPSAMPLE)
        face_locations = [(d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()) for d in dets]
    else:
        face_locations = detector(rgb_small, UPSAMPLE)
        face_locations = [(r.top(), r.right(), r.bottom(), r.left()) for r in face_locations]

    # استخراج الـ encodings للوجوه المكتشفة
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # مقارنة مع الوجوه المعروفة
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
        name = "غير معروف"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # تعديل المواقع للحجم الأصلي
        top    = int(top    / RESIZE_FACTOR)
        right  = int(right  / RESIZE_FACTOR)
        bottom = int(bottom / RESIZE_FACTOR)
        left   = int(left   / RESIZE_FACTOR)

        # رسم مربع واسم
        color = (0, 255, 0) if name != "غير معروف" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # تسجيل الحضور لو مش مسجل قبل كده
        if name != "غير معروف" and name not in already_marked:
            already_marked.add(name)
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str, time_str, "حضور داخل الباص"])

            print(f"تم تسجيل حضور → {name}  في {time_str}")

    # عرض الشاشة
    cv2.imshow("BusVision 360 - نظام حضور الطلاب", frame)

    # خروج بالضغط على q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تنظيف
video_capture.release()
cv2.destroyAllWindows()
print("تم إغلاق النظام. الحضور محفوظ في:", ATTENDANCE_FILE)