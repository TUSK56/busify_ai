import face_recognition
import pickle
import os

known_encodings = []
known_names = []
dataset_dir = "students"  # اسم المجلد اللي فيه المجلدات الفرعية

print("جاري استخراج الوجوه من الصور ...\n")

if not os.path.isdir(dataset_dir):
    print(f"❌ المجلد غير موجود: {os.path.abspath(dataset_dir)}")
    print("أنشئ مجلد students/ ثم ضع داخلها مجلد لكل طالب، وداخل كل مجلد صور للوجه.")
    print("مثال: students/Ahmed/1.jpg  students/Ahmed/2.jpg")
    raise SystemExit(1)

for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    print(f"→ معالجة الطالب: {person_name}")
    count = 0
    
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_folder, filename)
            try:
                image = face_recognition.load_image_file(img_path)
                # ملاحظة: الأفضل أن تكون الصورة واضحة وبوجه واحد
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    count += 1
                    print(f"   • تم استخراج وجه من: {filename}")
                else:
                    print(f"   • ما لقاش وجه في: {filename}")
            except Exception as e:
                print(f"   • خطأ في {filename}: {e}")
    
    if count == 0:
        print(f"   تحذير: ما اتعرفش أي وجه لـ {person_name}")

if known_encodings:
    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    print(f"\nتم حفظ {len(known_encodings)} encoding بنجاح في encodings.pickle")
else:
    print("\nمشكلة: ما لقاش وجوه صالحة في الصور!")