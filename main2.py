import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

emotion = "Detecting..."
frame_count = 0

print("Optimized Emotion Detection Started")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,   # tighter detection
        minNeighbors=7,   # reduces false positives
        minSize=(100, 100) # ignore small objects
    )

    frame_count += 1

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        # Run emotion every 30 frames
        if frame_count % 30 == 0:
            try:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(
                    face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='mtcnn'  # faster backend
                )

                if isinstance(result, list):
                    result = result[0]

                emotion = result['dominant_emotion']

            except:
                emotion = "..."

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()