import cv2
import time
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture device.")
    exit()

video_capture.set(cv2.CAP_PROP_FPS, 1)

start_time = time.time()  # get the start time
emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0}

while (time.time() - start_time) < 10:  # loop for 10 seconds
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            emotion = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            dominant_emotion = emotion[0]['dominant_emotion']
            #print(dominant_emotion)
            emotion_counts[dominant_emotion] += 1

        cv2.imshow('Video', frame)
        time.sleep(1)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
maxs = 0
# print the emotion counts
for emotion, count in emotion_counts.items():
    #print(emotion + ': ' + str(count))
    if count >= maxs:
        finale = emotion
        maxs=count
print("Overall Emotion Dominance :",finale)

