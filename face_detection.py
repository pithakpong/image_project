import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('f.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for i, (x, y, w, h) in enumerate(faces):
    face_roi = img[y:y+h, x:x+w]
    new_width, new_height = 100, 100  
    resized_face = cv2.resize(face_roi, (new_width, new_height))
    cv2.imwrite(f'face_{i}.jpg', resized_face)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2.imshow('img', img)
cv2.waitKey()
