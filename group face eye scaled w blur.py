import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('groupimg.jpg')


#  Scale
scale_factor = 3.5  
new_width = int(img.shape[1] * scale_factor)
new_height = int(img.shape[0] * scale_factor)
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Convert to grayscale and equalize histogram
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#  Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

for (x, y, w, h) in faces:
    #  RED rectangle face box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # ROI for eyes
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_gray = cv2.equalizeHist(face_roi_gray)
    face_roi_color = img[y:y+h, x:x+w]

    # Define eye size constraints
    eye_width_min = int(w * 0.1)
    eye_width_max = int(w * 0.3)

    #  Detect eyes
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(eye_width_min, eye_width_min),
        maxSize=(eye_width_max, eye_width_max)
    )

    for (ex, ey, ew, eh) in eyes:
        if ey > h * 0.6:
            continue  

        #  Blur the eye region
        eye_roi = face_roi_color[ey:ey+eh, ex:ex+ew]
        eye_roi_blur = cv2.GaussianBlur(eye_roi, (23, 23), 30)
        face_roi_color[ey:ey+eh, ex:ex+ew] = eye_roi_blur

        #  GREEN rectangle for blurred eye
        cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # Green in BGR

#  Display final result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Face detection with blurred eyes')
plt.axis('off')
plt.show()
