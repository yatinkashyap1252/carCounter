import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
top_line_position = 550
bottom_line_position = 650
offset = 10
detect = []
counter = 0

def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)), iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(img, (25, bottom_line_position), (1500, bottom_line_position), (255, 0, 0), 3)
    
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w >= 60 and h >= 60 and 0.5 < w / h < 2.0 and cv2.contourArea(c) > 800:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(img, center, 4, (0, 0, 255), -1)

            if (top_line_position - offset) < center[1] < (top_line_position + offset):
                label = f"Vehicle {i+1}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if (bottom_line_position - offset) < center[1] < (bottom_line_position + offset):
                counter += 1
                cv2.line(img, (25, bottom_line_position), (1500, bottom_line_position), (0, 127, 255), 3)
                detect.remove(center)

    cv2.putText(img, f"Count: {counter}", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()