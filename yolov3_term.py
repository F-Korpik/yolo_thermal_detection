import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3_termal_old.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Danger"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# reading the video
camera_ip = "192.168.100.1"
#source_term = cv2.VideoCapture("rtsp://"+camera_ip+"/stream0")
source_term = cv2.VideoCapture(r"TERMAL\v1\test3.mp4")

detected_old=False
detected_new=False
img_nr=0
# running the loop
while True:
    img_nr += 1
    frames_rejected = 25
    if img_nr == frames_rejected: img_nr = 0
    # extracting the frames
    ret_term, img_term = source_term.read()
    if not ret_term:
        break

    img = cv2.resize(img_term, (416, 416))
    height, width, channels = img_term.shape

    # Detecting objects
    if (img_nr == 1) or (detected_old == False and (img_nr == frames_rejected // 3 or img_nr == 2 * frames_rejected // 3)):
      blob = cv2.dnn.blobFromImage(img_term, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

      net.setInput(blob)
      outs = net.forward(output_layers)

      # Showing informations on the screen
      class_ids = []
      confidences = []
      boxes = []
      for out in outs:
          for detection in out:
              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              conf=0.4
              if confidence > conf:

                  center_x = int(detection[0] * width)
                  center_y = int(detection[1] * height)
                  w = int(detection[2] * width)
                  h = int(detection[3] * height)

                  # Rectangle coordinates
                  x = int(center_x - w / 2)
                  y = int(center_y - h / 2)

                  boxes.append([x, y, w, h])
                  confidences.append(float(confidence))
                  class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, conf)
    font = cv2.FONT_HERSHEY_PLAIN

    if len(boxes)!=0:
        detected_new=True
    else: detected_new=False
    if detected_old != detected_new: print(detected_new)
    detected_old=detected_new
#'''
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img_term, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_term, label, (x, y + 30), font, 3, color, 2)



    # displaying the video
    cv2.imshow("YOLO", img_term)

    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# closing the window
cv2.destroyAllWindows()
#'''
source_term.release()