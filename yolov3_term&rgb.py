import cv2
import numpy as np

# Load Yolo
net_term = cv2.dnn.readNet("yolov3_termal.weights", "yolov3_testing.cfg")
net_rgb = cv2.dnn.readNet("yolov3_rgb.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Danger"]

layer_names_term = net_term.getLayerNames()
output_layers_term = [layer_names_term[i - 1] for i in net_rgb.getUnconnectedOutLayers()]
layer_names_rgb = net_rgb.getLayerNames()
output_layers_rgb = [layer_names_rgb[i - 1] for i in net_rgb.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# reading the video
camera_ip = "192.168.100.1"
#source_term = cv2.VideoCapture("rtsp://"+camera_ip+"/stream0")
source_term = cv2.VideoCapture(r"TERMAL\test3.mp4")
source_rgb = cv2.VideoCapture(r"RGB\test1.mp4")

detected_old=False
detected_new=False
detected_rgb=False
img_nr=0
# running the loop
while True:
    img_nr += 1
    frames_rejected = 30
    if img_nr == frames_rejected: img_nr = 0
    # extracting the frames
    ret_term, img_term = source_term.read()
    ret_rgb, img_rgb = source_rgb.read()
    if not ret_term:
        break

    img = cv2.resize(img_term, (416, 416))
    height, width, channels = img_term.shape

    # Detecting objects
    if (img_nr == 1) or (detected_old == False and (img_nr == frames_rejected // 3 or img_nr == 2 * frames_rejected // 3)):
        #termal
      blob_term = cv2.dnn.blobFromImage(img_term, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

      net_term.setInput(blob_term)
      outs_term = net_term.forward(output_layers_term)

        #rgb
      blob_rgb = cv2.dnn.blobFromImage(img_rgb, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

      net_rgb.setInput(blob_rgb)
      outs_rgb = net_term.forward(output_layers_rgb)


      # Showing informations on the screen
      class_ids_term = []
      confidences_term = []
      boxes_term = []
      for out in outs_term:
          for detection in out:
              scores_term = detection[5:]
              class_id_term = np.argmax(scores_term)
              confidence_term = scores_term[class_id_term]
              conf_term=0.4
              if confidence_term > conf_term:
                  center_x = int(detection[0] * width)
                  center_y = int(detection[1] * height)
                  w = int(detection[2] * width)
                  h = int(detection[3] * height)

                  # Rectangle coordinates
                  x = int(center_x - w / 2)
                  y = int(center_y - h / 2)

                  boxes_term.append([x, y, w, h])
                  confidences_term.append(float(confidence_term))
                  class_ids_term.append(class_id_term)

    indexes = cv2.dnn.NMSBoxes(boxes_term, confidences_term, 0.0, conf_term)
    font = cv2.FONT_HERSHEY_PLAIN

    if len(boxes_term)!=0:
        #testing RGB
        class_ids_rgb = []
        confidences_trgb = []
        boxes_rgb = []
        for out in outs_rgb:
            for detection in out:
                scores_rgb = detection[5:]
                class_id_rgb = np.argmax(scores_rgb)
                confidence_rgb = scores_term[class_id_rgb]
                conf_rgb = 0.4
                if confidence_rgb > conf_rgb: detected_rgb=True

        detected_new=True
    else: detected_new=False
    if detected_old != detected_new and detected_rgb==False : print(detected_new)
    detected_old=detected_new
    detected_rgb = False

    if(detected_rgb ==False):
        for i in range(len(boxes_term)):
            if i in indexes:
                x, y, w, h = boxes_term[i]
                label = str(classes[class_ids_term[i]])
                color = colors[class_ids_term[i]]
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
source_term.release()