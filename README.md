# Thermal Camera Hazard Detection for ADAS

Bachelor's Thesis project — real-time road hazard detection system using a custom-trained YOLOv3 model on thermal camera feed. Designed to detect pedestrians and animals on roadsides in low-light and no-light conditions where standard RGB cameras fail.

![Detection preview](docs/screenshot.png)
<!-- Replace with actual frame showing bounding box detection -->

---

## Problem

Standard ADAS cameras fail in complete darkness. Thermal cameras detect heat signatures regardless of lighting, enabling reliable detection of living hazards (people, animals) that would be invisible to a conventional camera at night.

---

## How It Works

A custom YOLOv3 model runs inference on thermal video frames, drawing bounding boxes around detected hazards in real time.

Detection logic:
1. Each frame is passed through the thermal YOLOv3 model
2. Detections above 0.4 confidence threshold are marked as "Danger"
3. Bounding box and label are drawn on the thermal frame

**Frame-skipping optimization:** inference runs every N frames, with additional checks mid-window when no hazard was detected in the previous cycle — balancing real-time performance with detection latency.

Input can be either a local video file or a live RTSP stream from a vehicle-mounted thermal camera.

---

## Tech Stack

**Model:** YOLOv3 (custom-trained weights)  
**Inference:** OpenCV DNN module  
**Input:** thermal video stream (RTSP or local file)  
**Language:** Python · NumPy · OpenCV  
**Hardware target:** vehicle-mounted thermal camera system

---

## Project Structure

```
yolo_custom_detection/
├── TERMAL/
│   ├── TERMO_HUMAN/              # Thermal human detection dataset
│   ├── TERMO_HUMAN&HORSE/        # Thermal multi-class dataset
│   └── v1/                       # First iteration test footage
├── baza bounding box/            # Bounding box annotation data
├── Optimize/                     # Optimization experiments
├── yolov3_term.py                # Main detection script
├── yolov3_termal.weights         # Trained model weights
└── yolov3_testing.cfg            # YOLOv3 architecture config
```

---

## Getting Started

```bash
git clone https://github.com/F-Korpik/yolo-thermal-adas
cd yolo-thermal-adas

pip install opencv-python numpy
```

Run detection:
```bash
python yolov3_term.py
```

By default the script reads from a local test video file. To switch to a live RTSP stream, uncomment the camera line in the script:
```python
# source_term = cv2.VideoCapture("rtsp://192.168.100.1/stream0")
```

---

## Detection Classes

The model was trained to detect a single unified class — **"Danger"** — covering:

- Pedestrians
- Horses (used as proxy for deer due to thermal dataset availability)
- Dogs and cats

The unified class approach was chosen intentionally: from a driver assistance perspective, the response to any of these objects is identical.

---

## Key Design Decisions

**Why thermal?** RGB cameras fail in complete darkness; thermal provides reliable signal from living heat sources regardless of lighting conditions.

**Why horses as deer proxy?** Thermal deer datasets are scarce. Horses share a similar body size and heat signature profile, making them a practical substitute for training data.

**Why a single "Danger" class?** Distinguishing between a pedestrian and a deer is irrelevant to the driver — what matters is that something is on the road. A unified class simplifies training and improves recall.

---

## Roadmap

- [ ] Replace frame-skipping heuristic with proper async inference pipeline
- [ ] Expand training dataset with actual deer thermal footage
- [ ] Evaluate on mAP metric across test sequences
- [ ] Package as deployable module for embedded hardware (Jetson Nano)
