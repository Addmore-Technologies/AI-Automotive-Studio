# AI Automotive Studio — AI Engine

Core computer vision pipeline for automated vehicle isolation, original background removal, fictional number plate replacement, and photorealistic studio compositing.

Designed as a modular, high-performance vision library for integration into FastAPI backends (Aparna) and presentation in React frontends (Arya).

---

## Architecture & Directory Structure

```text
ai_engine/
├── detection/
│   ├── __init__.py
│   └── vehicle_detector.py      # YOLO-based vehicle detector (cars, trucks, buses, bikes)
├── segmentation/
│   ├── __init__.py
│   └── vehicle_segmenter.py     # High-precision instance segmenter with boundary refinement
├── background_removal/
│   ├── __init__.py
│   └── remover.py               # RGBA transparent cutout extraction & edge fringe de-spill
├── number_plate/
│   ├── __init__.py
│   └── plate_processor.py       # Fictional license plate generation & perspective warping
├── compositing/
│   ├── __init__.py
│   ├── studio_compositor.py     # Fixed studio staging, aspect-ratio scaling, tyre grounding
│   └── shadow_reflection.py     # Contact shadows, ambient occlusion, polished floor reflections
├── common/
│   ├── __init__.py
│   └── types.py                 # Structured dataclasses, Enums, and configurations
├── pipeline.py                  # Stable end-to-end entry points for backend orchestration
├── requirements.txt             # Direct Python dependencies
└── README.md                    # System documentation & integration manual
```

---

## Integration Interface for Aparna (Backend)

Aparna imports the stable top-level functions from `ai_engine` without needing to interact directly with internal neural network model classes:

### 1. Image Processing Entry Point

```python
from ai_engine import process_image

# Process a single still image
result = process_image(
    input_path="uploads/car.jpg",
    output_path="outputs/car_studio.png"  # Optional, auto-generated if None
)

if result["success"]:
    print("Composited image:", result["output_path"])
    print("Transparent cutout:", result["cutout_path"])
    print("Execution time (ms):", result["execution_time_ms"])
else:
    print("Processing error:", result["error"])
```

#### Return Value Structure (`result` dict)

```json
{
  "success": true,
  "status": "COMPLETED",
  "input_path": "/absolute/path/to/car.jpg",
  "output_path": "/absolute/path/to/car_studio.png",
  "cutout_path": "/absolute/path/to/car_studio_cutout.png",
  "image_dimensions": {"width": 1280, "height": 853},
  "detection": {
    "has_detection": true,
    "count": 1,
    "primary_box": {
      "x1": 142, "y1": 210, "x2": 1138, "y2": 720,
      "width": 996, "height": 510, "confidence": 0.942, "class_name": "car"
    },
    "execution_time_ms": 42.1
  },
  "segmentation": {
    "mask_shape": [853, 1280],
    "coverage_pct": 38.6,
    "confidence": 0.935,
    "refined": true,
    "execution_time_ms": 68.4
  },
  "number_plate": {
    "found": true,
    "replaced_text": "STUDIO 01",
    "confidence": 0.88,
    "status_note": "Plate detected and replaced with fictional identifier",
    "execution_time_ms": 14.2
  },
  "compositing": {
    "canvas_dimensions": [1920, 1080],
    "vehicle_dimensions": [1382, 707],
    "scale_factor": 1.387,
    "ground_line_y": 799,
    "shadows_enabled": true,
    "reflections_enabled": true
  },
  "execution_time_ms": 135.2
}
```

---

### 2. Video Frame Entry Point

For processing video streams frame-by-frame:

```python
import cv2
from ai_engine import process_frame

video_state = {}  # Retained across frames for temporal stabilization

cap = cv2.VideoCapture("input_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process frame with temporal mask & bounding box smoothing
    composited_frame, meta = process_frame(frame, state=video_state)
    
    # composited_frame is uint8 BGR array matching studio dimensions (1920x1080)
    out_writer.write(composited_frame)
```

---

## Conventions & Standards

| Parameter | Specification | Note |
|---|---|---|
| **Color Order** | `BGR` (OpenCV standard) | Internally converts to `RGB` for Pillow/PyTorch where needed. Fully configurable via `ColorOrder.RGB`. |
| **Mask Convention** | `0` = Background, `255` = Foreground | 8-bit single channel (`uint8`). |
| **Cutout Format** | 4-channel `RGBA` (PNG) | Channel order `[R, G, B, A]`. `A=0` (transparent), `A=255` (opaque). |
| **Studio Canvas** | 1920 x 1080 (16:9 Full HD) | High-definition automotive cyclorama cove with overhead softbox lighting. |

---

## Core Capabilities & Technical Details

### 1. Vehicle Detection (`VehicleDetector`)
- Built on state-of-the-art YOLO architectures (`yolov8n.pt`, `yolov8m-seg.pt`).
- Filters COCO vehicle classes: `car` (2), `motorcycle` (3), `bus` (5), and `truck` (7).
- Multi-Vehicle Policies:
  - `MultiVehiclePolicy.LARGEST`: Selects primary subject with largest bounding box area.
  - `MultiVehiclePolicy.CENTER`: Selects vehicle closest to frame center.
  - `MultiVehiclePolicy.HIGHEST_CONF`: Selects detection with highest confidence score.
- Handles no-detection edge cases safely, returning `has_detection=False` without throwing unhandled exceptions.

### 2. Vehicle Segmentation (`VehicleSegmenter`)
- High-resolution polygon and mask extraction matching exact image dimensions `(H, W)`.
- Boundary Refinement:
  - **Window & Cabin Hole Filling**: Uses contour topological hierarchy analysis (`cv2.RETR_CCOMP`) to ensure dark vehicle interiors and windshields are preserved without unnatural transparent holes.
  - **Curvature Smoothing**: Morphological closing with elliptical kernels preserves smooth body lines, spoilers, and side mirrors while suppressing 1-pixel antenna noise spikes.
  - **Boundary Anti-Aliasing**: Alpha edge feathering eliminates pixelated staircase artifacts when pasted onto dark studio backgrounds.

### 3. Fictional Number Plate Replacement (`PlateProcessor`)
- **Computer Vision Localization**: Uses vertical Sobel edge gradients, horizontal rectangular closing kernels, aspect ratio filtering (`1.8 - 6.0`), and bumper region localization to detect visible plates.
- **Fictional Synthesis**: Generates an authentic embossed showroom plate (`"STUDIO 01"` or custom string) with European studio blue crest, realistic beveled border, mounting bolts, and environmental lighting adaptation.
- **Perspective Warping**: Warps the synthetic plate into position using 4-point homography transform (`cv2.getPerspectiveTransform`) and feathers boundary edges into the bumper surface.
- **Privacy Assurance**: Never substitutes a real vehicle registration. If a plate is obscured, angled away, or missing, the module safely skips replacement and documents the status.

### 4. Studio Composition & Physics-Based Lighting (`StudioCompositor`)
- **Proportional Scaling**: Resizes the vehicle to occupy ~72% of the studio width while strictly maintaining 100% of the original aspect ratio (never stretches or squashes vehicles).
- **Grounding Alignment**: Detects the lowest tyre contact line and anchors the car precisely on the cyclorama floor horizon.
- **Contact Shadows**: Renders dense tyre contact ambient occlusion patches and diffused chassis underbody shadows.
- **Floor Reflection**: Simulates polished epoxy showroom floors using vertical flipping, perspective vertical compression, progressive quadratic distance falloff, and surface roughness Gaussian blur.

### 5. Video Frame Stabilization
- Consecutive video frames benefit from stateful temporal smoothing:
  - **Box Smoothing**: Exponential moving average on bounding box coordinates prevents jitter.
  - **Mask Smoothing**: Alpha-blended temporal mask tracking eliminates segmentation flickering between adjacent frames.

---

## Model Weights Management
- Pretrained weights (`yolov8n.pt`, `yolov8n-seg.pt`, `yolov8m-seg.pt`) reside in `ai_engine/weights/` and are excluded from Git commits via `.gitignore`.
- Run `python scripts/setup_assets.py` to populate local weights and test data.
