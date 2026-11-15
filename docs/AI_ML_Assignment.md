# AI/ML Assignment — Computer Vision

## Goal
Build reusable Python functions that detect and isolate a vehicle, remove its original background, replace a visible plate with a fictional identifier, and composite it into the team's one approved studio environment.

## Responsibilities
1. Vehicle detection using YOLO or another validated detector.
2. Vehicle segmentation using SAM 2 or another validated method.
3. Refine masks around mirrors, windows, wheels, tyres, and fine edges.
4. Remove background and produce transparent foreground (e.g. RGBA PNG) and/or documented mask.
5. Detect visible number plate and replace registration with a clearly fictional identifier; never use another real registration.
6. Composite onto the approved fixed studio background while preserving vehicle proportions and appearance.
7. Add contact shadows and floor reflections through image processing/compositing.
8. Expose frame-level processing functions and test consecutive frames for flicker.

## Files to create
```text
ai_engine/
├── detection/vehicle_detector.py
├── segmentation/vehicle_segmenter.py
├── background_removal/remover.py
├── number_plate/plate_processor.py
├── compositing/{studio_compositor.py,shadow_reflection.py}
├── common/types.py
├── pipeline.py
├── requirements.txt
└── README.md
```

## Integration interface
Agree with Aparna on stable functions, for example:
```python
def process_image(input_path: str, output_path: str) -> dict:
    ...

def process_frame(frame, studio_background):
    ...
```
Document arguments, return values, RGB/BGR order, mask format, dimensions, and errors. Finalize function contracts with Aparna before integration.

## Suggested stack
Python, YOLO, SAM 2, OpenCV, NumPy, PyTorch; PaddleOCR or another OCR tool if needed. Validate versions, licences, hardware needs, and performance before choosing.

## Work order
Collect permitted sample images → detection → segmentation → transparent cut-outs → fictional plate replacement → fixed-studio compositing → shadows/reflections → stable callable pipeline → frame consistency tests.

## Done when
A documented function processes a supported image end-to-end, produces a clean isolated vehicle and final studio composite, preserves vehicle appearance as closely as possible, replaces a visible plate with a fictional identifier where detectable, and can be called by the backend.

## Boundaries
AI/ML owns computer vision and per-frame processing. Aparna owns APIs and video extraction/reconstruction. Arya owns UI.
