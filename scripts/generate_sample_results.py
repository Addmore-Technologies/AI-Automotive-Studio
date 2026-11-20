"""
Generate end-to-end sample results for all input images in sample_data/input/
Saves both the composited studio images and the isolated transparent cutouts
into sample_data/expected/ for verification and documentation.
"""

import os
import sys
import glob

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from ai_engine.pipeline import get_pipeline


def main():
    input_dir = os.path.abspath("sample_data/input")
    output_dir = os.path.abspath("sample_data/expected")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = get_pipeline()
    
    images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(images)} sample images in {input_dir}")

    for img_path in sorted(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        out_studio_path = os.path.join(output_dir, f"{stem}_studio.png")
        
        print(f"\nProcessing {basename}...")
        res = pipeline.process_image(img_path, output_path=out_studio_path)
        
        if res["success"]:
            print(f"  [SUCCESS] -> {out_studio_path}")
            print(f"  Cutout: {res['cutout_path']}")
            print(f"  Detection: {res['detection']['count']} vehicle(s) found, primary: {res['detection']['primary_box']['class_name']}")
            print(f"  Plate: {res['number_plate']['status_note']} (replaced: '{res['number_plate']['replaced_text']}')")
            print(f"  Execution time: {res['execution_time_ms']} ms")
        else:
            print(f"  [FAILED] -> {res.get('error')}")


if __name__ == "__main__":
    main()
