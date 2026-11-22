import os
import sys
import urllib.request


def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"Already exists: {dest_path}")
        return True

    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    temp_path = f"{dest_path}.tmp"

    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if len(data) > 1000:
            os.replace(temp_path, dest_path)
            print(f"Downloaded: {dest_path} ({len(data)} bytes)")
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Failed to download valid content from {url} (received {len(data)} bytes)")
            return False
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        print(f"Failed to download {url}: {e}")
        return False


def main():
    weights_dir = os.path.abspath("ai_engine/weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Download YOLO models
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt",
        "yolov8m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-seg.pt",
    }
    for name, url in models.items():
        download_file(url, os.path.join(weights_dir, name))

    # 2. Download permitted sample car images
    samples_dir = os.path.abspath("sample_data/input")
    os.makedirs(samples_dir, exist_ok=True)

    sample_images = {
        "car_sedan_front.jpg": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=1280&q=80",
        "car_suv_side.jpg": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=1280&q=80",
        "car_sports_red.jpg": "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=1280&q=80",
        "car_rear_plate.jpg": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=1280&q=80",
        "car_street_white.jpg": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=1280&q=80",
    }
    for name, url in sample_images.items():
        download_file(url, os.path.join(samples_dir, name))


if __name__ == "__main__":
    main()
