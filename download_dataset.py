#!/usr/bin/env python3
"""
Download a small object detection dataset for benchmarking
"""
import os
import urllib.request
import json
from pathlib import Path

def create_sample_dataset():
    """Create a small sample dataset with images and annotations"""
    
    # Create directories
    data_dir = Path("data")
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images URLs (using free stock images)
    sample_images = [
        {
            "url": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=640&h=480&fit=crop",
            "filename": "dog_001.jpg",
            "annotations": [{"class": "dog", "bbox": [100, 150, 300, 400]}]
        },
        {
            "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640&h=480&fit=crop",
            "filename": "cat_001.jpg", 
            "annotations": [{"class": "cat", "bbox": [120, 100, 350, 380]}]
        },
        {
            "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=640&h=480&fit=crop",
            "filename": "dog_002.jpg",
            "annotations": [{"class": "dog", "bbox": [80, 120, 280, 350]}]
        },
        {
            "url": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640&h=480&fit=crop",
            "filename": "cat_002.jpg",
            "annotations": [{"class": "cat", "bbox": [150, 80, 400, 320]}]
        },
        {
            "url": "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=640&h=480&fit=crop",
            "filename": "dog_003.jpg",
            "annotations": [{"class": "dog", "bbox": [90, 100, 320, 380]}]
        }
    ]
    
    # Download images and create annotations
    dataset_info = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"}
        ]
    }
    
    for i, img_info in enumerate(sample_images):
        try:
            # Download image
            img_path = images_dir / img_info["filename"]
            print(f"Downloading {img_info['filename']}...")
            urllib.request.urlretrieve(img_info["url"], img_path)
            
            # Create image info
            image_info = {
                "id": i + 1,
                "file_name": img_info["filename"],
                "width": 640,
                "height": 480
            }
            dataset_info["images"].append(image_info)
            
            # Create annotations
            for j, ann in enumerate(img_info["annotations"]):
                annotation = {
                    "id": len(dataset_info["annotations"]) + 1,
                    "image_id": i + 1,
                    "category_id": 1 if ann["class"] == "dog" else 2,
                    "bbox": ann["bbox"],  # [x, y, width, height]
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0
                }
                dataset_info["annotations"].append(annotation)
                
        except Exception as e:
            print(f"Error downloading {img_info['filename']}: {e}")
    
    # Save annotations in COCO format
    with open(annotations_dir / "annotations.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset created with {len(dataset_info['images'])} images")
    print(f"Saved to: {data_dir.absolute()}")
    
    return data_dir

if __name__ == "__main__":
    create_sample_dataset()

