# Copyright (c) OpenMMLab. All rights reserved.
import os
import argparse

import json
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='labelme to coco')
    parser.add_argument('--root', 
                        help='labelme json files folder')
    parser.add_argument('--classes',
                        help="ex. {'bz': 0, 'jansa': 1, 'penetration': 2}'")
    parser.add_argument('--save_dir',
                        default='./coco_format_data',
                        help='the dir to save')
    parser.add_argument('--split_ratio',
                        default=0.8,
                        help='set train split rate')
    
    args = parser.parse_args()
    return args

def labelme_to_coco(args, labelme_files, image_type):
    category_mapping = args.classes
    labelme_folder = args.root
    
    make_dir_list = ['annotations', 'train', 'val']
    for dir in make_dir_list:
        if not os.path.exists(os.path.join(args.save_dir, dir)): os.makedirs(os.path.join(args.save_dir, dir))
    
    # Create COCO format dictionary
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [{"id": id, "name": name} for name, id in category_mapping.items()],
        "images": [],
        "annotations": []
    }

    
    # Mapping for category IDs
    # Process each labelme file
    annotation_id = 1
    image_id = 1
    for labelme_file in labelme_files:
        with open(os.path.join(labelme_folder, labelme_file), "r") as f:
            labelme_data = json.load(f)

        # Convert images
        image_info = {
            "id": image_id,
            "file_name": labelme_data["imagePath"],
            "height": labelme_data["imageHeight"],
            "width": labelme_data["imageWidth"]
        }
        coco_data["images"].append(image_info)

        # Convert annotations
        for shape in labelme_data["shapes"]:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[shape["label"]],
                "segmentation": [],
                "area": 0,
                "bbox": [],
                "iscrowd": 0
            }

            # Convert segmentation
            points = np.asarray(shape["points"], dtype=np.float32)
            annotation["segmentation"].append(points.ravel().tolist())

            # Compute bounding box and area
            x, y, w, h = cv2.boundingRect(points)
            annotation["bbox"] = [x, y, w, h]
            annotation["area"] = w * h

            coco_data["annotations"].append(annotation)
            annotation_id += 1

        # Save image
        image_path = os.path.join(labelme_folder, labelme_data["imagePath"])
        save_path = os.path.join(args.save_dir, image_type, labelme_data["imagePath"])
        img = cv2.imread(image_path)
        cv2.imwrite(save_path, img)

        image_id += 1

    # Save COCO data as JSON file
    coco_json_path = os.path.join(args.save_dir, f"annotations/{image_type}.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f)


def main():
    args = parse_args()
    
    
        
    # Get list of labelme files
    labelme_files = [f for f in os.listdir(args.root) if f.endswith(".json")]

    num_train = int(args.split_ratio * len(labelme_files))
    train_files = labelme_files[:num_train]
    val_files = labelme_files[num_train:]
    
    labelme_to_coco(args, train_files, 'train')
    labelme_to_coco(args, val_files, 'val')
    

if __name__ == '__main__':
    main()
