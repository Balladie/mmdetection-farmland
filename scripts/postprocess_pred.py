import argparse
import cv2
import json
import os
import pycocotools.mask as mask_util
from pathlib import Path


CATEGORIES = [
    {
        "id": 0,
        "name": "농축산업용 시설",
        "category_id": "eee4dc97-0599-48eb-808f-eff2c4ebbdad",
        "supercategory": None,
    },
    {
        "id": 1,
        "name": "일반시설(건축물)",
        "category_id": "f925b900-abb1-4eff-954e-6098c61da4af",
        "supercategory": None,
    },
    {
        "id": 2,
        "name": "비닐하우스",
        "category_id": "3aeaec8a-8cb4-4a39-ae0f-9cb889110c3a",
        "supercategory": None,
    },
    {
        "id": 3,
        "name": "축사",
        "category_id": "52c92da2-88eb-46e9-b80e-d14242af20d0",
        "supercategory": None,
    },
    {
        "id": 4,
        "name": "태양광-노지형",
        "category_id": "c19eef15-c5e2-40ea-bd97-100da5c99af9",
        "supercategory": None,
    },
    {
        "id": 5,
        "name": "태양광-시설형(축사)",
        "category_id": "cc6f147f-4995-40d7-a179-21b48945b43e",
        "supercategory": None,
    },
    {
        "id": 6,
        "name": "태양광-시설형(농축산업용 시설)",
        "category_id": "aa39802e-bbbe-4d30-b408-fba2c7737cd2",
        "supercategory": None,
    },
    {
        "id": 7,
        "name": "태양광-시설형(기타시설)",
        "category_id": "7449c0c6-5a9d-4850-90c6-d5d986bad53f",
        "supercategory": None,
    },
    {
        "id": 8,
        "name": "농막",
        "category_id": "058da039-9442-4103-878a-f19b7f2ad30d",
        "supercategory": None,
    },
    {
        "id": 9,
        "name": "임야화농지",
        "category_id": "41e4b368-6dcf-41b9-86ec-8fb8d448f261",
        "supercategory": None,
    },
    {
        "id": 10,
        "name": "도로",
        "category_id": "856f9fbd-630c-4509-87d3-d4701083cc3f",
        "supercategory": None,
    },
    {
        "id": 11,
        "name": "주차장",
        "category_id": "7e704136-b61a-43b3-b267-cf86b70aafb3",
        "supercategory": None,
    }
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def mask2polygon(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, hierarchy = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    net_area = 0.0
    cnt_valid = 0
    for i, contour in enumerate(contours):
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
            net_area += cv2.contourArea(contour, oriented=True)
            cnt_valid += 1
    if cnt_valid == 0:
        print("Invalid mask, returning empty polygon")
        return [], 0.0
    return segmentation, abs(net_area)


def postprocess(data_path, output_dir):
    with open(data_path) as f:
        data = json.load(f)

    masks_binary = mask_util.decode(data["masks"])

    for i in range(masks_binary.shape[2]):
        mask_bin = masks_binary[:, :, i]
        mask_polygon, area = mask2polygon(mask_bin)
        area_naive = int((mask_bin > 0).sum())  # TODO: investigte difference between contour integral and pixel summation
        data["masks"][i]["polygon"] = mask_polygon
        data["masks"][i]["area"] = area

    data["metadata"] = {
        "image_id": Path(data_path).stem,
        "categories": CATEGORIES,
    }

    if args.score_thr:
        rm_idxs = []
        for i, score in enumerate(data["scores"]):
            if score < args.score_thr:
                rm_idxs.append(i)
        data.update({
            "labels": [label for i, label in enumerate(data["labels"]) if i not in rm_idxs],
            "scores": [score for i, score in enumerate(data["scores"]) if i not in rm_idxs],
            "bboxes": [bbox for i, bbox in enumerate(data["bboxes"]) if i not in rm_idxs],
            "masks": [mask for i, mask in enumerate(data["masks"]) if i not in rm_idxs],
        })
        data["metadata"]["score_thr"] = args.score_thr
        print(f"Score thresholded: {len(data['labels'])} preds left")

    output_path = os.path.join(output_dir, os.path.basename(data_path))
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def main(args):
    if os.path.isdir(args.input_path):
        for fn in os.listdir(args.input_path):
            if ".json" not in fn:
                continue
            data_path = os.path.join(args.input_path, fn)
            postprocess(data_path, args.output_dir)
    else:
        postprocess(args.input_path, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
