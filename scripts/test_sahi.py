import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from gis_processor import GISProcessor


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



def get_all_tif_files(input_dir):
    """Recursively get all .tif files from input directory and its subdirectories."""
    tif_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files

def create_output_dirs(input_path, base_out_dir):
    """Create output directories maintaining input directory structure."""
    rel_path = os.path.relpath(os.path.dirname(input_path), start=args.input_dir)
    output_base_dir = os.path.join(base_out_dir, rel_path)
    output_pred_dir = os.path.join(output_base_dir, "preds")
    output_vis_dir = os.path.join(output_base_dir, "vis")
    
    Path(output_pred_dir).mkdir(parents=True, exist_ok=True)
    Path(output_vis_dir).mkdir(parents=True, exist_ok=True)
    
    return output_pred_dir, output_vis_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_dirs")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--export-vis", action="store_true", default=False)
    parser.add_argument("--center-bbox", action="store_true", default=False, help="bbox의 중심 좌표를 계산하여 [center] 키로 저장합니다.")
    parser.add_argument("--gis", action="store_true", default=False, help="위도 및 경도를 계산하여 [gis] 키로 저장합니다.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 기본 출력 디렉토리 생성
    base_input_dir = os.path.basename(os.path.normpath(args.input_dir))
    base_output_dir = os.path.join(args.out_dir, base_input_dir)
    error_log_path = os.path.join(base_output_dir, "errorlog.txt")
    
    # 에러 로그 파일 생성
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    with open(error_log_path, "w") as error_log:
        error_log.write("Error Log\n")
        error_log.write("==========\n")

    # 모델 로드
    model = AutoDetectionModel.from_pretrained(
        model_type="mmdet",
        model_path=args.ckpt,
        config_path=args.config,
        confidence_threshold=args.score_thr,
        image_size=640,
        device="cuda",
    )

    # 모든 .tif 파일 경로 수집
    tif_files = get_all_tif_files(args.input_dir)

    for img_path in tqdm(tif_files):
        try:
            # 출력 디렉토리 생성
            output_pred_dir, output_vis_dir = create_output_dirs(img_path, args.out_dir)
            
            # 예측 수행
            result = get_sliced_prediction(
                img_path,
                model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_match_threshold=0.5,
                perform_standard_pred=False,
            )
            preds = [pred.to_coco_prediction() for pred in result.object_prediction_list]
            fn = os.path.basename(img_path)
            preds_dict = {
                "metadata": {
                    "image_id": Path(fn).stem,
                    "categories": CATEGORIES,
                },
                "labels": [pred.category_id for pred in preds],
                "scores": [pred.score for pred in preds],
                "bboxes": [pred.bbox for pred in preds],
                "masks": [{
                    "polygon": pred.segmentation,
                    "area": pred.area,
                } for pred in preds],
            }

            if args.center_bbox:
                preds_dict["center"] = [GISProcessor.get_center_bbox(bbox) for bbox in preds_dict["bboxes"]]

            if args.gis:
                preds_dict["centers_gis"] = GISProcessor.populate_gis_from_dict(preds_dict, img_path.replace(".tif", ".tfw"))
                preds_dict["bbox_lat_lon"] = [
                    GISProcessor.get_bbox_pos_with_lat_lon(bbox, img_path.replace(".tif", ".tfw")) for bbox in preds_dict["bboxes"]
                ]
            
            # 예측 결과 저장
            output_path = os.path.join(output_pred_dir, fn.replace(Path(fn).suffix, ".json"))
            with open(output_path, "w") as f:
                json.dump(preds_dict, f, indent=4, ensure_ascii=False)

            # 시각화 결과 저장 (옵션)
            if args.export_vis:
                result.export_visuals(export_dir=output_vis_dir, file_name=Path(fn).stem, rect_th=1, hide_conf=True, hide_labels=True)
        
        except Exception as e:
            # 에러 발생 시 로그에 기록
            with open(error_log_path, "a") as error_log:
                error_log.write(f"Error processing file: {img_path}\n")
                error_log.write(f"Error message: {str(e)}\n\n")
