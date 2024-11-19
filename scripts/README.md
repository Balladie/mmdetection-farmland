# 농지

모든 CLI commands는 본 repository의 최상위 경로에서 실행하는 기준입니다.

## Prerequisites

사용하는 python 환경에 커스텀된 [sahi](https://github.com/Balladie/sahi) 가 설치되어 있어야 .tif 항공 이미지 추론/평가가 가능합니다. 

```
git clone https://github.com/Balladie/sahi.git
cd sahi
pip install .
```


## 추론

### 단일 이미지

640x640 으로 크롭된 단일 .png 이미지들을 추론합니다.

```bash
python scripts/test.py --config configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py --ckpt [CKPT_PATH] --input-dir [IMAGE_DIR] --out-dir [OUTPUT_DIR]
```

### 항공 이미지

해상도가 큰 원본 .tif 항공 이미지를 슬라이싱하여 각 타일마다 추론합니다.

```bash
python scripts/test_sahi.py --config configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py --ckpt [CKPT_PATH] --input-dir [TIF_DIR] --out-dir [OUTPUT_DIR]
```

그 외 사용 가능한 인자들은 다음과 같습니다.

- `--score-thr`: 최종 결과물에 남길 prediction들의 confidence score 임계점을 설정합니다. 성능에 따라 값을 조절해서 사용합니다.
- `--export-vis`: 시각화 결과를 같이 출력합니다.
- `--center-bbox` : bbox 가운데 좌표를 JSON 에 추가합니다.

## 평가

### 단일 이미지

config에 지정된 test set의 크롭된 단일 이미지들에 대해 COCO eval 평가를 진행합니다.

```bash
bash tools/dist_test.sh configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py [CKPT_PATH] [NUM_GPU]
```

### 항공 이미지

해상도가 큰 원본 .tif 항공 이미지를 슬라이싱하여 각 타일마다 추론하고 그 결과를 바탕으로 COCO eval 평가를 진행합니다.

`TIF_DIR` 경로에 다음과 같이 파일을 저장해 둡니다. 데이터 준비 시 주의사항은 다음과 같습니다.
- COCO annotation 내 이미지 파일 경로가 json 파일 위치에 대한 상대 경로와 일치해야 합니다.
- 필드명이 정확히 COCO format을 따라야 합니다 (예를 들어 카테고리 필드명은 `category`가 아닌 `categories` 입니다. 라벨링된 파일에는 잘못 표기되어 있음)

```
TIF_DIR
├── coco_segmentation.json
├── data1.tif
├── data2.tif
...
```

데이터가 준비되면, 다음과 같이 추론을 진행하고 그 결과를 추출합니다.

```bash
sahi predict --slice_width 640 --slice_height 640 --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 --model_confidence_threshold 0.3 --postprocess_match_threshold 0.5 --source [TIF_DIR] --model_path [CKPT_PATH] --model_config_path configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py --dataset_json_path [TIF_DIR]/coco_segmentation.json --no_standard_prediction --novisual
```

추출한 결과를 바탕으로 아래와 같이 COCO eval을 진행할 수 있습니다. 

이때 평가하고자 하는 `categories` 내 카테고리 항목만 `coco_segmentation.json` 내에 남긴 상태로 실행하면, 해당 카테고리들에 대해서만 평가를 진행할 수 있습니다.

```bash
sahi coco evaluate --dataset_json_path [TIF_DIR]/coco_segmentation.json --result_json_path [RESULT_PATH] --iou_thrs 0.01 --type mask --max_detections 500 --areas "[0 0 100000000000]" --classwise
```
