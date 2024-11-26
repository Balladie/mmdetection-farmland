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

다음과 같이 데이터셋을 준비합니다.

```
IMG_DIR
├── annotations
    ├── instances_train.json
    ├── instances_val.json
├── KG-KR-301
├── krcc
...
```

다음으로 위 경로를 config에 반영해줍니다. 

사용하려는 `mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py` 는 `mask2former_r50_8xb2-lsj-50e_farmland.py`에 데이터 경로가 지정되어 있으므로, `mask2former_r50_8xb2-lsj-50e_farmland.py` 내 `data_root` 변수를 다음과 같이 경로에 맞게 수정해줍니다. 

```python
...
dataset_type = 'FarmlandDataset'
data_root = 'IMG_DIR'   # 여기를 경로에 맞게 수정

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_train.json',
...
```

이후 다음과 같이 추론 및 평가를 진행합니다. 

```bash
bash tools/dist_test.sh configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py [CKPT_PATH] [NUM_GPU]
```

결과는 다음과 같이 출력됩니다. 

IoU=0.05:0.05, area=all, maxDets=100 에 대한 AP와 AR을 조화평균하면 F1 score를 얻습니다. 

아래 예시에 따르면 F1은 `2 x 0.737 x 0.920 / (0.737 + 0.920) = 0.8184` 입니다.

```
 Average Precision  (AP) @[ IoU=0.05:0.05 | area=   all | maxDets=100 ] = 0.737
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.05:0.05 | area= small | maxDets=1000 ] = 0.572
 Average Precision  (AP) @[ IoU=0.05:0.05 | area=medium | maxDets=1000 ] = 0.714
 Average Precision  (AP) @[ IoU=0.05:0.05 | area= large | maxDets=1000 ] = 0.694
 Average Recall     (AR) @[ IoU=0.05:0.05 | area=   all | maxDets=100 ] = 0.920
 Average Recall     (AR) @[ IoU=0.05:0.05 | area=   all | maxDets=300 ] = 0.920
 Average Recall     (AR) @[ IoU=0.05:0.05 | area=   all | maxDets=1000 ] = 0.920
 Average Recall     (AR) @[ IoU=0.05:0.05 | area= small | maxDets=1000 ] = 0.855
 Average Recall     (AR) @[ IoU=0.05:0.05 | area=medium | maxDets=1000 ] = 0.927
 Average Recall     (AR) @[ IoU=0.05:0.05 | area= large | maxDets=1000 ] = 0.897
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
