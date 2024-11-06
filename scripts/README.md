# 농지

모든 CLI commands는 본 repository의 최상위 경로에서 실행하는 기준입니다.

## 추론

### 단일 이미지

640x640 으로 크롭된 단일 .png 이미지들을 추론합니다.

```bash
python scripts/test.py --config configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py --ckpt [CKPT_PATH] --input-dir [IMAGE_DIR] --out-dir [OUTPUT_DIR]
```

### 항공 이미지

해상도가 큰 원본 .tif 항공 이미지를 슬라이싱하여 각 타일마다 추론합니다.

사용하는 python 환경에 [sahi](https://github.com/obss/sahi) 가 설치되어 있어야 사용 가능합니다. 

```bash
python scripts/test_sahi.py --config configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py --ckpt [CKPT_PATH] --input-dir [TIF_DIR] --out-dir [OUTPUT_DIR]
```

그 외 사용 가능한 인자들은 다음과 같습니다.

- `--score-thr`: 최종 결과물에 남길 prediction들의 confidence score 임계점을 설정합니다. 성능에 따라 값을 조절해서 사용합니다.
- `--export-vis`: 시각화 결과를 같이 출력합니다.

## 평가

config에 지정된 test set에 대해 COCO eval 평가를 진행합니다.

```bash
bash tools/dist_test.sh configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_farmland.py [CKPT_PATH] [NUM_GPU]
```
