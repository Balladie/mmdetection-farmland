import argparse
import os
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

def get_lat_lon_from_pixel(tfw_path, pixel_x, pixel_y):
    """
    TFW 파일과 픽셀 좌표를 사용하여 위도 및 경도를 계산합니다.

    :param tfw_path: TFW 파일 경로
    :param pixel_x: 이미지 내 픽셀의 X 좌표 (열)
    :param pixel_y: 이미지 내 픽셀의 Y 좌표 (행)
    :return: 해당 픽셀의 위도 및 경도 (x, y 좌표)
    """
    # TFW 파일 읽기
    with open(tfw_path, 'r') as tfw_file:
        lines = tfw_file.readlines()
        pixel_size_x = float(lines[0])  # X 방향 픽셀 크기
        rotation_x = float(lines[1])   # X 축 회전 (0인 경우가 많음)
        rotation_y = float(lines[2])   # Y 축 회전 (0인 경우가 많음)
        pixel_size_y = float(lines[3]) # Y 방향 픽셀 크기 (일반적으로 음수)
        upper_left_x = float(lines[4]) # 왼쪽 상단 X 좌표
        upper_left_y = float(lines[5]) # 왼쪽 상단 Y 좌표

    # 픽셀 좌표를 지리 좌표로 변환
    coord_x = upper_left_x + pixel_x * pixel_size_x + pixel_y * rotation_x
    coord_y = upper_left_y + pixel_x * rotation_y + pixel_y * pixel_size_y

    return coord_x, coord_y

def process_json_and_calculate_coordinates(json_path, tfw_path):
    """
    JSON 파일에서 픽셀 좌표를 읽고 위도 및 경도를 계산합니다.

    :param json_path: JSON 파일 경로
    :param tfw_path: TFW 파일 경로
    """
    # JSON 파일 읽기
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # "center" 키에서 픽셀 좌표 가져오기
    centers = data.get("center", [])

    # 각 중심점에 대해 위도와 경도 계산
    results = []
    for center in centers:
        pixel_x = center["x"]
        pixel_y = center["y"]
        latitude, longitude = get_lat_lon_from_pixel(tfw_path, pixel_x, pixel_y)
        results.append({"x": pixel_x, "y": pixel_y, "latitude": latitude, "longitude": longitude})

    # 결과 출력
    for result in results:
        print(f"픽셀 좌표 ({result['x']}, {result['y']}) -> 위도: {result['latitude']}, 경도: {result['longitude']}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=str, required=True, help="json 파일이 있으며, centers 키가 있어야 합니다.")
    parser.add_argument("--input-dir", type=str, required=True, help="tfw 파일이 있어야 합니다.")
    parser.add_argument("--out-dir", type=str, default="out_dirs", help="위도 및 경도를 저장할 디렉토리 입니다.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_gis_dir = os.path.join(args.out_dir, "gis")
    Path(output_gis_dir).mkdir(exist_ok=True)

    for fn in tqdm(os.listdir(args.input_dir)):
        if ".tif" not in fn:
            continue

json_file_path = 'example.json'  # 실제 JSON 파일 경로 입력
tfw_file_path = 'example.tfw'   # 실제 TFW 파일 경로 입력

process_json_and_calculate_coordinates(json_file_path, tfw_file_path)