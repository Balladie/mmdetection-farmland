import argparse
import copy
import json
import os
from pathlib import Path
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
    if not centers:
        print(f"JSON 파일에 'center' 키가 없습니다: {json_path}")
        return

    # 각 중심점에 대해 위도와 경도 계산
    results = []
    for center in centers:
        pixel_x = center["x"]
        pixel_y = center["y"]
        latitude, longitude = get_lat_lon_from_pixel(tfw_path, pixel_x, pixel_y)
        results.append({"x": pixel_x, "y": pixel_y, "latitude": latitude, "longitude": longitude})

    return results

def save_results_with_gis_to_json(results, json_file_path, output_file_path):
    """
    results 리스트의 'latitude'와 'longitude'를 JSON 파일의 'gis' 키로 추가하여 저장합니다.

    :param results: 위도와 경도를 포함한 리스트
    :param json_file_path: 원본 JSON 파일 경로
    :param output_file_path: 수정된 JSON 파일 저장 경로
    """
    # JSON 파일 읽기
    with open(json_file_path, 'r') as file:
        _data = json.load(file)
    
    # 이전 파일과 충돌나지 않도록 새로 copy 본을 만든다.
    data = copy.deepcopy(_data)

    # 'gis' 키 추가
    data['gis'] = []
    for result in results:
        data['gis'].append({ 
            "latitude": result["latitude"],
            "longitude": result["longitude"]
        })

    # 수정된 JSON 저장
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"JSON 파일을 저장하였습니다 {output_file_path}")

def populate_gis(json_dir, input_dir, out_dir, print):
    output_gis_dir = os.path.join(out_dir, "gis")
    Path(output_gis_dir).mkdir(exist_ok=True)

    for fn in tqdm(os.listdir(json_dir)):
        if ".json" not in fn:
            continue
        json_file_path = os.path.join(json_dir, fn)
        tfw_file_name = fn.replace(".json", ".tfw")
        tfw_file_path = os.path.join(input_dir, tfw_file_name)
        if tfw_file_name not in os.listdir(input_dir):
            print(f"{tfw_file_path} 파일이 {input_dir} 에 없습니다.")
            continue
        results = process_json_and_calculate_coordinates(json_file_path, tfw_file_path)
        if print:
            # 위도 및 경도 출력
            print(f"{fn}: {results}")
        else:
            # 위도 및 경도를 JSON 파일에 저장
            output_file_path = os.path.join(output_gis_dir, fn)
            save_results_with_gis_to_json(results, json_file_path, output_file_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=str, required=True, help="json 파일이 있으며, centers 키가 있어야 합니다.")
    parser.add_argument("--input-dir", type=str, required=True, help="tfw 파일이 있어야 합니다.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="위도 및 경도를 저장할 디렉토리 입니다.")
    parser.add_argument("--print", action="store_true", default=False, help="위도 및 경도를 출력합니다.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_gis_dir = os.path.join(args.out_dir, "gis")
    Path(output_gis_dir).mkdir(exist_ok=True)
    populate_gis(args.json_dir, args.input_dir, args.out_dir, args.print)
