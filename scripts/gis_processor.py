import argparse
import copy
import json
import os
from pathlib import Path
import pyproj
from tqdm import tqdm


class GISProcessor:
    # Class variable
    # True : 중부원점 기준 위성좌표계 정보를 WGS84 기준 경위도로 변경
    # False: 중부원점 기준 위성좌표계 정보를 그대로 사용
    gis_wgs84 = True

    def __init__(self, json_dir, input_dir, out_dir="outputs", print_results=False):
        self.json_dir = json_dir
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.print_results = print_results
        self.output_gis_dir = os.path.join(self.out_dir, "gis")
        Path(self.output_gis_dir).mkdir(exist_ok=True)

    @staticmethod
    def get_center_bbox(bbox):
        x, y, w, h = bbox
        return {"x": x + w / 2, "y": y + h / 2}
    

    # 중부원점 기준 위성좌표계 정보를 WGS84 기준 경위도로 변경
    def convert_tm_wgs(x, y):
        # Define the source and target coordinate systems
        source_crs = pyproj.CRS.from_epsg(5186)  # EPSG:5186 (중부원점)
        target_crs = pyproj.CRS.from_epsg(4326)  # EPSG:4326 (WGS 84 - World Geodetic System) (경위도)

        # Create a transformer to convert from the source to the target CRS
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        # transform retruns longitude, latitude order
        lon, lat = transformer.transform(x, y) 
        return lat, lon

    @staticmethod
    def get_lat_lon_from_pixel(tfw_path, pixel_x, pixel_y):
        with open(tfw_path, 'r') as tfw_file:
            lines = tfw_file.readlines()
            pixel_size_x = float(lines[0])
            rotation_x = float(lines[1])
            rotation_y = float(lines[2])
            pixel_size_y = float(lines[3])
            upper_left_x = float(lines[4])
            upper_left_y = float(lines[5])

        coord_x = upper_left_x + pixel_x * pixel_size_x + pixel_y * rotation_x
        coord_y = upper_left_y + pixel_x * rotation_y + pixel_y * pixel_size_y

        if GISProcessor.gis_wgs84:
            coord_x, coord_y = GISProcessor.convert_tm_wgs(coord_x, coord_y)

        return coord_x, coord_y

    def process_json_and_calculate_coordinates(self, json_path, tfw_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        results = GISProcessor.populate_gis_from_dict(data, tfw_path)
        return results
    
    def process_json_and_calculate_polygon(self, json_path, tfw_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        results = GISProcessor.populate_gis_from_polygon(data, tfw_path)
        return results

    def save_results_with_gis_to_json(self, results_center, results_polygon, json_file_path, output_file_path):
        with open(json_file_path, 'r') as file:
            _data = json.load(file)

        data = copy.deepcopy(_data)

        data['center_gis'] = results_center
        data['masks_gis'] = results_polygon

        with open(output_file_path, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        print(f"JSON 파일을 저장하였습니다: {output_file_path}")

    def populate_gis(self):
        for fn in tqdm(os.listdir(self.json_dir)):
            if ".json" not in fn:
                continue
            json_file_path = os.path.join(self.json_dir, fn)
            tfw_file_name = fn.replace(".json", ".tfw")
            tfw_file_path = os.path.join(self.input_dir, tfw_file_name)

            if tfw_file_name not in os.listdir(self.input_dir):
                print(f"{tfw_file_path} 파일이 {self.input_dir} 에 없습니다.")
                continue

            results_center = self.process_json_and_calculate_coordinates(json_file_path, tfw_file_path)
            results_polygon = self.process_json_and_calculate_polygon(json_file_path, tfw_file_path)

            if self.print_results:
                print(f"{fn}: {results_center}")
                print(f"{fn}: {results_polygon}")
            else:
                output_file_path = os.path.join(self.output_gis_dir, fn)
                self.save_results_with_gis_to_json(results_center, results_polygon, json_file_path, output_file_path)       

    @staticmethod
    def populate_gis_from_dict(preds_dict, tfw_path):
        """
        preds_dict: {
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
        
        tfw_path: str

        sample output:
        [
            {"latitude": 37.123456, "longitude": 127.123456},
            {"latitude": 37.123456, "longitude": 127.123456},
            ...
        ]

        """
        centers = preds_dict.get("center", [])
        if not centers:
            centers = [GISProcessor.get_center_bbox(bbox) for bbox in preds_dict["bboxes"]]

        results = []
        for center in centers:
            pixel_x = center["x"]
            pixel_y = center["y"]
            latitude, longitude = GISProcessor.get_lat_lon_from_pixel(tfw_path, pixel_x, pixel_y)
            results.append({"latitude": latitude, "longitude": longitude})

        return results
    
    def get_lat_lon_from_pixel_param(pixel_x, pixel_y, pixel_size_x, rotation_x, rotation_y, pixel_size_y, upper_left_x, upper_left_y):
        coord_x = upper_left_x + pixel_x * pixel_size_x + pixel_y * rotation_x
        coord_y = upper_left_y + pixel_x * rotation_y + pixel_y * pixel_size_y

        if GISProcessor.gis_wgs84:
            coord_x, coord_y = GISProcessor.convert_tm_wgs(coord_x, coord_y)

        return coord_x, coord_y
    
    @staticmethod
    def populate_gis_from_polygon(preds_dict, tfw_path):
        """
        preds_dict: {
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

        tfw_path: str

        sample output:
        [
            {"mask_gis": [[37.123456, 127.123456, 37.123456, 127.123456, ...], 123456]},
            {"mask_gis": [[37.123456, 127.123456, 37.123456, 127.123456, ...], 123456]},
            ...
        ]

        """

        with open(tfw_path, 'r') as tfw_file:
            lines = tfw_file.readlines()
            pixel_size_x = float(lines[0])
            rotation_x = float(lines[1])
            rotation_y = float(lines[2])
            pixel_size_y = float(lines[3])
            upper_left_x = float(lines[4])
            upper_left_y = float(lines[5])

        results = []
        for mask in preds_dict.get("masks", []):
            mask_gis = []
            polygons = mask.get("polygon", [])
            for polygon in polygons:
                polygon_gis = []
                for i in range(0, len(polygon), 2):
                    x, y = polygon[i], polygon[i + 1]
                    latitude, longitude = GISProcessor.get_lat_lon_from_pixel_param(x, y, pixel_size_x, rotation_x, rotation_y, pixel_size_y, upper_left_x, upper_left_y)
                    polygon_gis.append(latitude)
                    polygon_gis.append(longitude)
                mask_gis.append(polygon_gis)
                mask_gis.append(mask.get("area", 0))
            results.append({"polygon_gis": mask_gis})
        return results
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=str, required=True, help="JSON 파일이 있으며, centers 키가 있어야 합니다.")
    parser.add_argument("--input-dir", type=str, required=True, help="TFW 파일이 있어야 합니다.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="위도 및 경도를 저장할 디렉토리입니다.")
    parser.add_argument("--print", action="store_true", default=False, help="위도 및 경도를 출력합니다.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor = GISProcessor(args.json_dir, args.input_dir, args.out_dir, args.print)
    processor.populate_gis()