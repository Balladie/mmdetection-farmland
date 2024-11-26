import argparse
import json

def count_values_in_key(json_data, key):
    count = 0
    for item in json_data.get(key, []):
        if len(item) == 0:
            count += 1
    return count

def len_values_in_key(json_data, key):
    if key not in json_data:
        return 0
    count = json_data.get(key, 0)
    return len(count)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def main():
    parser = argparse.ArgumentParser(description="Process a JSON file and count values for specific keys.")
    parser.add_argument("--file", required=True, help="Path to the JSON file")
    args = parser.parse_args()
    json_data = read_json_file(args.file)

    key_to_count = "bboxes"
    value_count = len_values_in_key(json_data, key_to_count)
    print(f"The key '{key_to_count}' appears {value_count} times.")
    key_to_count = "centers_gis"
    value_count = len_values_in_key(json_data, key_to_count)
    print(f"The key '{key_to_count}' appears {value_count} times.")
    key_to_count = "bbox_lat_lon"
    value_count = len_values_in_key(json_data, key_to_count)
    print(f"The key '{key_to_count}' appears {value_count} times.")


if __name__ == "__main__":
    main()