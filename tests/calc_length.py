import json

def count_values_in_key(json_data, key):
    count = 0
    length = len_values_in_key(json_data, key)
    for item in json_data.get(key, []):
        if len(item) == 0:
            count += 1
    return count

def len_values_in_key(json_data, key):
    count = json_data.get(key, 0)
    return len(count)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Example usage
file_path = './outputs/empty/preds/36608088.json'
json_data = read_json_file(file_path)
key_to_count = "bboxes"
value_count = len_values_in_key(json_data, key_to_count)
print(f"The key '{key_to_count}' appears {value_count} times.")
key_to_count = "centers_gis"
value_count = len_values_in_key(json_data, key_to_count)
print(f"The key '{key_to_count}' appears {value_count} times.")
key_to_count = "bbox_lat_lon"
value_count = len_values_in_key(json_data, key_to_count)
print(f"The key '{key_to_count}' appears {value_count} times.")