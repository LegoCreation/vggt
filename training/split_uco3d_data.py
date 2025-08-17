import tqdm
import os
import glob
import json

DATASET_DIR = "/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new"
TRAIN_PERCENT = 0.85

# Gather all data
full_dataset = {}
train_dataset = {}
test_dataset = {}
for category_path in tqdm.tqdm(list(glob.glob(f"{DATASET_DIR}/*/"))):
    category = category_path.split('/')[-2]
    for object_path in tqdm.tqdm(list(glob.glob(f"{category_path}/*/"))):
        object_name = object_path.split('/')[-2]
        full_dataset[object_name] = {}
        for sequence_path in glob.glob(f"{object_path}/*/"):
            sequence = sequence_path.split('/')[-2]
            camera_path = os.path.join(sequence_path, 'camera_data.npz')
            if os.path.exists(camera_path):
                sub_path = os.path.join(category, object_name, sequence)
                annotation = {}
                annotation['camera_data'] =  os.path.join(sub_path, 'camera_data.npz')
                annotation['image_paths'] = []
                for idx, image in enumerate(sorted(os.listdir(f"{sequence_path}/images"))):
                    annotation['image_paths'].append((os.path.join(sub_path, 'images', f'frame{idx:06d}.jpg'), idx))
                full_dataset[object_name][sequence] = annotation
        gathered_objects = list(full_dataset[object_name].items())
        num_train = int(round(len(gathered_objects) * TRAIN_PERCENT))
        train_dataset = {**train_dataset, **dict(gathered_objects[:num_train])}
        test_dataset = {**test_dataset, **dict(gathered_objects[num_train:])}

# Split into train and test based on percent
with open('./tests_new.json', 'w') as f:
    json.dump(test_dataset, f, indent=2)

with open('./train_new.json', 'w') as f:
    json.dump(train_dataset, f, indent=2)

print(f"Wrote {len(train_dataset.keys())} train sequences and {len(test_dataset.keys())} tests sequences")