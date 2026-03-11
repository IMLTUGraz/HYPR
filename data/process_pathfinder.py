import numpy as np
import os
import pickle
import tensorflow as tf
import random


root_dir = "/PATH/TO/LRA_RELEASE/"
target_dir = "/path/to/datasets/PATHFINDER/"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for subdir in ["pathfinder32"]:
    for diff_level in ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]:
        data_dir = os.path.join(root_dir, subdir, diff_level)
        metadata_list = [
            os.path.join(data_dir, "metadata", file)
            for file in os.listdir(os.path.join(data_dir, "metadata"))
            if file.endswith(".npy")
        ]
        ds_list = []
        for idx, metadata_file in enumerate(metadata_list):
            print(idx, len(metadata_list), metadata_file, "\t\t", end = "\r")
            for inst_meta in tf.io.read_file(metadata_file).numpy().decode("utf-8").split("\n")[:-1]:
                metadata = inst_meta.split(" ")
                img_path = os.path.join(data_dir, metadata[0], metadata[1])
                img_bin = tf.io.read_file(img_path)
                if len(img_bin.numpy()) == 0:
                    print()
                    print("detected empty image")
                    continue
                img = tf.image.decode_png(img_bin)
                seq = img.numpy().reshape(-1).astype(np.int32)
                label = int(metadata[3])
                ds_list.append({"input_ids_0":seq, "label":label})

        with open(os.path.join(target_dir, f"{subdir}-{diff_level}.pickle"), "wb") as f:
            pickle.dump(ds_list, f)
