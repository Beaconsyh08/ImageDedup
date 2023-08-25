from imagededup.methods import PHash, AHash, DHash, WHash, CNN
from tqdm import tqdm
import os
import json
from imagededup.utils import CustomModel
from imagededup.utils.models import ViT, MobilenetV3, EfficientNet
from pathlib import Path
import time


def initialization(method: str):
    if method == "PHash":
        encoder = PHash()
    elif method == "AHash":
        encoder = AHash()
    elif method == "DHash":
        encoder = DHash()
    elif method == "WHash":
        encoder = WHash()
    elif method == "ViT":
        custom_config = CustomModel(name=ViT.name, model=ViT(), transform=ViT.transform)
        encoder = CNN(model_config=custom_config)
    elif method == "MobilenetV3":
        custom_config = CustomModel(name=MobilenetV3.name, model=MobilenetV3(), transform=MobilenetV3.transform)
        encoder = CNN(model_config=custom_config)
    elif method == "EfficientNet":
        custom_config = CustomModel(name=EfficientNet.name, model=EfficientNet(), transform=EfficientNet.transform)
        encoder = CNN(model_config=custom_config)
    
    return encoder

def process(encoder, method: str, image_paths: str):
    # Generate encodings for all images in an image directory
    encodings = encoder.encode_images(image_paths=image_paths)
    print(encodings)
    
    # if method.endswith("Hash"):
    #     duplicates = encoder.find_duplicates(encoding_map=encodings, max_distance_threshold=0)
    # else:
    #     duplicates = encoder.find_duplicates(encoding_map=encodings, min_similarity_threshold=0.99)
        
    # duplicates = {key: value for key, value in duplicates.items() if len(value) > 0}
    # print("duplicateds", len(duplicates))
    # # print(duplicates)

    # if method.endswith("Hash"):
    #     duplicates_to_remove = encoder.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=0)
    # else:
    #     duplicates_to_remove = encoder.find_duplicates_to_remove(encoding_map=encodings, min_similarity_threshold=0.99)
        
    # print("to_remove", len(duplicates_to_remove))
    # duplicates_to_remove = [Path("/" + _) for _ in duplicates_to_remove]
    # left = list(set(image_paths) - set(duplicates_to_remove))
    # print(set(image_paths))
    # print(set(duplicates_to_remove))
    # print("left", len(left), [_ for _ in left])


    # # plot duplicates obtained for a given file using the duplicates dictionary
    # from imagededup.utils import plot_duplicates
    # for key in tqdm(duplicates.keys()):
    #     plot_duplicates(image_dir=image_root,
    #                     duplicate_map=duplicates,
    #                     filename=key,
    #                     outfile="%s/%s" % (output_root, key))
    
    
if __name__ == "__main__":
    # image_root = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/imgs"
    # image_root = "/mnt/ve_share/songyuhao/generation/data/exp"
    # image_root = "/mnt/ve_share/songyuhao/generation/data/icu30/bundle_and_clip_info/out_sampled_10f.txt"
    image_root = "/mnt/ve_share/songyuhao/generation/data/icu30/bundle_and_clip_info/exp_10.txt"
    # image_root = "/mnt/ve_share/songyuhao/generation/data/icu30/bundle_and_clip_info/out_sampled_10f.txt"
    # output_root = "/mnt/ve_share/songyuhao/generation/data/result/dedup/exp/"
    # os.makedirs(output_root, exist_ok=True)

    # ["PHash", "AHash", "DHash", "WHash", "ViT", "MobilenetV3", "EfficientNet"]
    method = "EfficientNet"
    viewpoints = ["front_middle_camera"]
    sampled_size = 10
    res_dir = "/mnt/ve_share/songyuhao/generation/data/icu30/%s_%d.txt" % (method, sampled_size)
    

    dict_paths = {key: [] for key in viewpoints}
    carids = []
    if image_root.endswith(".hds") or image_root.endswith(".txt"):
        print("Loading")
        with open(image_root, "r") as input_file:
            inputs = input_file.readlines()
            for json_path in inputs:
                if json_path not in carids:
                    carids.append(json_path.strip().split(",")[0].split("_")[0])
            json_paths = [_.strip().split(",")[-1] if _.startswith("HP") else _.strip() for _ in inputs]
            json_paths = [_ if _.startswith("/") else "/" + _ for _ in json_paths]

        # print(json_paths[:10])
        print("Total Json Amout: %d" % len(json_paths))
        for json_path in tqdm(json_paths):
            with open(json_path, "r") as input_json:
                json_obj = json.load(input_json)
                cam_objs = json_obj["camera"]
                for cam_obj in cam_objs:
                    if cam_obj["name"] in viewpoints:
                        dict_paths[cam_obj["name"]] += [Path("/" + cam_obj["oss_path"])]
                        
    print(dict_paths)
    front_middle = dict_paths[viewpoints[0]]
    for i in tqdm(range(0, len(front_middle), sampled_size)):
        group_list = front_middle[i : i+sampled_size]
        
        start_time = time.time()
                
        encoder = initialization(method=method)
        
        encodings = encoder.encode_images(image_paths=group_list)
        print(encodings)
        
        with open(res_dir, "w") as output_file:
            for enc in encodings.keys():
                output_file.writelines("%s#/%s#%s" % (carids[i], enc, list(encodings[enc])) + "\n")
        
        # process(encoder=encoder, method=method, image_paths=group_list)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time} seconds")
        print(res_dir)
        