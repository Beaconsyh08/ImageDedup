from imagededup.methods import PHash, AHash, DHash, WHash, CNN
from tqdm import tqdm
import os
from imagededup.utils import CustomModel
from imagededup.utils.models import ViT, MobilenetV3, EfficientNet

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

def process(encoder, method: str, image_dir: str):
    # Generate encodings for all images in an image directory
    encodings = encoder.encode_images(image_dir=image_root)
    print(encodings)
    
    # if method.endswith("Hash"):
    #     duplicates = encoder.find_duplicates(encoding_map=encodings, max_distance_threshold=0)
    # else:
    #     duplicates = encoder.find_duplicates(encoding_map=encodings)
        
    # duplicates = {key: value for key, value in duplicates.items() if len(value) > 0}
    # print(len(duplicates))

    # if method.endswith("Hash"):
    #     duplicates_to_remove = encoder.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=0)
    # else:
    #     duplicates_to_remove = encoder.find_duplicates_to_remove(encoding_map=encodings)
        
    # print(duplicates_to_remove)


# # plot duplicates obtained for a given file using the duplicates dictionary
# from imagededup.utils import plot_duplicates
# for key in tqdm(duplicates.keys()):
#     plot_duplicates(image_dir=image_root,
#                     duplicate_map=duplicates,
#                     filename=key,
#                     outfile="%s/%s" % (output_root, key))
    
    
if __name__ == "__main__":
    # image_root = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/5000/imgs"
    image_root = "/mnt/ve_share/songyuhao/generation/data/exp"
    # output_root = "/mnt/ve_share/songyuhao/generation/data/result/dedup/exp/"
    # os.makedirs(output_root, exist_ok=True)

    # ["PHash", "AHash", "DHash", "WHash", "ViT", "MobilenetV3", "EfficientNet"]
    method = "PHash"

    encoder = initialization(method=method)
    process(encoder=encoder, method=method, image_dir=image_root)
    

