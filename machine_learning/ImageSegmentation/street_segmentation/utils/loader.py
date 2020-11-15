import os

# R/utils
def load_images(img_dir):
    imgs = {}
    jsons = {}
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                full_file = os.path.join(root, file)
                file_key = full_file.split(".")[0]  # Will make comparisions easier
                imgs[file_key] = full_file
            elif file.endswith(".json"):
                full_file = os.path.join(root, file)
                file_key = full_file.split(".")[0]  # Will make comparisions easier
                jsons[file_key] = full_file  
    print("Len imgs: ", len(imgs.keys()), " Len jsons: ", len(jsons.keys()))
    keys_to_delete = []
    for key in imgs.keys():
        if key not in jsons.keys():
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del imgs[key]
    print("Len imgs: ", len(imgs.keys()), " Len jsons: ", len(jsons.keys()))
    return imgs, jsons