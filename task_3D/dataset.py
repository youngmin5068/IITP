import os


def data_dicts(image_path,label_path,size=100):
    image_subpaths = os.listdir(image_path)
    label_subpaths = os.listdir(label_path)
    image_list = [image_path + "/" + f for f in image_subpaths]
    label_list = [label_path + "/" + f for f in label_subpaths]
    image_list.sort()
    label_list.sort()
    image_list = image_list[:size]
    label_list = label_list[:size]
    data_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(image_list, label_list)]

    return data_dicts
