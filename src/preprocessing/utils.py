import colorsys
import random
import matplotlib.pyplot as plt

def find_start_count(key_list):
    if("frame0" in key_list):
        return 0
    elif("frame1" in key_list):
        return 1
    else:
        raise ValueError("Naming Convention failure for the frames.")

def make_colormap(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colormap = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colormap)
    return colormap

def verify_ID_uniqueness(file, id_idx =0):
    #TODO: Fix it, this function is completelly broken (you want track not id to be unique)
    start_count = find_start_count(list(file.keys()))

    frame_indices = range(start_count, file['frame_number'].value[0])

    ids = {}

    for i in frame_indices:
        frame = "frame{}".format(i)
        for ID in file[frame]['IDs']:
            if(ID[id_idx] in ids):
                print(ID[id_idx])
                print(ids)
                return False
            else:
                ids[ID[id_idx]]=True
    return True


def visualise_image(image, save_path = None):
    plt.imshow(image)
    if(save_path):
        plt.savefig(save_path , bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
