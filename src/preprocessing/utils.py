import colorsys
import random
import matplotlib.pyplot as plt
import os
import sys
import h5py

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib import patches,  lines
import numpy as np




def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def save_instances(image,save_path, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16),
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, extra_padding = 10):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + extra_padding, -extra_padding)
    ax.set_xlim(-extra_padding, width + extra_padding)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), interpolation='nearest')
    plt.axis("off")
    plt.savefig(save_path , bbox_inches='tight')
    plt.close()


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


def visualise_image(image, save_path = None, centroid_coordinates = None):
    if(centroid_coordinates is not None):
        if(len(image.shape) ==2):
            image[centroid_coordinates[0]-2:centroid_coordinates[0]+2,centroid_coordinates[1]-2:centroid_coordinates[1]+2]=0
        else:
            image[centroid_coordinates[0]-2:centroid_coordinates[0]+2,centroid_coordinates[1]-2:centroid_coordinates[1]+2]=0

    plt.imshow(image)
    if(save_path):
        plt.savefig(save_path , bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()



def incorporate_ratio(initial_dims,max_height, max_width):
    initial_height = initial_dims[0]
    initial_width = initial_dims[1]
    ratio = min(max_height/initial_height, max_width/initial_width)

    new_h = ratio*initial_height
    new_w = ratio * initial_width

    return int(new_h), int(new_w), ratio




def visualise_masks(data_file, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    f=h5py.File(data_file, "r")

    start_count = find_start_count(list(f.keys()))

    for i in range(start_count, f['frame_number'].value[0]):
        frame = "frame{}".format(i)
        r = f[frame]
        image = r['image'].value
        class_names = f["class_names"]


        save_path = os.path.join(target_folder,"{}.jpg".format(frame))
        save_instances(image, save_path, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'])


    f.close()
