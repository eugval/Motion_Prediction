
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
import matplotlib.pyplot as plt
import h5py
import numpy as np
from Mask_RCNN.mrcnn import visualize
from preprocessing.utils import make_colormap, find_start_count

from matplotlib import patches
from scipy.stats import multivariate_normal


from skimage.measure import find_contours
from matplotlib.patches import Polygon





def find_centroid(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    return int(x),int(y)


def add_centroids(data_file, f = None):
    close = False
    if (f == None):
        close = True
        f = h5py.File(data_file, "r+")

    start_count = find_start_count(list(f.keys()))

    for i in range(start_count, f['frame_number'].value[0]):
        frame = "frame{}".format(i)


        bboxes = f[frame]['rois'].value
        centroids = [find_centroid(bbox) for bbox in bboxes]

        centroid_key = "{}/centroids".format(frame)

        if (centroid_key in f):
            f[centroid_key][:]=centroids
        else:
            f.create_dataset(centroid_key, data=centroids)
    if(close):
        f.close()

def make_gaussian_masks(data_file, use_covariance=False ,verbose = 0):
    f = h5py.File(data_file, "r+")

    if('frame1/centroids' not in f):
        add_centroids(data_file,f)

    start_count = find_start_count(list(f.keys()))

    image_dims = (f["frame1"]['image'].shape[0] , f["frame1"]['image'].shape[1])

    for i in range(start_count, f['frame_number'].value[0]):
        frame = "frame{}".format(i)

        if(verbose==1): print("treating frame {}".format(i))

        gauss_key = "{}/gaussians".format(frame)

        if (gauss_key not in f):
            f.create_dataset(gauss_key, f[frame]["masks"].value.shape)

        if (verbose == 2): print("masks {}".format(f[frame]['masks'].value.shape))
        if (verbose == 2): print("centroids {}".format(f[frame]['centroids'].value.shape))
        if (verbose == 2): print("rois {}".format(f[frame]['rois'].value.shape))

        for mask_idx in range(f[frame]['masks'].value.shape[2]):
            mask = f[frame]['masks'].value[:,:,mask_idx]
            pix_pos_tuple = np.where(mask>0)
            pix_pos_list = [i for i in zip(pix_pos_tuple[0],pix_pos_tuple[1])]
            pix_pos = np.array(pix_pos_list).T


            if(not np.sum(use_covariance) > 0):
                cov = np.cov(pix_pos)
            else:
                cov = use_covariance

            mean = f[frame]['centroids'].value[mask_idx]

            x = np.broadcast_to(np.arange(image_dims[0]).reshape(image_dims[0],1), (image_dims[0], image_dims[1]))
            y = np.broadcast_to(np.arange(image_dims[1]),(image_dims[0], image_dims[1]))
            d = np.dstack((x,y))
            v = np.reshape(d, (-1, 2))
            p = multivariate_normal.pdf(v, mean, cov)
            gauss_mask = p.reshape(image_dims[0], image_dims[1])

            f[frame]['gaussians'][:,:,mask_idx] = gauss_mask


    f.close()



def apply_centroid(image, x, y, verbose =0):
    if (verbose == 1):
        print(image.shape)
        print(x, y)

    if(x-3>0 and y-3>0 and x+3<image.shape[0] and y+3<image.shape[1]):
        for c in range(3):
            image[x-2:x+2,y-2:y+2,c] = np.zeros((4,4))

    return image





def apply_gaussian(image, gaussian, color):
    """Apply the given mask to the image.
    """

    alpha = 0.8/ np.max(gaussian)
    for c in range(3):
        image[:, :, c] =  image[:, :, c] * (1 - alpha*gaussian) + alpha* gaussian * color[c] * 255
    return image

def save_gaussians(image,save_path, gaussians,centroids, boxes, title="",
                      figsize=(16, 16),
                      colors=None, captions=None, show_bbox=True, extra_padding = 10, verbose=0):
    """
    gaussians: [height, width, num_instances]
    title: (optional) Figure title
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = gaussians.shape[2]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert centroids.shape[0] == gaussians.shape[-1]==boxes.shape[0]

    _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height+ extra_padding, -extra_padding)
    ax.set_xlim(-extra_padding, width+extra_padding)
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

        x,y = centroids[i]



        # Label
        if(captions):
            caption = captions[i]
            ax.text(y, x + 10, caption,
                    color='w', size=11, backgroundcolor="none")

        # Mask
        mask = gaussians[:, :, i]
        masked_image = apply_gaussian(masked_image, mask, color)

        # Centroids

        masked_image = apply_centroid(masked_image,x,y, verbose=verbose)

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
    plt.savefig(save_path , bbox_inches='tight', pad_inches = 0)
    plt.close()




def visualise_gaussians(data_file, target_folder, id_idx = 0, captions = True, verbose =0):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    f=h5py.File(data_file, "r")
    #Get a colormap for the ID
    tracks_n = f["tracks_n"].value[0]
    colormap = make_colormap(int(tracks_n+1))


    #Save visualisations
    #TODO: Get rid of the rest of the padding on the saved visualisations
    #TODO: FIX THE CAPTION PROBLEM

    start_count = find_start_count(list(f.keys()))

    for i in range(start_count, f['frame_number'].value[0]):
        frame = "frame{}".format(i)
        r = f[frame]
        image = r['image'].value
        IDs = r['IDs'].value[:,id_idx]

        colors = [colormap[int(j)] for j in IDs]

        if(captions):
            captions = ['ID: {}'.format(IDs[i]) for i in range(IDs.shape[0])]
        else:
            captions = None

        save_path = os.path.join(target_folder,"{}.jpg".format(frame))

        save_gaussians(image, save_path, r['gaussians'].value, r['centroids'].value,r['rois'].value,
                       colors=colors, captions=captions, verbose= verbose)


    f.close()







if __name__ =='__main__':
    gauss = True
    gauss_vis = True

    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    name = "Football1_sm6"

    data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name, name))
    class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))

    tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name, name))
    tracked_file_c = os.path.join(PROCESSED_PATH, "{}/{}_tracked_c.hdf5".format(name, name))
    resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name, name))
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
    set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))

    target_folder = os.path.join(PROCESSED_PATH, "{}/mask_images/".format(name))
    target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
    target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))

    if(gauss):
        print("Making gaussian masks...")
        make_gaussian_masks(resized_file, verbose =0)

    if(gauss_vis):
        print("Visualising gaussians...")
        visualise_gaussians(resized_file,target_folder)