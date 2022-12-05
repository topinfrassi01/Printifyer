import argparse
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_holes, binary_closing, binary_dilation, disk
from skimage.filters import median


def post_process_cluster(clustered_image):
    processed = np.zeros_like(clustered_image)
    for i in range(clustered_image.shape[-1]):
        processed[:,:,i] = post_process_cluster_mask(clustered_image[:,:,i].astype(bool))

    return processed


def post_process_cluster_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)

    mask = median(mask, np.ones((3,3)))

    closing_morph_element = np.ones((5,5), dtype=bool)
    mask = binary_closing(mask, closing_morph_element)

    holes_closing_morph_element = np.ones((20,20), dtype=bool)
    mask = remove_small_holes(mask, holes_closing_morph_element)

    return mask


def create_outline_from_mask(mask: np.ndarray, outline_thickness=3) -> np.ndarray:
    mask = mask.astype(bool)

    outline_mask = disk(outline_thickness)
    dilated = binary_dilation(mask, outline_mask)

    return np.clip(dilated - mask, a_min=0, a_max=1)


def cluster_single_image(image: np.ndarray,
                         kmeans:Optional[KMeans]=None,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Groups the intensity values of the image by n clusters using a KMeans algorithm.

    :param image: Image to cluster. Can be in any color mode, must be of shape [H,W,C] or [H,W] in case of a grayscale image.
    :param kmeans: Trained KMeans classifier. If None, one will be trained from the image.
    :param return_image_as_cluster_indices: If True, will return the image with the cluster indices instead of the cluster values.
    :param **kwargs: Parameters applied to the KMeans algorithm if kmeans is None.
    :return: The clustered image and the clusters created by the KMeans argument
    :raises: ValueError if image doesn't have the expected shape.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape is [H,W,C], but found {image.ndim} dimensions.")

    if kmeans is None:
        kmeans = KMeans(**kwargs)

    image = kmeans.fit_predict(image)

    return image, kmeans.cluster_centers_


def one_hot_clusters_in_image(clustered_image: np.ndarray, n_clusters: Optional[int]=None) -> Tuple[np.ndarray, OneHotEncoder]:
    """
    Transforms an image of shape [H, W, 1] to an image of shape [H, W, n_clusters].

    :param image: Array of shape [H, W, 1] where an element's value is a cluster ID
    :param n_clusters: Optional parameter used only if some cluster values do not exist in the image.
                       This might happen if clusters trained on an image are used on another image.
                       If the parameter is None, we assume n_clusters = clustered_image.max()
    :return: The one hot encoded image of shape [H, W, n_clusters]
    """
    image_shape = clustered_image.shape
    flattened_image = clustered_image.flatten()

    categories_names = list(map(str, range(n_clusters if n_clusters is not None else flattened_image.max())))
    encoder = OneHotEncoder(categories=categories_names)
    encoded = encoder.fit_transform(flattened_image)

    return np.reshape(encoded, image_shape), encoder


def visualize_clustering(image: np.ndarray,
                         clustered_image: np.ndarray,
                         save_to: Optional[Path]=None) -> None:
    """
    Visualize the clustering applied to an image.

    :param image: Original image
    :param clustered_image: Clustered image
    :save_to: Path where the comparison figure should be saved. If this path is None, the image is shown instead.
    """
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(rescale_intensity(clustered_image, out_range='uint8'))

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)
        plt.close()


def main():
    """
    Parses arguments, clusters the image, visualize it and saves it depending on the verbosity and the save_path argument.
    """
    argument_parser = argparse.ArgumentParser(description="Clusters an image intensity values and saves it to a folder or ")

    argument_parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the image to cluster.")
    argument_parser.add_argument("-c", "--n_clusters", type=int, default=10, help="Number of clusters to create.")
    argument_parser.add_argument("-v", "--verbose", type=int, default=0, help="""
        Sets the verbose mode.
        0 : No verbosity, clustered image isn't shown.
        1 : No verbosity, Plots the clustered image.
        2 : Fully verbose, Saves the clustered image comparison.
    """)
    argument_parser.add_argument("-s", "--save_path", type=str, help="Folder where to save the clustered image using the created clusters")

    arguments = argument_parser.parse_args()

    image_path = Path(arguments.image_path)
    save_path = None if arguments.save_path is None else Path(arguments.save_path)

    image = io.imread(str(image_path))

    clustered_image, _ = cluster_single_image(image, **{'n_clusters':arguments.n_clusters, 'verbose': arguments.verbose})

    one_hot_img, encoder = one_hot_clusters_in_image(clustered_image)

    processed_img = post_process_cluster(one_hot_img)

    clustered_image = encoder.inverse_transform(one_hot_img)

    if arguments.verbose > 0:
        visualize_clustering(image, clustered_image, (image_path.parent / image_path.stem + '_comp_clusters' + image_path.suffix) if arguments.verbose == 2 else None)

    if save_path:
        io.imsave(save_path / (image_path.stem + '_clustered' + image_path.suffix), clustered_image)

if __name__ == '__main__':
    main()
