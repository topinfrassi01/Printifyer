from sklearn.cluster import KMeans
from skimage import io
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


def cluster_image(image, n_clusters, return_image_as_cluster_indices=True):
    clustering = KMeans(n_clusters)
    
    if return_image_as_cluster_indices:
        image = clustering.fit_predict(image)
    else:
        image = clustering.fit_tranform(image)
    
    return image, clustering.cluster_centers_


def visualize_clustering(image, clustered_image, save_to=None):
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(rescale_intensity(clustered_image, dtype='uint8'))

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)
        plt.close()


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the image to cluster.")
    argument_parser.add_argument("-c", "--n_clusters", type=int, default=10, help="Number of clusters to create.")
    argument_parser.add_argument("-v", "--verbose", action="store_true", help="Sets the verbose mode.")
    argument_parser.add_argument("-s", "--save_path", type=str, help="Folder where to save the clustered image using the created clusters")

    arguments = argument_parser.parse_args()

    image_path = Path(arguments.image_path)
    save_path = None if arguments.save_path is None else Path(arguments.save_path)

    image = io.imread(str(image_path))

    clustered_image, clusters = cluster_image(image, arguments.n_clusters, False)

    if arguments.verbose:
        visualize_clustering(image, clustered_image)

    if save_path:
        io.imsave(save_path / (image_path.stem + '_custered' + image_path.suffix), clustered_image)
