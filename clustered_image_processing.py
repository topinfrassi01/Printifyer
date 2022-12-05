import numpy as np
from skimage.morphology import remove_small_holes, binary_closing, binary_dilation, disk
from skimage.filters import median


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