import numpy as np



def stack_array_pictures(img_one: np.ndarray,img_two: np.ndarray,img_three: np.ndarray,axis: int = 0) -> np.ndarray:
    """
     Combine three 2D arrays into a single 3D array.

    Args:
        img_one, img_two, img_three: 2D arrays of shape (H, W)
        axis: axis to stack along, default 0 → (C, H, W)

    Returns:
        3D numpy array of shape (3, H, W) if axis=0
    """
    assert img_one.shape == img_two.shape == img_three.shape,\
        "All input arrays must have the same shape"

    assert img_one.ndim == 2, \
        "Input arrays must be 2D (H, W)"

    x = np.stack([img_one, img_two, img_three], axis=axis)
    return x