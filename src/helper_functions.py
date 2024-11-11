import numpy as np

def random_select(indices, num_select):
    """
     randomly select a specified number of points from given indices.
    """
    if indices.size == 0:
        return np.empty((0, 2), dtype=int)
    num_select = min(num_select, indices.shape[0])
    selected = indices[np.random.choice(indices.shape[0], num_select, replace=False)]
    return selected