import numpy as np


class AtomList(object):
    def __init__(self, elements: np.ndarray, coordinates: np.ndarray):
        """
        Parameters
        ----------
        elements: array
            in shape (n_elems, ).
            specifies elements by their element number Z.
        coordinates: array
            in shape (..., n_elems, 3).
            specifies batched coordinates for the elements.

        Notice
        ------
        the coodinates can have batch dimensions before dim = -2. In other words, each element in elements
        can hanve multiple locations. Specifically, `elements[i]` occurs at locations `coordinates[..., i, :]`
        """
        self.elements = elements
        self.coordinates = coordinates

