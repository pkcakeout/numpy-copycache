import tempfile
from pathlib import Path

import atexit
import numpy as np


SHADOW_LIST = []


def close_shadows():
    for x in SHADOW_LIST: x.close()


atexit.register(close_shadows)


class ShadowedNumpyMemmap:
    """
    This class clones a subset of numpy functions in order to create a
    memory-mapped copy of another numpy array. The class only supports
    a subset of the functions a numpy array supports:

     - __getitem__
     - __len__
     - shape
     - ndim
     - size
     - dtype

    When accessing a member for the first time it is read from the source
    numpy array and put into the memory mapped local cache file. The second time
    an item is accessed, it will  be read from the local cache file.
    """
    def __init__(
            self,
            data: np.ndarray,
            cache_location=None,
            sync_chunk_size: int=64*1024):
        self.__src_data = data

        filemode = "r+"
        if cache_location is None:
            filemode = "w+"
            cache_location = tempfile.NamedTemporaryFile()

        self.__dtype = data.dtype
        self.__memmap = np.memmap(
            cache_location,
            dtype=self.__src_data.dtype,
            shape=self.__src_data.shape,
            mode=filemode,
        )
        self.__loaded_indexes = set()

    @property
    def copy_ratio(self):
        return len(self.__loaded_indexes) / self.shape[0]

    @property
    def shape(self):
        return self.__memmap.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.__dtype

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__src_data[item]
        elif isinstance(item, slice):
            return self.__src_data[item]
        else:
            item = tuple(item)
            result = self[item[0]]
            if len(item) == 1:
                return result
            elif len(item) == 2:
                return result[item[1]]
            else:
                return [result[1:]]

    def close(self):
        """
        Closes the internal memmap and releases associated resources. Please
        note that this will make the object unusable. All data access functions
        will raise an IOError after executing this function.
        """
        if self.__memmap is not None:
            del self.__memmap
            self.__memmap = None
