import tempfile
import threading
from pathlib import Path

import atexit
from queue import Queue

import numpy as np

from sync_thread import SyncThread

SHADOW_LIST = []


def close_shadows():
    for x in SHADOW_LIST: x.close()


atexit.register(close_shadows)


class ShadowedNumpyMemmap(SyncThread):
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
        super(ShadowedNumpyMemmap, self).__init__()
        
        self.__src_data = data
        self.__sync_chunk_size = sync_chunk_size

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

    def _sync_thread_create_item_generator(self):
        return range(len(self))

    def _sync_thread_item_count(self):
        return len(self)

    def __len__(self):
        return self.__memmap.shape[0]

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
        super(ShadowedNumpyMemmap, self).close()
        if self.__memmap is not None:
            del self.__memmap
            self.__memmap = None
