import tempfile

import atexit
from pathlib import Path
from typing import Tuple, Generator

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

    def _sync_thread_create_item_generator(self) -> Generator:
        return iter(range(len(self)))

    def _sync_thread_item_count(self) -> int:
        return len(self)

    def _sync_item(self, item) -> int:
        self.__memmap[item] = self.__src_data[item]
        if isinstance(item, int):
            self.__loaded_indexes.add(item)
            return 1
        elif isinstance(item, slice):
            start, stop, step = self.__resolve_slice(item)
            new_items = set(range(start, stop, step))
            new_items.difference_update(self.__loaded_indexes)
            self.__loaded_indexes.update(new_items)
            return len(new_items)
        else:
            raise ValueError("Cannot resolve index")

    def _sync_is_item_synced(self, item) -> bool:
        if isinstance(item, int):
            return item in self.__loaded_indexes
        elif isinstance(item, slice):
            start, stop, step = self.__resolve_slice(item)
            if stop - start <= 0:
                return True
            return all(
                i in self.__loaded_indexes
                for i in range(start, stop, step))
        else:
            raise ValueError("Cannot resolve index")

    def __resolve_slice(self, s: slice) -> Tuple[int, int, int]:
        step = s.step if s.step else 1
        start = s.start if s.start else 0
        if start < 0:
            start = max(0, len(self) + start)
        stop = s.stop if s.stop is not None else len(self)
        if stop < 0:
            stop = max(0, len(self) + stop)
        if step < 0:
            step = abs(step)
            start, stop = stop, start
        return start, stop, step

    def __len__(self):
        return self.__memmap.shape[0]

    @property
    def filename(self):
        return Path(self.__memmap.filename)

    @property
    def shape(self):
        return self.__memmap.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.__dtype

    def __defer_getitem(self, item):
        if self.fully_copied:
            return self.__memmap[item]
        self._sync_bypass(item)
        return self.__memmap[item]

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.__defer_getitem(item)
        else:
            item = tuple(item)
            result = self.__defer_getitem(item[0])
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
