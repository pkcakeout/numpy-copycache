import queue
import threading
from typing import Generator


class SyncingFailedException(Exception): pass


class SyncThread:
    def __init__(self):
        self.__background_syncing = False
        self.__synced_items = 0

        self.__bypass = None

        self.__request_lock = threading.RLock()
        self.__response_queue = queue.Queue()
        self.__queue = queue.Queue()
        self.__sync_thread = threading.Thread(target=self.__sync_thread_main)
        self.__sync_thread.daemon = True
        self.__sync_thread.start()

    def _sync_thread_item_count(self) -> int:
        raise NotImplementedError()

    def _sync_thread_create_item_generator(self) -> Generator:
        raise NotImplementedError()

    def _sync_is_item_synced(self, item) -> bool:
        raise NotImplementedError()

    def _sync_item(self, item) -> int:
        raise NotImplementedError()

    def _sync_bypass(self, item):
        with self.__request_lock:
            if self.fully_copied:
                return
            if self._sync_is_item_synced(item):
                return

            if self.__bypass is not None:
                raise RuntimeError("self.__bypass must be None")
            self.__bypass = item
            self.__queue.put('bypass')

            while True:
                if not self.__sync_thread.isAlive():
                    if self.fully_copied:
                        self.__bypass = None
                        break  # This can happen because of a race condition
                    else:
                        raise SyncingFailedException(
                            "Item not synced but: SyncThread died")
                try:
                    self.__response_queue.get(block=True, timeout=.1)
                except queue.Empty:
                    pass
                else:
                    break

    @property
    def background_syncing(self):
        return self.__background_syncing

    @background_syncing.setter
    def background_syncing(self, value):
        self.__background_syncing = bool(value)
        self.__queue.put(None) # knock loose

    @property
    def copy_ratio(self):
        """
        Returns the ratio of keys that have been copied locally. This value
        is updated asynchronously.
        """
        return self.__synced_items / self._sync_thread_item_count()

    @property
    def fully_copied(self):
        """
        True if all data has been copied locally.
        """
        return self.copy_ratio >= 1.0

    def close(self):
        """
        Actively stops asynchronous syncing thread and waits for it to complete
        all pending actions.
        """
        if self.__sync_thread.isAlive():
            if not self.__background_syncing:
                self.__queue.put('stop')
            self.__sync_thread.join()

    def __sync_thread_main(self):
        item_generator = None

        while (item_generator is None) or (not self.fully_copied):
            try:
                item = self.__queue.get(block=not self.__background_syncing)
            except queue.Empty:
                if self.__background_syncing:
                    item = 'next'
                else:
                    item = None
                task_done = lambda : None
            else:
                task_done = self.__queue.task_done

            if item_generator is None:
                item_generator = self._sync_thread_create_item_generator()
                if self.fully_copied:
                    break

            try:
                def do_sync(sync_item):
                    if not self._sync_is_item_synced(sync_item):
                        self.__synced_items += self._sync_item(sync_item)
                if item is None:
                    continue
                elif item == 'stop':
                    break
                elif item == 'bypass':
                    bypass = self.__bypass
                    self.__bypass = None
                    do_sync(bypass)
                    self.__response_queue.put("done")
                elif item == 'next':
                    try:
                        do_sync(next(item_generator))
                    except StopIteration:
                        if not self.fully_copied:
                            print(
                                "SyncThread Error: finished item enumeration"
                                "but syncing not finished")
                        break
                else:
                    print("SyncThread Error: received invalid task")
                    break
            finally:
                task_done()
