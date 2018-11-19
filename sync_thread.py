import queue
import threading
import time
from typing import Generator


class SyncingFailedException(Exception): pass


class SyncThread:
    def __init__(self):
        self.__synced_items = 0

        self.__bypass = None

        self.__bandwidth_share = 0
        self.__item_sync_time = None

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
    def bandwidth_share(self):
        return self.__bandwidth_share

    @bandwidth_share.setter
    def bandwidth_share(self, value):
        self.__bandwidth_share = min(1., max(0., float(value)))
        self.__queue.put(None) # knock loose

    @property
    def sync_item_duration(self):
        """
        Gives the time it takes to synchornize a single item from the
        synchronization queue. None if it could not yet be measured.
        """
        return self.__item_sync_time

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
            self.__queue.put('stop')
            self.__sync_thread.join()

    def __sync_thread_main(self):
        item_generator = None
        remaining_wait = None

        while (item_generator is None) or (not self.fully_copied):
            try:
                timeout = remaining_wait
                if timeout is not None:
                    timeout = max(0, timeout)
                start_time = time.time()
                item = self.__queue.get(
                    block=self.__bandwidth_share != 1, timeout=timeout)
            except queue.Empty:
                remaining_wait = 0
                if self.__bandwidth_share > 0:
                    item = 'next'
                else:
                    item = None
                task_done = lambda : None
            else:
                end_time = time.time()
                task_done = self.__queue.task_done
                if remaining_wait is not None:
                    remaining_wait -= end_time - start_time

            if item_generator is None:
                item_generator = self._sync_thread_create_item_generator()
                if self.fully_copied:
                    break

            try:
                def do_sync(sync_item):
                    if not self._sync_is_item_synced(sync_item):
                        start_time = time.time()
                        synced_item_count = self._sync_item(sync_item)
                        self.__synced_items += synced_item_count
                        end_time = time.time()
                        if end_time == start_time:
                            # Assume we are dealing with the 16ms limitation
                            # on windows - default to 16ms.
                            delta_t = 0.016
                        else:
                            delta_t = end_time - start_time

                        delta_t /= synced_item_count
                        if self.__item_sync_time is None:
                            self.__item_sync_time = delta_t
                        else:
                            # Keep a running average
                            p = 0.95
                            self.__item_sync_time = \
                                p * self.__item_sync_time + (1 - p) * delta_t
                        return synced_item_count
                    return None

                if item is None:
                    if remaining_wait is None:
                        if self.__bandwidth_share > 0:
                            remaining_wait = 0
                    continue
                elif item == 'stop':
                    break
                elif item == 'bypass':
                    bypass = self.__bypass
                    self.__bypass = None
                    count = do_sync(bypass)
                    if count is None:
                        count = 0
                    if (remaining_wait is not None) and self.__item_sync_time:
                        remaining_wait -= count * self.__item_sync_time
                    self.__response_queue.put("done")
                elif item == 'next':
                    try:
                        count = do_sync(next(item_generator))
                        if count is None:
                            pass
                        elif self.__item_sync_time:
                            remaining_wait = count * self.__item_sync_time
                            if self.__bandwidth_share == 0:
                                remaining_wait = None
                            else:
                                remaining_wait = \
                                    remaining_wait / self.__bandwidth_share \
                                    - remaining_wait
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
