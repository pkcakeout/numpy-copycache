import queue
import threading


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

    def _sync_thread_item_count(self):
        raise NotImplementedError()

    def _sync_thread_create_item_generator(self):
        raise NotImplementedError()

    def _sync_is_item_synced(self, item):
        raise NotImplementedError()

    def _sync_item(self, item):
        raise NotImplementedError()

    def _sync_bypass(self, item):
        with self.__request_lock:
            if self.__bypass is not None:
                raise RuntimeError("self.__bypass must be None")
            self.__bypass = item
            self.__queue.put('bypass')
            self.__response_queue.get(block=True)

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
                        self._sync_item(sync_item)
                        self.__synced_items += 1

                if item is None:
                    continue
                elif item == 'stop':
                    break
                elif item == 'bypass':
                    bypass = self.__bypass
                    self.__bypass = None
                    self.__response_queue.put(do_sync(bypass))
                elif item == 'next':
                    try:
                        do_sync(item_generator.next())
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
