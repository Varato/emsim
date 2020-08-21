from typing import Iterable, Callable
from threading import Thread, Event
from queue import Queue
import numpy as np
import sys
import warnings
import logging
import time

from . import atoms as atm
from . import pipe


logger = logging.getLogger(__name__)


class ImagePipeRunFailed(Exception):
    pass


class EMSim(object):
    def __init__(self, image_pipe: pipe.Pipe, mols: Iterable[atm.AtomList], result_handler: Callable):

        self.result_handler = result_handler
        self.image_pipe = image_pipe
        self.mols_iter = iter(mols)

        self.stop_event = Event()
        self.stop_event.clear()

        self._task_q = Queue()
        self._result_q = Queue()
        self._producer_thread = Thread(target=self.producer, args=())
        self._consumer_thread = Thread(target=self.consumer, args=())

    def run(self):
        self._producer_thread.start()
        self._consumer_thread.start()

        i = 0
        while True:
            try:
                i += 1
                result, label = self._result_q.get()
                if result is None and label == "done":
                    break
                elif isinstance(result, Exception) and label == "error":
                    # warnings.warn(f"#{i} image run failed")
                    logger.warning(f"#{i} image run failed with {type(result)}: {str(result)}")
                else:
                    if type(result) is not np.ndarray:
                        # transfer data from device to host if using cuda
                        self.result_handler(result.get(), label)
                    else:
                        self.result_handler(result, label)
            except (KeyboardInterrupt, SystemExit):
                print("exit signal received, waiting for threads to finish...")
                self.stop_event.set()
                self._producer_thread.join()
                self._consumer_thread.join()
                sys.exit("exit successfully.")
        logger.debug("all tasks done")

    def producer(self):
        for mol in self.mols_iter:
            if self.stop_event.is_set():
                return
            self._task_q.put(mol)
        self._task_q.put(None)

    def consumer(self):
        counter = 0
        while not self.stop_event.is_set():
            task = self._task_q.get()
            label = getattr(task, "label", None)

            start = time.perf_counter()
            logger.debug(f"consumer got task {task}")
            if task is None:
                self._result_q.put((None, "done"))
                return
            else:
                try:
                    result = self.image_pipe.run(task)
                except Exception as e:
                    self._result_q.put((e, "error"))
                else:
                    self._result_q.put((result, label))
            elapsed = time.perf_counter() - start
            counter += 1
            logger.info(f"emsim: {counter} images processed, label = {label}, time elapsed = {elapsed:.3f}")
