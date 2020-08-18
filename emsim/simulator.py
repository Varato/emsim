from typing import Iterable, Callable
from threading import Thread
from queue import Queue
import numpy as np
import warnings
import logging

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

        self._task_q = Queue()
        self._result_q = Queue()
        self._producer_thread = Thread(target=self.producer, args=())
        self._consumer_thread = Thread(target=self.consumer, args=(), daemon=True)

    def run(self):
        self._producer_thread.start()
        self._consumer_thread.start()

        i = 0
        while True:
            i += 1
            result, label = self._result_q.get()
            if result is None and label == "done":
                break
            elif result is Exception and label == "error":
                # warnings.warn(f"#{i} image run failed")
                logger.warning(f"#{i} image run failed")
            else:
                if type(result) is not np.ndarray:
                    # transfer data from device to host if using cuda
                    self.result_handler(result.get(), label)
                else:
                    self.result_handler(result, label)
        logger.debug("all tasks done")

    def producer(self):
        for mol in self.mols_iter:
            self._task_q.put(mol)

        self._task_q.put(None)

    def consumer(self):
        while True:
            task = self._task_q.get()
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
                    label = getattr(task, "label", None)
                    self._result_q.put((result, label))
