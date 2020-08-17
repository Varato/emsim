from typing import Iterable, Callable
from threading import Thread
from queue import Queue
import warnings
import numpy as np

from . import atoms as atm
from . import pipe


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
        self._consumer_thread = Thread(target=self.consumer, args=())

    def run(self):
        self._producer_thread.start()
        self._consumer_thread.start()

        i = 0
        while True:
            i += 1
            result = self._result_q.get()
            if result is None:
                break
            elif result is Exception:
                warnings.warn("#{i} image run failed")
            else:
                if type(result) is not np.ndarray:
                    # transfer data from device to host if using cuda
                    self.result_handler(result.get())
                else:
                    self.result_handler(result)
        print("all tasks done")

    def producer(self):
        for mol in self.mols_iter:
            self._task_q.put(mol)

        self._task_q.put(None)

    def consumer(self):
        while True:
            task = self._task_q.get()
            print(f"consumer got task {task}")
            if task is None:
                self._result_q.put(None)
                return
            else:
                try:
                    result = self.image_pipe.run(task)
                except Exception as e:
                    self._result_q.put(e)
                else:
                    self._result_q.put(result)
