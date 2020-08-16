from typing import Iterable, Callable
import queue
from threading import Thread
import warnings

from . import atoms as atm
from . import pipe


class ImagePipeRunFailed(Exception):
    pass


class EMSim(object):
    def __init__(self, image_pipe: pipe.Pipe, mols: Iterable[atm.AtomList], result_handler: Callable):
        self._q = queue.Queue()
        self.result_handler = result_handler
        self.image_pipe = image_pipe
        self.mols_iter = iter(mols)
        self._worker_thread = Thread(target=self.consumer, args=(self._q, ), daemon=True)

    def run(self):
        self._worker_thread.start()
        for mol in self.mols_iter:
            self._q.put(mol)
        self._q.join()
        print("all tasks done")

    def consumer(self, q):
        while True:
            mol = q.get()
            print(f"consumer got task: {mol}")
            try:
                result = self.image_pipe.run(mol)
            except Exception as e:
                warnings.warn(f"one image generation failed: {e}")
                raise ImagePipeRunFailed
            self.result_handler(result)
            q.task_done()
