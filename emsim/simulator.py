import queue
from threading import Thread
from itertools import zip_longest

from . import em
from . import atoms as atm
from . import pipe


class EMSim(object):
    def __init__(self, mol_gen, pipe_gen, result_handler):
        self._q = queue.Queue()
        self.result_handler = result_handler
        self._worker_thread = Thread(target=self.consumer, args=(self._q))

    def run(self):
        self._worker_thread.start()
        for mol, p in zip_longest(self.mol_gen(), self.pipe_gen()):
            self._q.put((mol, pipe))

    @staticmethod
    def consumer(q):
        while True:
            mol, p = a.get()
            result = p.run(mol)
            self.result_handler(result)
