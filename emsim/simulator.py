import queue

from . import em
from . import atoms as atm
from . import pipe


class EMSim(object):
    def __init__(self):
        self._q = queue.Queue()

    def run(self):
        for mol in self.mol_gen():
            self._q.put(mol)

    def mol_gen(self):
        yield None


