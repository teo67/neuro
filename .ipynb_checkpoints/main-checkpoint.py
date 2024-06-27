from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

class TestProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get('shape', (1,))
        self.val = Var(shape=shape, init=kwargs.pop('val', 0))
        self.s_out = OutPort(shape=shape)

class TestReceiver(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get('shape', (1,))
        self.a_in = InPort(shape=shape)


@implements(proc=TestProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyTestProcessModel(PyLoihiProcessModel):
    val: np.ndarray = LavaPyType(np.ndarray, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.s_out.send(self.val)

@implements(proc=TestReceiver, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyTestReceiverModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self):
        val = self.a_in.recv()
        print(val)

test = TestProcess(shape=(3,4,5), val=5)
receiver = TestReceiver(shape=(3,4,5))

test.s_out.connect(receiver.a_in)

run_cfg = Loihi1SimCfg()
test.run(condition=RunSteps(num_steps=10), run_cfg=run_cfg)
print(test.val.get())