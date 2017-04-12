from progress4.detext import TextDetectorNetwork
from progress4.detext import TextDetectorTrainer

net = TextDetectorNetwork()
tr = TextDetectorTrainer(net)

try:
    net.loadWeights("weights.dat")
except:
    pass

tr.trainingLoop(100000)
