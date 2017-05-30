import scipy.io as sio
import numpy as np

class DataWrapper(object):
    def __init__(self, filename="inp.mat", batch_size=32, liveSaveFile="live.mat", savefile="result.mat"):
        data = sio.loadmat(filename)
        self.x = data["x"]
        self.y = data["y"]
        self.filename = filename
        self.batch_size = batch_size
        self.liveSaveFile = liveSaveFile
        self.savefile = savefile
    def getBatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sample = np.random.randint(0,high=len(self.x), size=batch_size)
        return {"x":self.x[sample], "y":self.y[sample]}

