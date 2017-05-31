import scipy.io as sio
import numpy as np

class DataWrapper(object):
    def __init__(self, filename="inp.mat", batch_size=32, liveSaveFile="live.mat", savefile="result.mat"):
        data = sio.loadmat(filename)
        self.x = data["x"]
        self.y = data["y"]
        tenpercent = int(len(self.x)/10)
        np.random.seed(217132)
        sample = np.random.randint(0,high=len(self.x), size=tenpercent)
        self.vx = self.x[sample]
        self.vy = self.y[sample]
        self.x = np.delete(self.x,sample,0)
        self.y = np.delete(self.y,sample,0)


        np.random.seed(777777)
        sample = np.random.randint(0,high=len(self.x), size=tenpercent)
        self.tx = self.x[sample]
        self.ty = self.y[sample]
        self.x = np.delete(self.x,sample,0)
        self.y = np.delete(self.y,sample,0)

        np.random.seed()

        self.filename = filename
        self.batch_size = batch_size
        self.liveSaveFile = liveSaveFile
        self.savefile = savefile

    def getTestData(self):
        return {"x":self.tx, "y":self.ty}


    def getValidationData(self):
        return {"x":self.vx, "y":self.vy}

    def getBatch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sample = np.random.randint(0,high=len(self.x), size=batch_size)
        return {"x":self.x[sample], "y":self.y[sample]}

