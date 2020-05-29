import pickle
import numpy as np
import random

if __name__ == "__main__":
    demo_number = '02'
    savename = ["demonstrations/demo_00_" + demo_number + ".pkl",   "demonstrations/demo_01_" + demo_number + ".pkl"]
    data  = pickle.load(open(savename[0], "rb")) + pickle.load(open(savename[1], "rb"))
    random.shuffle(data)
    pickle.dump(data,open('datasets/traj.pkl',"wb"))
    print(len(data))
    data = np.array(data)
    print(data.shape)