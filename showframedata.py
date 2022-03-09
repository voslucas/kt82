import pickle
import matplotlib.pyplot as plt

# Code to generate a histogram image, displaying the relative amount of traffic blocks in each frame.

frames = pickle.loads(open("framedata.pickle", "rb").read())

reds = [f['red']/72 for f in frames]

#plt.grid()
plt.hist(reds, bins=20, rwidth=0.8)
plt.xlim([0.0, 1.0])
plt.xlabel('Relative number of Traffic blocks')
plt.ylabel('Number of frames')
plt.show()


