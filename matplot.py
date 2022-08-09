import matplotlib.pyplot as plt
import numpy as np
fold = 'big_run_50\\' # blank for current dir, don't forget \\
names = [fold+'outputs_'+str(i)+'.csv' for i in range(6)]
outMats = []
xvalslist,yvalslist = np.loadtxt(fold+'vals.csv',delimiter=',')
for i, name in enumerate(names):
    outMats.append(np.loadtxt(name,delimiter=','))

fig, ax = plt.subplots()
ax.imshow(outMats[4],cmap='hot')
nticks = 11
ticks = np.linspace(0,len(xvalslist)-1,nticks)
tlabelsx = np.linspace(xvalslist[0],xvalslist[-1],nticks)
tlabelsy = np.linspace(yvalslist[0],yvalslist[-1],nticks)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(["{:.2f}".format(label) for label in tlabelsx])
ax.set_yticklabels(["{:.2f}".format(label) for label in tlabelsy])
ax.set_xlabel('e_dd')
ax.set_ylabel('omega_z/omega_r')
plt.show()