
import numpy as np

def distVec(A, B):
    C = np.power(A - B, 2) / 2
    return np.sum(C, axis = - 1)

def factsElems(facts):
     return facts[..., :-1], facts[..., -1]

def smElems(sm):
    return sm[..., :-1], sm[..., -1:]

def smLinPredict(sm, features):
  smW, smB = smElems(sm)
  return np.matmul( smW, np.transpose(features) ) + smB

def sigmoid(x, a):
    return 1 / (1 + np.exp(-a * x))
 
def loss(sm, facts):
    features, omLabels = factsElems(facts)
    smLabels = smLinPredict(sm, features)
    return distVec(smLabels, omLabels)


fact1 = np.array([1,5]); facts = np.array([ fact1 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact2 = np.array([3,25]); facts = np.array([ fact2 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

facts = np.array([ fact1, fact2 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
facts = np.array([ fact3, fact4 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
facts = np.array([fact1, fact2, fact3, fact4 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

facts = np.array([ fact1, fact2, fact3, fact4])
sm11 = np.array([-10, 25]); sm12 = np.array([-9, 20]); sm21 = np.array([-12, 20]); sm22 = np.array([-8, 28])
sm = np.array([ [sm11, sm12], [sm21, sm22] ])
print("loss = {}".format(loss(sm, facts)))


w = np.arange(3, 13, 1)
b = np.arange(-8, 18, 1)
W, B = np.meshgrid(w, b)
smW = W.reshape(tuple(np.append(W.shape,1)))
smB = B.reshape(tuple(np.append(B.shape,1)))
sm = np.append(smW, smB, axis = -1)

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

fact1 = np.array([1, 5]); fact2 = np.array([3, 25])
Z = loss(sm, np.array([fact1, fact2]))
fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
Z1 = loss(sm, np.array([fact3, fact4]))
Z2 = loss(sm, np.array([fact1, fact2, fact3, fact4]))

thick = 5
width = (W.max()-W.min())/100*thick
depth = (B.max()-B.min())/100*thick
top = ((Z+Z1).max()-(Z+Z1).min())/100*thick


fig = plt.figure(dpi = 120)
ax = fig.gca(projection = '3d')

title = 'Loss by (w, b) for (fact1, fact2)'; zmax = Z2.max()
ax.set_title(title, fontsize=12, fontweight='normal', color='b')
ax.set_xlabel('w', fontsize=12, fontweight='normal', color='b')
ax.set_ylabel('b', fontsize=12, fontweight='normal', color='b')
#ax.set_zlabel('loss', fontsize=14, fontweight='normal', color='b')
ax.set_zlim(0, zmax)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))

norm = plt.Normalize(Z.min(), Z.max())
colors = cm.viridis(norm(Z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(W, B, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
ax.bar3d([10], [-5], [0], width, depth, top, color = 'magenta', shade = True)
ax.text(10+1, -5+1, 0, '0', color='magenta')

plt.show()