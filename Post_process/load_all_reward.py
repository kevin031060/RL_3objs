import torch
from tasks import motsp
from tasks.motsp import TSPDataset, reward
from torch.utils.data import DataLoader
from model import DRL4TSP
from trainer_motsp import StateCritic
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from Post_process.dis_matrix import dis_matrix
import time
from mpl_toolkits.mplot3d import Axes3D


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "../tsp_transfer_100run_500000_5epoch_20city/20"效果一般。应该再训练一遍
save_dir = "../tsp_3objs/40"
# save_dir = "../tsp_4static/20"
# param
update_fn = None
STATIC_SIZE = 4  # (x, y)
DYNAMIC_SIZE = 1  # dummy for compatibility

# claim model
actor = DRL4TSP(STATIC_SIZE,
                DYNAMIC_SIZE,
                128,
                update_fn,
                motsp.update_mask,
                1,
                0.1).to(device)
critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, 128).to(device)

# data 143
from Post_process.convet_kro_dataloader import Kro_dataset
kro = 0
D = 200
if kro:
    D = 200
    Test_data = Kro_dataset(D)
    Test_loader = DataLoader(Test_data, 1, False, num_workers=0)
else:
    # 40city_train: city20 13 city40 143 city70 2523
    #
    Test_data = TSPDataset(D, 1, 2523)
    Test_loader = DataLoader(Test_data, 1, False, num_workers=0)

iter_data = iter(Test_loader)
static, dynamic, x0 = iter_data.next()
static = static.to(device)
dynamic = dynamic.to(device)
x0 = x0.to(device) if len(x0) > 0 else None

# load 50 models
N=90
tmp = scio.loadmat('../point.mat')
w2_list = tmp['M']
w2_list = np.flip(w2_list, 0)
objs = np.zeros((N+1,3))
start  = time.time()
t1_all = 0
t2_all = 0
for i in range(0, N+1):
    t1 = time.time()
    ac = os.path.join(save_dir, "w_%2.2f_%2.2f_%2.2f" % (w2_list[i,0],w2_list[i,1],w2_list[i,2]),"actor.pt")
    cri = os.path.join(save_dir, "w_%2.2f_%2.2f_%2.2f" % (w2_list[i,0],w2_list[i,1],w2_list[i,2]),"critic.pt")
    actor.load_state_dict(torch.load(ac, device))
    critic.load_state_dict(torch.load(cri, device))
    t1_all = t1_all + time.time()-t1
    # calculate

    with torch.no_grad():
        # t2 = time.time()
        tour_indices, _ = actor.forward(static, dynamic, x0)
        # t2_all = t2_all + time.time() - t2
    obj, obj1, obj2, obj3 = reward(static, tour_indices, w2_list[i,0], w2_list[i,1], w2_list[i,2])

    objs[i,:] = [obj1, obj2, obj3]

print("time_load_model:%2.4f"%t1_all)
print("time_predict_model:%2.4f"%t2_all)
print(time.time()-start)

print(objs)
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(objs[:,0],objs[:,1], objs[:,2], cmap='Blues')  #绘制散点图

plt.show()

obj1_matrix, obj2_matrix, obj3_matrix = dis_matrix(static, STATIC_SIZE, True)
scio.savemat("data/obj1_%d_%d.mat"%(5, D), {'obj1':obj1_matrix})
scio.savemat("data/obj2_%d_%d.mat"%(5, D), {'obj2':obj2_matrix})
scio.savemat("data/obj3_%d_%d.mat"%(5, D), {'obj3':obj3_matrix})
scio.savemat("data/rl%d_%d.mat"%(5, D),{'rl':objs})

# from load_test_plot import show
# show_if = 1
# if show_if:
#     i = 0
#     ac = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"actor.pt")
#     cri = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"critic.pt")
#     actor.load_state_dict(torch.load(ac, device))
#     critic.load_state_dict(torch.load(cri, device))
#
#     show(Test_loader, actor)

