import sys, time
import numpy as np, torch, schedulefree
from torch import nn
from model import NanoTabPFNModel
from train import PriorDumpDataLoader
torch.manual_seed(0); np.random.seed(0)
dev="cpu"
m=NanoTabPFNModel(96,4,192,3,2).to(dev)
opt=schedulefree.AdamWScheduleFree(m.parameters(),lr=4e-3,weight_decay=0.0)
crit=nn.CrossEntropyLoss()
prior=PriorDumpDataLoader("300k_150x5_2.h5",num_steps=700,batch_size=32,device=dev)
m.train(); opt.train(); t0=time.time()
for step,fd in enumerate(prior,1):
    sp=fd["train_test_split_index"]
    out=m((fd["x"], fd["y"][:,:sp]), train_test_split_index=sp)
    loss=crit(out.reshape(-1,out.shape[-1]), fd["y"][:,sp:].reshape(-1).long())
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); opt.zero_grad()
    if step%50==0: print(f"step {step:4d} | loss {loss.item():.4f} | split {sp} | {time.time()-t0:5.1f}s",flush=True)
print("done",flush=True)
