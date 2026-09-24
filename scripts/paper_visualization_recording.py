"""Opt-in, observational capture of the actual policy inference and execution.

No RNG calls are added. Fused attention still computes policy outputs; materialized
probabilities are diagnostics only. The scoped SDPA wrapper is for single-threaded
model evaluation and is always restored, including on exceptions.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

def probabilities(q,k,mask=None,scale=None,is_causal=False):
    logits=q.float()@k.float().transpose(-1,-2)
    logits*=float(scale) if scale is not None else q.shape[-1]**-0.5
    if is_causal:
        visible=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        logits=logits.masked_fill(~visible,float('-inf'))
    if mask is not None:
        logits=logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask.float()
    return logits.softmax(-1)

@contextmanager
def capture_predictor_attention(storage):
    original=F.scaled_dot_product_attention
    def wrapped(q,k,v,*args,**kwargs):
        output=original(q,k,v,*args,**kwargs)
        if q.shape[-2]<=2 and k.shape[-2]<128:
            mask=kwargs.get('attn_mask',args[0] if args else None)
            p=probabilities(q,k,mask,kwargs.get('scale'),kwargs.get('is_causal',False))
            storage.append(p.mean(1).detach().cpu().numpy())
        return output
    F.scaled_dot_product_attention=wrapped
    try:yield
    finally:F.scaled_dot_product_attention=original

class PaperRecorder:
    def __init__(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.file=h5py.File(path,'w');self.file.attrs['metadata']=json.dumps(metadata)
        self.frames=self.file.create_group('observations');self.plans=self.file.create_group('plans')
        self.group=None;self.predictor=[];self.attention={};self.denoising_steps=[]
    def observe(self,step,frame,proprio,action=None,geometry=None):
        name=f'{step:05d}'
        if name in self.frames:return
        g=self.frames.create_group(name);g.create_dataset('rgb',data=frame,compression='lzf')
        g.create_dataset('proprio',data=proprio)
        if action is not None:g.create_dataset('action_executed',data=action)
        if geometry is not None:g.attrs['geometry']=json.dumps(geometry)
    def start(self,meta):
        self.group=self.plans.create_group(f"{int(meta['step']):05d}")
        self.group.attrs['metadata']=json.dumps(meta)
        self.predictor=[];self.attention={};self.denoising_steps=[]
    def condition(self,observed,times,future_times,condition):
        for name,value in [('observed_rgb',observed),('observed_times',times),('future_times',future_times),
                           ('tokens',condition.future_tokens),('current',condition.current_vfm_features),('selected_branch',condition.selected_branch)]:
            arr=value.detach().cpu().numpy() if value.dtype in (torch.uint8,torch.int64) else value.detach().float().cpu().numpy()
            dataset=self.group.create_dataset(name,data=arr,compression='lzf')
            dataset.attrs['torch_dtype']=str(value.dtype)
        c=condition.kv_cache
        self.group.attrs['anchor_tokens']=c.num_anchor_tokens;self.group.attrs['future_tokens']=c.num_future_tokens
        self.group.create_dataset('key_mask',data=c.key_mask.detach().cpu().numpy())
        for i,value in enumerate(self.predictor):self.group.create_dataset(f'world_attention/{i:03d}',data=value)
    def velocity(self,network,actions,timestep,condition):
        velocity,attention=network.action_model(actions,timestep,condition.kv_cache,return_cross_attention=True)
        for layer,weight in attention.items():
            mean=weight.detach().float().mean(1)[0]
            self.attention[layer]=self.attention.get(layer,0)+mean
        self.denoising_steps.append(float(timestep[0]))
        return velocity
    def finish_plan(self,actions):
        for layer,total in self.attention.items():
            self.group.create_dataset(f'action_attention/{layer:02d}',data=(total/len(self.denoising_steps)).cpu().numpy(),compression='lzf')
        self.group.create_dataset('denoising_timesteps',data=self.denoising_steps)
        self.group.create_dataset('action_chunk',data=actions)
        self.group.attrs['complete']=True;self.attention={};self.file.flush()
    def close(self,success):
        self.file.attrs['success']=bool(success);self.file.attrs['complete']=True;self.file.close()
