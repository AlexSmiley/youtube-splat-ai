"""Experimental differentiable isotropic Gaussian renderer, small scenes only."""
import json, sys
from pathlib import Path
import numpy as np

def train(scene,out,steps):
    import tensorflow as tf
    from PIL import Image
    from plyfile import PlyData,PlyElement
    scene=Path(scene); out=Path(out); out.mkdir(exist_ok=True)
    meta=json.loads((scene/'transforms.json').read_text()); frames=meta['frames']; cloud=PlyData.read(scene/meta.get('ply_file_path','sparse_pc.ply'))['vertex'].data
    xyz=np.stack([cloud[k] for k in ('x','y','z')],1).astype('float32'); rng=np.random.default_rng(42); xyz=xyz[rng.choice(len(xyz),min(512,len(xyz)),replace=False)]
    pos=tf.Variable(xyz); rgb=tf.Variable(tf.zeros((len(xyz),3))); scale=tf.Variable(tf.fill([len(xyz)],-3.)); opacity=tf.Variable(tf.zeros(len(xyz))); opt=tf.keras.optimizers.Adam(.005)
    size=48; grid=tf.cast(tf.stack(tf.meshgrid(tf.range(size),tf.range(size)),axis=-1),tf.float32)
    for i in range(steps):
        f=frames[i%len(frames)]; im=np.asarray(Image.open(scene/f['file_path']).convert('RGB').resize((size,size)),dtype=np.float32)/255
        mat=np.linalg.inv(np.array(f['transform_matrix'],dtype=np.float32)); intr=lambda k: f.get(k,meta.get(k)); fx=intr('fl_x')*size/intr('w'); fy=intr('fl_y')*size/intr('h'); cx=intr('cx')*size/intr('w'); cy=intr('cy')*size/intr('h')
        with tf.GradientTape() as tape:
            cam=pos@tf.constant(mat[:3,:3].T)+mat[:3,3]; z=-cam[:,2]; safe=tf.maximum(z,.01); uv=tf.stack([fx*cam[:,0]/safe+cx,-fy*cam[:,1]/safe+cy],1); sigma=tf.maximum(tf.exp(scale)*fx/safe,.3); dist=tf.reduce_sum((grid[None]-uv[:,None,None,:])**2,axis=-1); a=tf.exp(-dist/(2*sigma[:,None,None]**2))*tf.sigmoid(opacity)[:,None,None]*tf.cast(z[:,None,None]>0,tf.float32); order=tf.argsort(z); a=tf.gather(a,order); weights=a*tf.math.cumprod(1-a,axis=0,exclusive=True); render=tf.einsum('nhw,nc->hwc',weights,tf.gather(tf.sigmoid(rgb),order)); loss=tf.reduce_mean((render-im)**2)
        variables=[pos,rgb,scale,opacity]; opt.apply_gradients(zip(tape.gradient(loss,variables),variables))
        if (i+1)%10==0 or i==0 or i+1==steps: print(f'iteration {i+1}/{steps} loss={float(loss):.6f}',flush=True)
    names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']; data=np.zeros(len(xyz),dtype=[(n,'f4') for n in names]); values=np.column_stack([pos.numpy(),np.zeros_like(xyz),(tf.sigmoid(rgb).numpy()-.5)/.28209479177,opacity.numpy(),np.repeat(scale.numpy()[:,None],3,1),np.tile([1,0,0,0],(len(xyz),1))]);
    for k,n in enumerate(names): data[n]=values[:,k]
    PlyData([PlyElement.describe(data,'vertex')]).write(out/'model.ply')
if __name__=='__main__': train(sys.argv[1],sys.argv[2],int(sys.argv[3]))
