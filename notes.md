## 2026-04-28

### 'ddw/utils/fourier.py'

- fft_dim = (-1, -2, -3) selects the dimension starting from the last entry (so from behind basically). This means -1 corresponds to x, -2 to y and -3  to z. This is also useful since other dimensions which come before zyx are ignored and the correct three dimensions are taken for the fft.
- shift does rearange from -max, ..., 0, ..., +max. Important such that the wedge is around the axis and not to the left or right also important to use it in both, fft and ifft.
- tomo_ft * mask is a index wise multp. NOT matrix multiplication even thou both have 128x128x128 dimensions (or 3 x 128).

### missing_wedge.py

- Important dimensions are k = kz, ky, kx such that alpha is the angle between kz and the limiting plane. Else the normal_left and normal_right vectors would be wrong. Sketch in folder!
- adjusted_grid_size, enlarges the grid such that rotated grid still fit inside.


### subtomo_dataset.py

- both are rotated with the same rotation! Important because the neural network should not learn the rotation or being able to invert it. It should focus on missing fourier information and noise!
- rotvec = Rotation.random(...).as_rotvec()
- rot_axis = rotvec / rotvec.norm()
- rot_angle = torch.rad2deg(rotvec.norm())
- rotvec is a compact representation of a 3D rotation. The direction of rotvec gives the rotation-axis. And the length of rotvec is the rotation-angle in radiant. (rotvec = angle * axis) 
- generates model_input from mw_mask and subtomo0


### masked_loss.py

- calculates the loss as outside_mw_loss +  mw_weight * inside_mw_loss
- outside is for denoising, inside is for reconstruction of missing wedge
- rot_mw_mask makes sure no region is used where we have the initial wedge (this region is not helpful or accurate so should not be used for training)


### learning structure
- load subtomo0 and subtomo1
- sample a random 3D rotation and rotate both sub-tomograms using the same rotation
- generate mw_mask for the artifical missing wedge and apply it to subtomo0 only! This gives the "model_input"
- generate rot_mw_mask for the original missing wedge after applying the rotation
- return model_input, subtomo1, mw_mask, rot_mw_mask
- then the network produces "model_output"
- loss then compares the model_output with subtomo1 but depends on regions
- commonly known regions: denoising
- artifical removed regions: missing wedge reconstruction
- in originally unknown target regions: no comparison