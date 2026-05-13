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


## 2026-05-04

### subtomo_dataset.py
This file defines the dataset used during model fitting. It loads corresponding pairs of sub-tomograms:

- subtomo0: used to create the model input
- subtomo1: used as the target

Both usually come from two independent reconstructions, e.g. an even/odd split, so they contain similar structure but independent noise.

During training, both sub-tomograms are randomly rotated by the same 3D rotation. Then an artificial missing wedge is applied to subtomo0. This produces the model input.

The returned item contains:

- model_input: rotated subtomo0 with an artificial missing wedge
- model_target: rotated subtomo1
- mw_mask: artificial missing wedge mask
- rot_mw_mask: original missing wedge mask after rotation

The purpose is to create self-supervised input-target pairs for denoising and missing-wedge reconstruction.



### masked_loss.py
This file defines the training loss. The model output is compared to the target in Fourier space, but only in regions where the target contains valid information.

Two regions are used:

outside_mw_mask = rot_mw_mask * mw_mask
inside_mw_mask = rot_mw_mask * (1 - mw_mask)

loss = outside_mw_loss + mw_weight * inside_mw_loss

Here important to note, since the rot_mw_mask accounts for the original missing wedge, we do not want to use this information. Since our subtomos are both rotated in the same direction, this means that the original mw is accounted for in both subtomos with just rot_mw_mask.



### unet.py
Contains 3D U-Net model and the PyTorch Lightning training wrapper.
LitUnet3D handles:
- training and validation steps
- optimizer setup
- logging
- normalization updates
- iterative updating of missing wedges in the stored sub-tomograms

Where as Unet3D defines the actual architecture:
Input: 1 × 64³

Encoder:
32 × 64³
64 × 32³
128 × 16³

Bottleneck:
128 → 256 → 128 at 8³

Decoder:
128 × 16³
64 × 32³
32 × 64³

Final:
1 × 64³

The encoder reduces spatial resolution and increases feature channels. The bottleneck contains the most compressed representation. The decoder upsamples back to the original size.

Skip connections concatenate encoder features with decoder features at the same spatial resolution. This helps preserve spatial detail.

Each DownConvBlock consists of: 3 × (Conv3D → InstanceNorm3D → Dropout3D → LeakyReLU)
The final output is added to the input through a residual connection, so the network learns a correction rather than reconstructing everything from scratch.




## 2026-05-04


### fit_model.py

fit_model.py defines the main training command ddw fit-model. It loads parameters from config.yaml, sets up the subtomogram directory and log directory, creates the training and validation datasets, initializes the 3D U-Net Lightning model, configures the PyTorch Lightning trainer, and starts model fitting.

The training dataset is created using SubtomoDataset with random rotations. Validation data, if available, uses deterministic rotations so that validation loss is comparable across epochs.

The script saves checkpoints in three ways:

1. Every N epochs
   Checkpoints are saved to checkpoints/epoch/ every save_model_every_n_epochs.

2. Best fitting loss
   The best save_n_models_with_lowest_fitting_loss checkpoints are saved according to lowest training loss.

3. Best validation loss
   If validation data exists, the best save_n_models_with_lowest_val_loss checkpoints are saved according to lowest validation loss.

The parameter update_subtomo_missing_wedges_every_n_epochs controls how often the stored input subtomograms are updated during training. Every N epochs, the current model predicts the missing-wedge content, and this predicted Fourier content is inserted into the missing regions of the input subtomograms. The known Fourier regions are kept from the original/current subtomograms.

This iterative refinement is needed because the training subtomograms are themselves incomplete due to the original missing wedge. By gradually filling missing regions with the current model prediction, the model inputs become closer to missing-wedge-corrected volumes, which improves subsequent training.

The original missing wedge in the target is masked out in the loss and is therefore not evaluated directly. However, the original missing wedge in the input is iteratively filled with model predictions, so that future training inputs become less incomplete and the model can learn more effectively.

Therefore, this is not direct supervised learning on the original missing wedge region. Instead, it is an iterative improvement of the model inputs.


### subtomos.py

subtomos.py contains utility functions for extracting small cubic sub-tomograms from a larger tomogram and reassembling them after model prediction.

- Sliding window extraction
extract_subtomos uses a 3D sliding window to crop cubic sub-tomograms from the full tomogram. The cube size is given by subtomo_size, and the step size is given by subtomo_extraction_strides.
If the stride equals the subtomogram size, the extracted cubes do not overlap. If the stride is smaller than the subtomogram size, neighboring sub-tomograms overlap.
The function also returns subtomo_start_coords, which store the original position of each sub-tomogram. These coordinates are needed later for reassembly.
If enlarge_subtomos_for_rotating=True, the extracted sub-tomograms are enlarged by approximately sqrt(2) so they can be randomly rotated and cropped back without zero-filled corners.
If pad_before_subtomo_extraction=True, the tomogram is padded before extraction so that the sliding window covers the full volume.

- Reassembly
reassemble_subtomos places the processed sub-tomograms back into their original positions using the stored start coordinates.
If sub-tomograms overlap, their values are combined by a weighted average. The function accumulates both the weighted predictions and the corresponding weights, then divides by the total weight.

- Linear ramp weighting
When subtomo_overlap is provided, get_linear_ramp_weights creates a 3D weight map. The center of each sub-tomogram receives high weight, while the edges receive lower weights using a linear ramp.
This avoids visible seams during reassembly. The model predictions near the edges of a sub-tomogram can be less reliable because the network has less surrounding context there. Linear ramp weighting therefore trusts the center of each sub-tomogram more than its boundaries.
This is better than simple averaging because it blends overlapping sub-tomograms smoothly instead of giving boundary regions the same weight as central regions.

Q1: How does the sliding window extraction work?
A1: A 3D sliding window moves a cubic window through the full tomogram. At each start coordinate, it extracts a cube of size subtomo_size x subtomo_size x subtomo_size. The step size is controlled by subtomo_extraction_strides. F. ex. subtomo_size = 64 and stride = 64, this would mean no overlap always cubes next to each other.

Q2: What is the linear ramp weighting during reassembly? Why not just average?
A2: Linear ramp weighting means that each subtomogram is not inserted with equal weight everywhere. Instead: center of subtomo → high weight and edges of subtomo  → low weight. The weight decreases linearly near the boundaries of the subtomogram. So during reassembly, overlapping subtomograms are combined by a weighted average: final value = weighted sum / sum of weights.
And simple averaging would give every part of every subtomogram equal importance, this is not reasonable because of multiple reasons: 1. U-Net has less spatial context near the edges. 2. Padding / boundary effects can create artifacts. 3. Neighboring subtomograms may predict slightly different values at their borders. 4. Hard averaging can create visible seams between tiles. 
Linear ramp weighting reduces these seams. It basically says: trust the center more, dont trust the edges, blend overlapping subtomograms smoothly.



### normalization.py

normalization.py computes the mean and standard deviation used for the internal input normalization of the U-Net. The U-Net normalizes each input volume as: (volume - normalization_loc) / (normalization_scale + 1e-6)
- get_avg_model_input_mean_and_std takes a full tomogram, temporarily extracts sub-tomograms from it, wraps them in SubtomoDataset, and computes statistics from model_input-type data. This means the statistics are not computed directly from the raw tomogram, but from inputs that are processed similarly to training inputs: cropped sub-tomograms, random rotations, and artificial missing wedge masking. get_avg_model_input_mean_and_std_from_dataloader iterates over batches and computes: the mean of each model_input sub-tomogram AND the variance of each model_input sub-tomogram
Then it returns: mean = average of all sub-tomogram means AND std  = sqrt(average of all sub-tomogram variances)
Q1: when and why do we recompute normalization at inference time?
A1: Normalization is recomputed at the start of training and again whenever the stored input sub-tomograms are updated during iterative missing-wedge refinement, because their intensity statistics can change. It is also recomputed at inference time, meaning after training, when the trained model is applied to refine a full tomogram. This is necessary because the tomogram being refined may have different intensity statistics from the training data. The model weights stay fixed, but the normalization parameters are updated so that the inference inputs are on the correct scale for the trained U-Net.


## Full Data Flow: Raw Tomogram → Subtomograms → Model → Refined Tomogram

The DeepDeWedge pipeline starts from one or more raw reconstructed tomograms. These tomograms are affected by two main problems: noise from the low electron dose and missing-wedge artifacts caused by the limited tilt range during cryo-ET acquisition. In Fourier space, this means that a wedge-shaped region of information is missing, which leads to anisotropic resolution and blurring, especially in the Z direction.

The first step is data preparation. The tilt series or reconstructed tomograms are split into two independent parts, for example using an even/odd split. From these two parts, two corresponding tomograms are reconstructed: tomo0 and tomo1. These contain the same underlying biological structure but different, approximately independent noise. The tomograms are then divided into many smaller cubic sub-tomograms using a 3D sliding window. These sub-tomograms are easier to process with the U-Net than the full tomogram.

During training, corresponding pairs of sub-tomograms are loaded: subtomo0 is used to generate the model input, and subtomo1 is used as the target. Both are randomly rotated by the same 3D rotation. Then an artificial missing wedge is applied to the rotated subtomo0, producing the actual model_input. The rotated subtomo1 becomes the model_target. This creates a self-supervised learning task: the network sees a sub-tomogram with an additional artificially removed Fourier region and learns to reconstruct it by comparing against the corresponding target.

The model itself is a 3D U-Net. It takes the input sub-tomogram, normalizes it, processes it through an encoder-decoder architecture with skip connections, and outputs a corrected sub-tomogram. The loss is computed in Fourier space using masks. Regions where the target has no valid information due to the original missing wedge are ignored. Regions that were artificially removed from the input but are available in the target are used to train missing-wedge reconstruction. Regions known in both input and target contribute mainly to denoising.

During model fitting, DeepDeWedge also performs iterative refinement of the input sub-tomograms. Every few epochs, the current model is applied to the stored input sub-tomograms, and the predicted Fourier content is inserted into their missing-wedge regions while keeping the originally known Fourier information unchanged. This gradually improves the model inputs and makes them closer to missing-wedge-corrected volumes.

After training, the fitted U-Net is used for inference, meaning it is applied to refine a full tomogram. The full tomogram is again split into overlapping sub-tomograms. Each sub-tomogram is passed through the trained model to obtain a denoised and missing-wedge-corrected prediction. Finally, the predicted sub-tomograms are reassembled into a full refined tomogram. In overlapping regions, linear ramp weighting is used so that the centers of sub-tomograms are trusted more than their boundaries. This avoids visible seams and produces a smooth final reconstruction.

The final output is a refined tomogram with reduced noise and partially filled missing-wedge information, leading to more isotropic resolution and better interpretability of structures in 3D.



## 2026-05-12

## FSC function
FSC measures how similar two 3D volumes in fourier space are, seperated by frequencies. So are they similar at low freq? similar at mid freq? and so on. Depending on the frequencies it corresponds to different things regarding the volumes. High frequencies -> more fine details. Low frequencies -> big, broader structures. When real structure is existing, the volumes should be similar. Noise on the other hand should be independent and thus no strong correlation.

One uses fourier shells, these are not infinitesimal shells, they actually have some depth. (f. ex. radius 0 to 1). Calculate the distance r for each point in fourier space. Same r corresponds to same spatial frequencies. Now in these shells we have: F1_shell = Fourier coeff. of vol1 in this shell and F2_shell = Four. coeff. of vol2 in this shell. Now calc. the correlation term: FSC = sum(F1 * conj(F2)) / sqrt(sum(|F1|^2) * sum(|F2|^2)) where the conj() is the complex conjugate (important since Fourier coeff. are complex numbers) We later still take the real part of that because we can still end up with complex numbers bc computation wise. This is normalized bc. of the denominator which leads to values from -1 to 1. (anti correlated to correlated) Which is known. Usually one looks at threshold numbers to make conclusions.


## Experiment 1 Week 2
- Interpretation of the FSC plots and the values
- The FSC against the FBP all-frames reference remains high for all inference-time angle assumptions. The smaller angle assumptions show slightly higher correlation with the FBP reference, likely because they modify the FBP reconstruction less strongly. Therefore, this FSC should be interpreted as similarity to the FBP reference rather than absolute reconstruction quality.