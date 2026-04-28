## 2026-04-27

### What I did
- Read the DeepDeWedge paper
- Made some sketches in 2D and 3D of my intuition about the missing wedge in the fourier space.
- Found a nice simulation online: Fourier Slice theorem illustration
  https://www.youtube.com/watch?v=4aDoKzqfCaM
- setup the conda environment on the jupyterhub (euler cluster)
- follow step0 to get my first tomograms. Noticed the blurriness in the XZ and YZ slices. 
- setup the github repo and created a clear directory structure

### What I tried / what failed
- First misunderstood the axis convention in the Fourier - tomography / imaging setup (still unsure, GPT and Claude supported my last notation of $k_x, k_y, k_z$)
- problems with using the euler cluster through the terminal and sending jobs

### Technical details & thoughts
- I clarified that Fourier space has the same dimensionality as the real-space object, but it does not have the same geometric shape. A 3D cuboid object does not become a 3D cuboid in Fourier space; instead, its Fourier transform describes spatial frequency content.
- I clarified how a projection is calculated. For example, if $f(x,y)$ is a 2D density function, then a projection along the $y$-direction is obtained by integrating over $y$ for each $x$:
  $$
  p(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy.
  $$
  The 1D Fourier transform of this projection corresponds to a line through the 2D Fourier transform of the object. In 3D, the analogous statement is that the 2D Fourier transform of a projection corresponds to a plane through the 3D Fourier transform.
- In cryo-ET, the starting object is a 3D volume $f(x,y,z)$. A projection at tilt angle $\theta$ gives a 2D image $p_\theta(x,y)$. Its 2D Fourier transform $\hat p_\theta(k_x,k_y)$ corresponds, by the Fourier Slice Theorem, to a central 2D slice through the 3D Fourier transform $\hat f(k_x,k_y,k_z)$.
- Because the tilt range is limited, e.g. to approximately $\pm 60^\circ$ instead of $\pm 90^\circ$, some Fourier planes are never measured. This leaves wedge-shaped missing regions in 3D Fourier space, which leads to anisotropic resolution and blurring/elongation especially in directions involving $z$.


## 2026-04-28

### What I did
- Setup the Step1 jupyter notebook from the tutorial
- Analyzing "prepare_data" parameters and "shared" parameter to understand what I can adjust and what is given as input
- Plotted 5 sub-tomogram pairs (even frame vs odd frame) side by side
- read fourier.py, understanding of the fft and iftt, dimensions and mask multiplication

### What I tried / what failed
- !conda run -n ddw_env ddw prepare-data → conda: command not found
- !ddw prepare-data → ddw: command not found
- Fixed: Euler's Jupyterhub shell has a minimal PATH, does not include the conda env.
- Had to workaround this by looking for the correct directory via sys.executable. This is in the same dir as the needed ddw. get the directory name and join the ddw path in order to use 'prepare-data'.
- Tried editing kernel.json, failed miserably and lost it, had to register again
- Struggled with the understanding of the mask in missing_wedge.py¨

### Technical details & thoughts
- Apparently my conda environment is setup correctly after checking with AI.
- Understanding the data input, the tilt series t is already split up and filtered back-projection (FBP) is applied. We simply load the two different tomogram data. Dimension analysis and understanding how to get different slices from the v0 and v1 volumes.
- 144 sub-tomograms saved for fitting, 36 for validation (80/20 split)
- after sketching the plane normal, realized k=(kx, ky, kz) cannot be true. Analyzed the dimensions of the tomogram. Found out:
tomo[half[0], :, :]  = X-Y slice  → axis 0 is z
tomo[:, half[1], :]  = X-Z slice  → axis 1 is y
tomo[:, :, half[2]]  = Y-Z slice  → axis 2 is x.
Thus k = (kz, ky, kx)

# TODO **End-of-day checkpoint:** 5-panel figure showing how the wedge shape changes with angle.



## 2026-04-28