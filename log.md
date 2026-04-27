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