## 2026-04-27

### What I did
- Read the DeepDeWedge paper
- Made some sketches in 2D and 3D of my intuition about the missing wedge in the fourier space.
- Found a nice simulation online: Fourier Slice theorem illustration
  https://www.youtube.com/watch?v=4aDoKzqfCaM

### What I tried / what failed
- First misunderstood the axis convention in the fourier - tomography / imaging case (still unsure, GPT and Claude supported my last notation of $k_x, k_y, k_z$)

### Technical details & thoughts
- I clarified that the Fourier space has the same dimensionality as the real-space object, but it does not have the same geometric shape. A 3D cuboid object does not become a 3D cuboid in Fourier space; instead, its Fourier transform describes spatial frequency content.
- Understanding how the projection is calculated. Imagine f(x,y) as a density function. Choosing a direction f. ex. the horizontal line. Imagine integrating for every x over the whole y from $-\infty $ to $+\infty.$ The FT of this corresponds to a line in the fourier space. (A plane if we increase the dimension)
- Starting Object: 3D Volume $f(x,y,z)$. Then projection: $p_{\theta}(x,y)$. Then FT: $\hat{p}_{\theta}(k_x, k_y)$ such that we end up with: $\hat{f}(k_x, k_y, k_z)$.