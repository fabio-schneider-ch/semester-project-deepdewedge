## 2026-04-28

### 'ddw/utils/fourier.py'

- fft_dim = (-1, -2, -3) selects the dimension starting from the last entry (so from behind basically). This means -1 corresponds to x, -2 to y and -3  to z. This is also useful since other dimensions which come before zyx are ignored and the correct three dimensions are taken for the fft.
- shift does rearange from -max, ..., 0, ..., +max. Important such that the wedge is around the axis and not to the left or right also important to use it in both, fft and ifft.
- tomo_ft * mask is a index wise multp. NOT matrix multiplication even thou both have 128x128x128 dimensions (or 3 x 128).

### missing_wedge.py

- Important dimensions are k = kz, ky, kx such that alpha is the angle between kz and the limiting plane. Else the normal_left and normal_right vectors would be wrong. Sketch in folder!
- adjusted_grid_size, enlarges the grid such that rotated grid still fit inside.

