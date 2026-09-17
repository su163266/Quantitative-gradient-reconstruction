## The dataset,training code can be obtained at here. The detail of DCT follows bellow.
# DCT-based phase integration

This document specifies the discrete phase-integration algorithm used in Quantitative-gradient-reconstruction: its gradient convention, boundary pixels, DC component, grid spacing, and a runnable reference implementation.

The description is pinned to [commit 6855fe81006809c625ffb76c49ac46d870813796](https://github.com/su163266/Quantitative-gradient-reconstruction/tree/6855fe81006809c625ffb76c49ac46d870813796). It describes the active DCTPoisson class, rather than the commented-out alternative solvers.

## 1. Inputs and grid spacing

The predicted gradients gx and gy have shape (B, 1, H, W). The x direction is the column/width direction; y is the row/height direction. All indices below are zero-based.

The data-generation code constructs adjacent-pixel forward differences:

$$
g_x(i,j)=\phi(i,j+1)-\phi(i,j),\quad 0\le j<W-1,
$$

$$
g_y(i,j)=\phi(i+1,j)-\phi(i,j),\quad 0\le i<H-1.
$$

The remaining entries are gx(i,W-1)=0 and gy(H-1,j)=0. The labels therefore represent phase increments on a unit pixel grid, hx=hy=1. For phase in radians, these are radians per adjacent-pixel step. No physical pixel-pitch conversion is applied by the solver.

Although the data script declares pixel_shear=4, its active label calculation uses adjacent-pixel differences without multiplying or dividing by that value. The integration spacing is 1, not 4. The solver integrates predicted gradients directly, without wrapping them modulo 2*pi.

Source: [gradient-label construction](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/dataset/Main%20%281%29.m#L92-L104).

## 2. Discrete equation and boundary pixels

For possibly non-integrable predictions, the reconstructed phase minimizes

$$
\frac12\sum_{i=0}^{H-1}\sum_{j=0}^{W-2}
[\phi(i,j+1)-\phi(i,j)-g_x(i,j)]^2
+
\frac12\sum_{i=0}^{H-2}\sum_{j=0}^{W-1}
[\phi(i+1,j)-\phi(i,j)-g_y(i,j)]^2.
$$

Let Dx and Dy denote forward differences on valid interior edges. Define

$$
L=-(D_x^{\mathsf T}D_x+D_y^{\mathsf T}D_y),\qquad
b=-(D_x^{\mathsf T}g_x+D_y^{\mathsf T}g_y).
$$

The normal equation is L phi = b. Here L is negative semidefinite. The code calls b "div": under this gradient convention, it is the **negative adjoint** of the forward gradient.

Explicitly,

$$
b_{i,j}=b^x_{i,j}+b^y_{i,j},
$$

$$
b^x_{i,j}=
\begin{cases}
g_x(i,0), & j=0,\\
g_x(i,j)-g_x(i,j-1), & 1\le j\le W-2,\\
-g_x(i,W-2), & j=W-1,
\end{cases}
$$

$$
b^y_{i,j}=
\begin{cases}
g_y(0,j), & i=0,\\
g_y(i,j)-g_y(i-1,j), & 1\le i\le H-2,\\
-g_y(H-2,j), & i=H-1.
\end{cases}
$$

Interior pixels use the five-point Laplacian:

$$
(L\phi)_{i,j}=\phi_{i,j-1}+\phi_{i,j+1}
+\phi_{i-1,j}+\phi_{i+1,j}-4\phi_{i,j}.
$$

At a boundary, only available neighbors contribute. The horizontal term is phi(i,1)-phi(i,0) at the left edge and phi(i,W-2)-phi(i,W-1) at the right edge. A non-corner boundary pixel has three neighbors; a corner has two. For example,

$$
(L\phi)_{0,0}=\phi_{0,1}+\phi_{1,0}-2\phi_{0,0},\qquad
b_{0,0}=g_x(0,0)+g_y(0,0).
$$

This is a Neumann-type discrete Laplacian with no periodic connection across opposite edges. Its stencil can equivalently be written using replicated ghost values outside the rectangle. Boundary gradient contributions enter through b as above; phase values are not fixed to zero at the boundary.

All H by W phase pixels are retained. The last column of gx and last row of gy are unused placeholders, not additional measured edges. The test scripts explicitly zero them before integration; the solver itself ignores them. Changing only those entries cannot change the reconstructed phase.

DCTPoisson solves on the full rectangular grid and accepts no pupil mask or internal aperture boundary.

Sources: [divergence assembly](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/Net%20%281%29.py#L424-L433), [test-time boundary handling](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/test_pre.py#L136-L144).

## 3. DCT solution and DC component

Define the orthonormal DCT-II matrix for a dimension of length N:

$$
C_N(k,n)=\alpha_k\cos\left[\frac{\pi k(2n+1)}{2N}\right],
\qquad
\alpha_k=
\begin{cases}
N^{-1/2}, & k=0,\\
(2/N)^{1/2}, & k>0.
\end{cases}
$$

Transform the right-hand side:

$$
\widehat b=C_H b C_W^{\mathsf T}.
$$

The eigenvalues of the discrete Laplacian are

$$
\lambda_{p,q}=2\left[
\cos\left(\frac{\pi p}{H}\right)+
\cos\left(\frac{\pi q}{W}\right)-2
\right],\quad 0\le p<H,\quad 0\le q<W.
$$

Solve and invert:

$$
\widehat\phi_{p,q}=
\begin{cases}
\widehat b_{p,q}/\lambda_{p,q}, & (p,q)\ne(0,0),\\
0, & (p,q)=(0,0),
\end{cases}
\qquad
\phi=C_H^{\mathsf T}\widehat\phi C_W.
$$

The sum of b telescopes to zero in exact arithmetic, satisfying the compatibility condition. The DC eigenvalue is zero. To avoid division by zero, the implementation temporarily replaces lambda(0,0) with 1, performs division, and explicitly sets the phase DC coefficient to zero **for every sample**.

No epsilon is added to the non-DC denominator. The constructor stores eps, but the active solver does not use it in the division.

Zero DC selects the zero-mean solution: gradients cannot determine the additive phase constant. After inverse transformation, the implementation also subtracts phi.mean(), taken over the entire batch and spatial grid. The preceding DC assignment already makes each sample zero-mean in exact arithmetic; the final subtraction removes a residual global numerical offset. The reference implementation below preserves this reduction.

DCT matrices, eigenvalues, and internal integration calculations use float64. The result is cast back to the original gx dtype, so output precision depends on that dtype. Precomputed arrays are buffers with no trainable parameters. The precomputed shape must match the input shape, and the solver and inputs must be on the same device. Supplied training/testing calls use 256 by 256 grids; the algorithm requires H,W>=2.

Source: [active solver and normalization](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/Net%20%281%29.py#L393-L458).

## 4. Runnable reference implementation

The following PyTorch function reproduces the active solver's operations. For readability it constructs matrices on every call; the repository class precomputes them.

~~~python
import math
import torch


def integrate_phase_dct(gx, gy):
    """Unit-spacing integration, with input/output shape (B, 1, H, W)."""
    if gx.ndim != 4 or gx.shape != gy.shape or gx.shape[1] != 1:
        raise ValueError("gx and gy must have matching shape (B, 1, H, W)")
    if gx.device != gy.device:
        raise ValueError("gx and gy must be on the same device")
    if not gx.is_floating_point() or not gy.is_floating_point():
        raise ValueError("Floating-point inputs are required")

    B, _, H, W = gx.shape
    if H < 2 or W < 2:
        raise ValueError("H and W must be at least 2")

    original_dtype = gx.dtype
    device, dtype = gx.device, torch.float64
    x = gx.squeeze(1).to(dtype)
    y = gy.squeeze(1).to(dtype)

    b = torch.zeros((B, H, W), dtype=dtype, device=device)
    b[:, :, 0] = x[:, :, 0]
    b[:, :, 1:W-1] = x[:, :, 1:W-1] - x[:, :, 0:W-2]
    b[:, :, W-1] = -x[:, :, W-2]
    b[:, 0, :] += y[:, 0, :]
    b[:, 1:H-1, :] += y[:, 1:H-1, :] - y[:, 0:H-2, :]
    b[:, H-1, :] -= y[:, H-2, :]

    def dct_matrix(N):
        n = torch.arange(N, dtype=dtype, device=device).reshape(1, N)
        k = torch.arange(N, dtype=dtype, device=device).reshape(N, 1)
        angle = math.pi * k * (2.0 * n + 1.0) / (2.0 * N)
        alpha = torch.full(
            (N, 1), math.sqrt(2.0 / N), dtype=dtype, device=device
        )
        alpha[0, 0] = math.sqrt(1.0 / N)
        return alpha * torch.cos(angle)

    CH, CW = dct_matrix(H), dct_matrix(W)
    q = torch.arange(W, dtype=dtype, device=device).reshape(1, W)
    p = torch.arange(H, dtype=dtype, device=device).reshape(H, 1)
    lam = 2.0 * (torch.cos(math.pi*q/W) + torch.cos(math.pi*p/H) - 2.0)
    lam[0, 0] = 1.0

    b_hat = CH @ (b @ CW.t())
    phi_hat = b_hat / lam
    phi_hat[:, 0, 0] = 0.0
    phi = CH.t() @ (phi_hat @ CW)
    phi = phi - phi.mean()
    return phi.unsqueeze(1).to(original_dtype)
~~~

The repository scripts import Net, whereas the pinned source filename is Net (1).py. For ordinary Python imports, provide that file as Training/Net.py or adapt the import. The standalone function above does not depend on the network file or trained weights.

## 5. Reproduction check

Run this after the reference function. It checks square and rectangular grids, minimum dimensions, multiple samples, recovery up to an additive constant, and invariance to unused boundary entries.

~~~python
torch.manual_seed(2026)

for H, W in [(2, 2), (9, 11), (256, 256)]:
    target = torch.randn(2, 1, H, W, dtype=torch.float64)
    gx, gy = torch.zeros_like(target), torch.zeros_like(target)
    gx[:, :, :, :-1] = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy[:, :, :-1, :] = target[:, :, 1:, :] - target[:, :, :-1, :]

    reconstructed = integrate_phase_dct(gx, gy)
    expected = target - target.mean(dim=(2, 3), keepdim=True)
    error = (reconstructed - expected).abs().max().item()
    assert error < 1e-9, (H, W, error)

    gx[:, :, :, -1] = 123.0
    gy[:, :, -1, :] = -456.0
    repeated = integrate_phase_dct(gx, gy)
    torch.testing.assert_close(repeated, reconstructed, rtol=0, atol=0)

    print(f"{H}x{W}: maximum absolute error = {error:.3e}")
~~~

This validates phase integration, not the neural network's prediction accuracy.

## 6. Offset conventions in training, evaluation, and export

| Stage | Implemented offset handling |
|---|---|
| DCTPoisson | Zero DC per sample, then subtract the global tensor mean |
| Training/validation phase loss | Subtract each ground-truth sample's spatial mean |
| test_pre.py phase-error calculation | Subtract the global ground-truth tensor mean; the default batch size is 1 |
| test_model.py save_pre | Subtract the minimum of the selected reconstructed image before saving |

An exported image can therefore have minimum zero even though the solver produces zero-mean phase. To reproduce export for one image, apply phi_export = phi - min(phi) after integration. This changes only the additive constant, not the phase differences, and does not recover an externally calibrated absolute phase offset.

Sources: [training phase loss](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/Main_op%20%283%29.py#L213-L220), [evaluation offset](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/test_pre.py#L163-L174), [export offset](https://github.com/su163266/Quantitative-gradient-reconstruction/blob/6855fe81006809c625ffb76c49ac46d870813796/Training/test_model.py#L341-L363).

## 7. Extension to physical grid spacing

The released implementation uses unit pixel spacing only. If adapting the algorithm to physical spacings hx,hy and physical derivative samples gx,gy, divide forward differences by the corresponding spacing. Divide the x and y contributions to b in Section 2 by hx and hy, respectively, and replace the eigenvalues with

$$
\lambda_{p,q}
=\frac{2[\cos(\pi q/W)-1]}{h_x^2}
+\frac{2[\cos(\pi p/H)-1]}{h_y^2}.
$$

The boundary and DC treatment remain the same. If starting with the repository's phase-increment outputs, first convert them to physical derivatives: gx_physical=gx_increment/hx and gy_physical=gy_increment/hy. Changing only the eigenvalues introduces an inconsistent scale. This is a possible extension, not a setting used by the released implementation.

