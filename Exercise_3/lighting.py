import numpy as np


def _as_light_array(x: np.ndarray) -> np.ndarray:
    """
    Normalize a light position/intensity input into a (N, 3) float array.

    Accepts any of the following and always returns shape (N, 3):
      - a single (3,) vector           -> one light  -> (1, 3)
      - a (N, 3) array                 -> N lights   -> (N, 3)
      - a list/tuple of (3,) vectors   -> N lights   -> (N, 3)

    Args:
        x : light position or intensity in one of the forms above

    Returns:
        (N, 3) float array
    """
    if isinstance(x, (list, tuple)):
        return np.array([np.asarray(xi, dtype=float).reshape(3) for xi in x],
                        dtype=float)

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:   # single (3,) vector
        return x.reshape(1, 3)
    return x


def light(pt: np.ndarray, nrm: np.ndarray, vclr: np.ndarray,
          cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: float,
          l_pos, l_int, l_amb: np.ndarray) -> np.ndarray:
    """
    Compute the Phong illumination (reflected RGB radiance) of a surface point.

    Args:
        pt      : (3,) coordinates of the surface point
        nrm     : (3,) outward unit normal at pt (toward the viewer side)
        vclr    : (3,) surface RGB colour / albedo at pt, each channel in [0, 1]
        cam_pos : (3,) camera (observer) position in WCS
        ka      : ambient reflection coefficient
        kd      : diffuse reflection coefficient
        ks      : specular reflection coefficient
        n       : Phong (shininess) exponent
        l_pos   : position(s) of one or more point light sources, (N, 3)
        l_int   : intensity(ies) of one or more point light sources, (N, 3)
        l_amb   : (3,) ambient light intensity

    Returns:
        pt_l : (3,) reflected RGB intensity for a single point, or (M, 3) when
               nrm is given as (M, 3)
    """
    # Single fixed point -> L and V directions are the same for every shaded
    # sample (per the assignment, computed once from the triangle barycenter)
    pt = np.asarray(pt, dtype=float).reshape(3)
    cam_pos = np.asarray(cam_pos, dtype=float).reshape(3)
    l_amb = np.asarray(l_amb, dtype=float).reshape(3)

    # Accept a single normal/colour or a batch; work internally as (M, 3)
    nrm = np.asarray(nrm, dtype=float)
    vclr = np.asarray(vclr, dtype=float)
    single = (nrm.ndim == 1)

    N = np.atleast_2d(nrm)    # (M, 3)
    C = np.atleast_2d(vclr)   # (1, 3) or (M, 3)
    M = N.shape[0]
    if C.shape[0] == 1 and M > 1:
        C = np.broadcast_to(C, (M, 3))

    # Normalise the surface normals row-wise (defensive)
    nlen = np.linalg.norm(N, axis=1, keepdims=True)
    nlen[nlen < 1e-12] = 1.0
    N = N / nlen

    # View direction V: from the surface point toward the camera (constant)
    V = cam_pos - pt
    v_len = np.linalg.norm(V)
    if v_len > 1e-12:
        V = V / v_len

    # Ambient term (independent of the point light sources)
    I = ka * (l_amb[None, :] * C)   # (M, 3)

    # Normalise the light inputs to (N, 3) so one or many sources work the same
    L_pos = _as_light_array(l_pos)
    L_int = _as_light_array(l_int)

    # Diffuse + specular, accumulated over every light source
    for pos, inten in zip(L_pos, L_int):
        # Light direction L: from the surface point toward the source (constant)
        L = pos - pt
        l_len = np.linalg.norm(L)
        if l_len < 1e-12:
            continue
        L = L / l_len

        # Lambertian factor per sample; only lit where the light faces the front
        ndotl = N @ L
        lit = ndotl > 0.0
        if not np.any(lit):
            continue

        # Diffuse contribution (modulated by surface colour C)
        diff = kd * ndotl[:, None] * inten[None, :] * C
        I = I + np.where(lit[:, None], diff, 0.0)

        # Specular contribution (classic Phong: reflect L about N).
        # Per the course notes, specular reflects the light-source colour and
        # is not tinted by the surface albedo -> white/coloured highlights.
        R = 2.0 * ndotl[:, None] * N - L[None, :]   # (M, 3)
        rdotv = R @ V   # (M,)
        spec_mask = lit & (rdotv > 0.0)
        if np.any(spec_mask):
            spec = ks * (np.clip(rdotv, 0.0, None)[:, None] ** n) * inten[None, :]
            I = I + np.where(spec_mask[:, None], spec, 0.0)

    return I[0] if single else I
