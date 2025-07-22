// examples/stencil.cl
// 2D Jacobi‑Stencil (4‑Point)

__kernel void jacobi(
    __global const float* src,
    __global       float* dst,
    const int width
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    if (x == 0 || y == 0 || x == width - 1 || y == width - 1) {
        // Rand: Kopieren
        dst[idx] = src[idx];
    } else {
        // Innen: 4‑Punkt‑Stencil
        float up    = src[(y - 1) * width + x];
        float down  = src[(y + 1) * width + x];
        float left  = src[y * width + x - 1];
        float right = src[y * width + x + 1];
        dst[idx] = 0.25f * (up + down + left + right);
    }
}
