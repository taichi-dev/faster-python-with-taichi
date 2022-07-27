import numpy as np
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.vulkan)

W, H = 800, 480
pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
Du, Dv, feed, kill = 0.160, 0.080, 0.060, 0.062
#Du, Dv, feed, kill = 0.210, 0.105, 0.018, 0.051

uv_grid = np.zeros((2, W, H, 2), dtype=np.float32)
uv_grid[0, :, :, 0] = 1.0
rand_rows = np.random.choice(range(W), 50)
rand_cols = np.random.choice(range(H), 50)
uv_grid[0, rand_rows, rand_cols, 1] = 1.0
uv = ti.Vector.field(2, ti.f32, shape=(2, W, H))
uv.from_numpy(uv_grid)

palette = ti.Vector.field(4, ti.f32, shape=(5,))
#palette[0] = [0, 0, 0, 0]  # [0.0, 0.0, 0.0, 0.31372549]
#palette[1] = [0, 1, 0, 0.2]  # [1.0, 0.1843, 0.53333333, 0.376470588]
#palette[2] = [1.0, 1.0, 0.0, 0.2078431373]  # [0.854901961, 1.0, 0.5333333, 0.3882353]
#palette[3] = [1, 0, 0, 0.4]  # [0.376471, 1.0, 0.47843, 0.39215686]
#palette[4] = [1.0, 1.0, 1.0, 0.6]
palette[0] = [0.0, 0.0, 0.0, 0.3137]
palette[1] = [1.0, 0.1843, 0.53333, 0.37647]
palette[2] = [0.8549, 1.0, 0.53333, 0.388]
palette[3] = [0.376, 1.0, 0.478, 0.392]
palette[4] = [1.0, 1.0, 1.0, 1]


@ti.kernel
def compute(phase: int):
    for i, j in ti.ndrange(W, H):
        cen = uv[phase, i, j]
        lapl = uv[phase, i + 1, j] + uv[phase, i, j + 1] + uv[phase, i - 1, j] + uv[phase, i, j - 1] - 4.0 * cen
        du = Du * lapl[0] - cen[0] * cen[1] * cen[1] + feed * (1 - cen[0])
        dv = Dv * lapl[1] + cen[0] * cen[1] * cen[1] - (feed + kill) * cen[1]
        val = cen + 0.5 * tm.vec2(du, dv)
        uv[1 - phase, i, j] = val


@ti.kernel
def render():
    for i, j in pixels:
        value = uv[0, i, j].y
        color = tm.vec3(0)
        if value <= palette[0].w:
            color = palette[0].xyz

        for k in range(4):
            c0 = palette[k]
            c1 = palette[k + 1]
            if c0.w < value < c1.w:
                a = (value - c0.w) / (c1.w - c0.w)
                color = tm.mix(c0.xyz, c1.xyz, a)

        pixels[i, j] = color


gui = ti.ui.Window("Reaction-Diffusion", res=(W, H))
canvas = gui.get_canvas()
substeps = 60
i = 0
while gui.running:
    for _ in range(substeps):
        compute(i % 2)
        i += 1
    render()
    canvas.set_image(pixels)
    #gui.write_image(f"frame{i // substeps:03d}.png")
    gui.show()
