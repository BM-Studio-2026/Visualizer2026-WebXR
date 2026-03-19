# Visualizer2026 WebXR

**Immersive linear algebra — SVD, PCA, and least squares visualized in virtual reality.**

Live demo (Meta Quest / WebXR browser): **[Launch VR →](https://bm-studio-2026.github.io/Visualizer2026-WebXR/vr.html)**
Landing page: https://bm-studio-2026.github.io/Visualizer2026-WebXR/

---

## What is this?

Visualizer2026-WebXR lets you stand *inside* a matrix transformation. It animates the Singular Value Decomposition (SVD) of a matrix as a continuous three-stage physical deformation of a 3D point cloud — rotate by **V**, stretch by **Σ**, rotate by **U** — while you float in a Milky Way environment with ambient music and full VR controller interaction.

No installation. No app store. Open the link in a Meta Quest browser and tap **Enter VR**.

This is the WebXR companion to the [Visualizer2026](https://bm-studio-2026.github.io/Visualizer2026/) desktop project.

---

## Five Mathematical Scenarios

Cycle through scenarios with the **left grip button** (VR) or **S key** (desktop):

| Mode | Scenario | What you see |
|---|---|---|
| 0 | **3×3 SVD** | 3D point cloud + deforming grid animate through V rotation → Σ scaling → U rotation. Three presets: Symmetric, Rotation+Scale, Shear+Scale. |
| 1 | **2×3 Projection** (R³→R²) | A 3D cube collapses flat onto a tilted image plane — dimension reduction made visible. |
| 2 | **3×2 Lifting** (R²→R³) | A flat 2D square lifts and rotates into 3D space — dimension lifting. |
| 3 | **3D PCA** | 60-point cloud aligns to its principal axes, then squashes to the best-fit plane, then to the best-fit line. |
| 4 | **Least Squares** | Four planes in 3D space. A glowing sphere marks the least-squares solution; residual lines show the perpendicular distance to each plane. |

---

## VR Controls

| Input | Action |
|---|---|
| Right trigger (hold) | Advance t: 0 → 3 (animate transformation) |
| Left trigger (hold) | Reverse t: 3 → 0 |
| Right grip | Cycle 3×3 matrix preset (mode 0) |
| **Left grip** | **Cycle scenario mode (0–4)** |
| Right thumbstick Y | Zoom in / out |
| A button | Grab, drag, rotate and throw the scene |
| B button | Toggle ambient music |
| Left thumbstick click | Teleport to aimed position |

**Desktop controls:** Arrow keys scrub t · 1/2/3 or G = cycle matrix · S = cycle scenario · +/− = zoom · M = music

---

## The SVD Animation

The parameter `t ∈ [0, 3]` drives a smooth decomposition of any 3×3 matrix A = UΣVᵀ:

- **t = 0 → 1** — Rotate the point cloud by **V** (right singular vectors)
- **t = 1 → 2** — Scale along each axis by **Σ** (singular values)
- **t = 2 → 3** — Rotate by **U** (left singular vectors) to arrive at **A**

Stage transitions trigger a haptic pulse on both controllers, a visual ring expanding outward, and spoken stage descriptions (desktop only).

---

## Technology

| | |
|---|---|
| Engine | [Three.js](https://threejs.org) r183 |
| XR | WebXR Device API (xr-standard gamepad profile) |
| Build | [Vite](https://vitejs.dev) 8 |
| Math | Self-contained Jacobi SVD — no external math library |
| Deployment | GitHub Pages (static, zero backend) |
| Environment | ESO/S. Brunier Milky Way panorama (CC BY 4.0) |
| Audio | Web Audio API synthesized Cmaj9 ambient pad |
| Size | ~1350 lines of JavaScript, ~590 kB bundle |

The entire application is a single `main.js` file. SVD, PCA, and least-squares are all computed in plain JavaScript using a Jacobi eigendecomposition — no server, no WebAssembly, no dependencies beyond Three.js.

---

## Project Structure

```
├── main.js               # Full application (~1350 lines)
├── vr.html               # VR entry point
├── index.html            # Public landing page
├── vite.config.js        # Multi-page build, GitHub Pages base path
├── public/
│   └── sky.jpg           # Milky Way equirectangular photo (7.9 MB)
└── 8_LSE.py              # Streamlit reference script (least squares)
```

---

## Related

- [Visualizer2026](https://github.com/bm-studio-2026/Visualizer2026) — the main desktop project (Streamlit + Plotly)
- [Live Desktop App](https://bm-studio-2026.github.io/Visualizer2026/)

---

*BM Studio 2026 · Three.js · WebXR · Linear Algebra*
