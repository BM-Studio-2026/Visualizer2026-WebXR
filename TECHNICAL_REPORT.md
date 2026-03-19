# Visualizer2026 WebXR — Technical Report

**Project:** Visualizer2026-WebXR
**Repository:** https://github.com/BM-Studio-2026/Visualizer2026-WebXR
**Live Demo:** https://bm-studio-2026.github.io/Visualizer2026-WebXR/
**Author:** Brayden Miao (BM Studio 2026)
**Date:** March 2026

---

## Executive Summary

Visualizer2026-WebXR is an immersive, browser-based virtual reality application for exploring linear algebra concepts in three-dimensional space. Built with Three.js and the WebXR Device API, it runs directly in a VR headset (Meta Quest 2/3) without any native installation. The core idea is to animate the Singular Value Decomposition (SVD) of a matrix as a continuous three-stage physical deformation of a point cloud and bounding box, letting a user literally walk around and grab the transformation while it unfolds. The project extends this framework to four additional mathematical scenarios: dimension reduction (2×3 projection), dimension lifting (3×2 lifting), Principal Component Analysis (3D PCA), and overdetermined least-squares error fitting.

---

## 1. Project Motivation and Goals

Linear algebra is notoriously abstract. Students learn to compute SVD on paper but rarely develop intuition for what the three matrices U, Σ, V actually *do* geometrically. The goal of this project is to make that geometry visceral:

- A viewer in VR space *sees* the point cloud rotate, stretch, and rotate again as the three SVD stages animate.
- They can grab the entire scene with a controller and throw it, physically interacting with the mathematical object.
- Multiple scenarios demonstrate how the same SVD framework applies to non-square matrices, PCA, and least-squares fitting.

The project is a companion to the main Visualizer2026 desktop application (https://bm-studio-2026.github.io/Visualizer2026/), which uses Streamlit and Plotly for 2D/3D interactive plots. The WebXR version explores what becomes possible when the constraint of a flat screen is removed.

---

## 2. Architecture and Technology Stack

### 2.1 Build System

| Component | Choice | Rationale |
|---|---|---|
| Bundler | Vite 8 | Fast HMR, native ES modules, minimal config |
| 3D engine | Three.js r183 | Industry-standard WebGL/WebXR library |
| Math library | Custom JS (no mathjs in hot path) | Self-contained Jacobi SVD avoids large dependency |
| Deployment | GitHub Pages (gh-pages branch) | Zero-cost static hosting, automatic CDN |

The Vite config declares a multi-page build: `index.html` is a static landing page and `vr.html` loads the main application (`main.js`). The `base` path `/Visualizer2026-WebXR/` ensures assets resolve correctly on GitHub Pages.

```
/
├── index.html          # Public landing page (plain HTML, no JS bundle)
├── vr.html             # VR app entry point → main.js
├── main.js             # ~1350-line single-file application
├── vite.config.js      # Multi-page build, base path
├── public/
│   └── sky.jpg         # ESO Milky Way panorama (7.9 MB, CC BY 4.0)
└── dist/               # Built output, copied to gh-pages branch
```

### 2.2 Rendering Pipeline

The renderer runs two distinct paths depending on context:

- **Desktop:** `EffectComposer` → `RenderPass` → `UnrealBloomPass` (strength 0.35, radius 0.4, threshold 0.75) → `OutputPass`. Bloom gives the singular-vector arrows and glow effects a neon look.
- **VR (WebXR):** Direct `renderer.render(scene, camera)`. The post-processing composer is bypassed because WebXR requires rendering directly to the XR framebuffer; bloom cannot be trivially composed with it.

The renderer uses `ACESFilmicToneMapping` for consistent color grading across both modes.

### 2.3 Background Environment

On load, a `TextureLoader` attempts to fetch `sky.jpg` — a 6000×3000 equirectangular photograph of the Milky Way by ESO/S. Brunier (CC BY 4.0). If successful, it is assigned to `scene.background` with `EquirectangularReflectionMapping` and dimmed via `scene.backgroundIntensity = 0.14` so the mathematical visualization reads clearly. If the photo fails (e.g., slow connection), a procedurally generated starfield is already present as a fallback: 3000 background field stars, 500 colour-varied bright stars, 16000-point Milky Way band with galactic-plane geometry, a 4000-point galactic-centre glow cluster, and four additive-blended nebula patches in blue, red, teal, and purple.

---

## 3. Core Mathematics

### 3.1 Self-Contained Jacobi SVD

No external math library is used in the animation hot path. The SVD is computed using a Jacobi eigendecomposition algorithm implemented entirely in JavaScript:

1. **`jacobiEig3(S)`** — Jacobi iteration on a 3×3 symmetric matrix. Each step finds the largest off-diagonal element, computes a Givens rotation angle, and applies it to zero that element. Converges in ≤ 60 iterations for all tested inputs.

2. **`svd3(A)`** — Forms AᵀA (3×3 symmetric), calls `jacobiEig3`, sorts eigenvalues descending, extracts right singular vectors V (eigenvectors), then computes left singular vectors U = AV/σ column by column.

3. **`makeRotationalSVD(A)`** — Enforces det(U) = det(V) = +1 by flipping the sign of the last column when the determinant is negative. This ensures U and V are proper rotation matrices (SO(3)), not reflections, so the animation passes through smooth rotation paths rather than improper flips.

4. **`svd2x3(A)`** — For 2×3 matrices: forms AᵀA (3×3), calls `jacobiEig3`, then computes U (2×2) from U[:,j] = AV[:,j]/σⱼ.

5. **`svd3x2(A)`** — For 3×2 matrices: forms AAᵀ (3×3), calls `jacobiEig3` to get U (3×3), then computes V (2×2) from V[:,j] = AᵀU[:,j]/σⱼ.

6. **`pca3(pts)`** — Centers a 3D point cloud, forms the unnormalized covariance matrix C = XᵀX, calls `jacobiEig3` to get eigenvectors (principal components) and eigenvalues (proportional to variance). Returns mean, principal component matrix V, and standard deviations σᵢ = √(λᵢ/n).

7. **`lseSolve(planes)`** — Given m planes each defined by (A,B,C,D) with Ax+By+Cz+D=0, normalizes each equation, assembles the m×3 matrix and RHS vector, and solves the normal equations AᵀAx = Aᵀb via Cramer's rule on the 3×3 system. Returns the least-squares point, per-plane residual distances, and the normalized normals.

### 3.2 The SVD Animation Path

The central visual metaphor is a continuous 3-stage deformation parameterized by `t ∈ [0, 3]`:

| Stage | t range | Transformation |
|---|---|---|
| 0 | t = 0 | Identity — original point cloud |
| 1 | t ∈ [0,1] | Interpolated rotation by **V** (right singular vectors) |
| 2 | t ∈ [1,2] | Diagonal scaling by **Σ** (singular values) |
| 3 | t ∈ [2,3] | Interpolated rotation by **U** (left singular vectors) |

Each point p is mapped by the matrix M(t) = svdPath(t, {U,Σ,V}):
- Stage 1: M(t) = R(axis_V, t·θ_V)
- Stage 2: M(t) = R(axis_V, θ_V) · diag(1 + (t-1)(σ-1))
- Stage 3: M(t) = R(axis_V, θ_V) · Σ · R(axis_U, -(t-2)·θ_U)

At t=3, M(3) = V · Σ · Uᵀ = Aᵀ (the transpose of A, since SVD gives A = UΣVᵀ). The full transformation is the matrix A applied to the original points.

Rotations are interpolated using axis-angle decomposition (`axisAngle(R)` → `rotAxisAngle(axis, angle)`) which gives a single smooth geodesic path on SO(3), avoiding gimbal lock.

### 3.3 Scenario-Specific Paths

**Scenario 1 — 2×3 Projection (R³ → R²)**
Matrix A = [[1,2,0],[0,1,-1]]. Points start as a 3D Gaussian cluster. The 3-stage path:
1. Rotate by V (3×3): aligns the cluster with the singular directions.
2. Scale x' by σ₁, y' by σ₂; squash z' by factor (1-β): the cluster collapses toward the V₁,V₂ span.
3. Rotate within the V₁,V₂ subspace by the 2×2 matrix U: final position is the true Ax projection.

A semi-transparent blue plane (span V₁,V₂) and a white cube (original) plus red cube (collapsing) visualize the projection geometry.

**Scenario 2 — 3×2 Lifting (R² → R³)**
Matrix A = [[1,2],[0,1],[−1,0]]. Points start as a 2D Gaussian cluster at z=0. The 3-stage path is the reverse topology:
1. Rotate in-plane by 2D matrix V (2×2 rotation by angle θ_V).
2. Scale by σ₁, σ₂ in the xy-plane.
3. Apply 3D rotation U (3×3) via axis-angle interpolation, lifting the plane into 3D space.

A semi-transparent red xy-plane (input domain), white flat square (original 2D bounding box), red lifting square, and white cube (output 3D space computed from t=3 positions) form the visual context.

**Scenario 3 — 3D PCA**
60 points sampled from a tilted ellipsoidal "pancake" distribution. PCA is performed via eigendecomposition of the covariance matrix. The animation:
1. Rotates the cloud to align PC1, PC2, PC3 with the coordinate axes.
2. Squashes the PC3 component to zero — projecting to the best-fit plane.
3. Also squashes PC2 — projecting to the best-fit line (PC1).

Three colored arrows (red=PC1, green=PC2, blue=PC3) emanate from the cloud centroid, scaled by their singular values. A semi-transparent green plane shows the PC1-PC2 best-fit plane.

**Scenario 4 — Least Squares**
Four planes defined by equations Ax+By+Cz+D=0. The least-squares solution minimizes the sum of squared perpendicular distances to all planes. The scene is static: four colored semi-transparent plane quads with square wireframe borders, white line segments showing the residual from the LS point to each plane, and a glowing white sphere at the LS solution.

---

## 4. Rendering and Visual Design

### 4.1 Point Cloud

Points are rendered with `THREE.InstancedMesh` (SphereGeometry, radius 0.05, 8×6 segments) — a single draw call for all N points regardless of count. Two instanced meshes per scene:
- **Original (cyan, `0x00eeff`, opacity 0.45)** — ghost reference showing where points started.
- **Transformed (orange, `0xff7722`, opacity 1.0)** — the animated current positions.

Per-frame, `updateIM()` iterates all instances, sets each `dummy.position`, calls `dummy.updateMatrix()`, and calls `setMatrixAt()` — the standard Three.js instancing pattern.

### 4.2 Deforming 3D Grid (Mode 0 only)

A 5×5×5 lattice (G_VALS = [−1.5, −0.75, 0, 0.75, 1.5]) of line segments deforms with the matrix transformation. Grid lines are pre-computed as `Float32Array` base buffers (`gridBaseX/Y/Z`); each frame `applyMToFloatArray(M, base, tmp)` transforms them in-place and `setLineSegs()` updates the GPU buffer. Three separate `LineSegments` objects cover the YZ, XZ, and XY planes of lines. Material: deep blue `0x1a3a8a`, opacity 0.50.

### 4.3 Point Trails

A ring buffer of 10 trail snapshots (`TRAIL_N = 10`) captures the point positions every time `tParam` moves more than 0.025. Each trail is a `THREE.Points` object with opacity fading exponentially: `opacity = 0.30 × exp(−age × 2.2)`. This creates smooth ghost trails showing the path of the transformation as the user scrubs `t`.

### 4.4 Stage Pulse Ring

On each stage boundary crossing (floor(t) changes), a `THREE.TorusGeometry` pulse ring expands outward (`scale = 1 + age × 10`) and fades to zero opacity over 0.75 seconds. This provides an instantaneous visual cue that a new geometric stage has begun.

### 4.5 Singular Vector Arrows and Labels

`THREE.ArrowHelper` objects point along each right singular vector v_i, colored cyan/magenta/yellow, scaled proportional to the singular value σᵢ. Floating `THREE.Sprite` labels (rendered on a `CanvasTexture`) display the vector name and σ value with a colored glow effect.

---

## 5. VR Interaction System

### 5.1 Controller Mapping (xr-standard profile)

| Button | Index | Right Controller | Left Controller |
|---|---|---|---|
| Trigger | 0 | Advance t (+) | Reverse t (−) |
| Grip | 1 | Cycle 3×3 matrix preset | Cycle scenario mode |
| Thumbstick click | 3 | — | Teleport |
| A button | 4 | Grab / throw scene | — |
| B button | 5 | Toggle ambient music | — |
| Thumbstick Y | axes[3] | Zoom (scale root group) | — |

### 5.2 Grab and Throw

The grab system tracks delta-position and delta-rotation each frame while the A button is held:

```
Δpos = currentControllerPos − prevControllerPos
throwVel = Δpos / dt                      (velocity for post-release throw)
root.position += Δpos

ΔQ = currentControllerQuat × prevControllerQuat⁻¹
relativePoint = root.position − controllerPos
root.position = controllerPos + ΔQ(relativePoint)
root.quaternion = ΔQ × root.quaternion
```

A critical bug in early development caused violent scene shaking: `_dQuat.copy(_cQuat).multiply(_dQuat.copy(grabCtrlQuat).invert())` aliased `_dQuat` — the inversion overwrote the value being multiplied. The fix was a dedicated `_invQ` pre-allocated quaternion. After release, `throwVel` drives a damped physics simulation: `position += throwVel × dt; throwVel × = (1 − 6dt)`.

### 5.3 Teleportation

The left controller casts a ray downward each frame. On intersection with the invisible floor plane, a green reticle ring appears. On thumbstick click, the XR reference space is offset using `XRRigidTransform`:
```
newRefSpace = baseRefSpace.getOffsetReferenceSpace(
  new XRRigidTransform({x: −hitPos.x, y: 0, z: −hitPos.z, w: 1}, identity_rotation))
```
The `baseRefSpace` is stored at session start; all teleport offsets are always applied relative to it (not the current position) to avoid cumulative drift.

### 5.4 Haptic Feedback

On stage transitions, `triggerHaptics()` iterates all input sources and tries:
1. `gamepad.hapticActuators[0].pulse(intensity, duration)` — the WebXR spec API.
2. `gamepad.vibrationActuator.playEffect('dual-rumble', {...})` — a fallback for browsers that implement the Gamepad Vibration API instead.

Both controllers vibrate simultaneously, providing physical confirmation that a geometric stage boundary has been crossed.

### 5.5 Zoom

The right thumbstick Y axis (index 3 in the xr-standard mapping — note: *not* index 1, which is the touchpad slot on older profiles) scales the root group uniformly: `rootScale = clamp(rootScale − stickY × 0.9 × dt, 0.10, 2.0)`. This lets the user zoom the entire mathematical scene in and out relative to their body.

---

## 6. Audio System

### 6.1 Ambient Music

A Cmaj9 chord pad is synthesized entirely via the Web Audio API at runtime — no audio files required:

- Five sine oscillators at C2–D3: 65.41, 82.41, 98.00, 123.47, 146.83 Hz
- Each oscillator has ±2.5 cents micro-detune for warmth
- Per-oscillator LFO tremolo (0.04–0.11 Hz) with amplitude 0.025
- Shared low-pass filter at 520 Hz for warmth
- Shared delay reverb: 2.4 s delay, feedback gain 0.22, wet mix 0.16
- Master gain ramps from 0 to 0.09 over 5 seconds on start

Music auto-starts on first click (required by browser autoplay policy) and can be toggled with B (VR) or M (desktop).

### 6.2 Speech Synthesis

`window.speechSynthesis` announces stage transitions and matrix names when the user scrubs through the SVD stages:
- *"Stage 1. Rotating by V, aligning to the right singular vectors."*
- *"Stage 2. Scaling along the principal axes by the singular values."*
- *"Stage 3. Rotating by U, arriving at the final matrix."*

**Important limitation:** Speech synthesis audio is not routed through the WebXR audio pipeline on Meta Quest 2/3. Utterances are silently dropped when the browser is in immersive-vr mode. Speech is functional only on desktop browsers.

---

## 7. Information Display

### 7.1 Floating Info Panel

A `1024×1080` `CanvasTexture` mapped onto a `PlaneGeometry` (1.8 × 1.9 m) floats to the right of the scene at `(2.4, 1.6, −1.8)`. It is redrawn each frame using Canvas 2D API, showing:
- Current matrix or scenario name
- t value and stage description
- SVD decomposition: σ values, rotation axes and angles for U and V
- Color-coded legend for axes, singular vectors, and rotation axes
- Full VR controls reference

### 7.2 Wrist HUD

A `256×128` canvas is attached to the left controller (`ctrl.add(wristMesh)`) on the first VR frame. It shows t value, matrix/scenario name, and current stage — readable without turning away from the scene.

### 7.3 Desktop HUD

A positioned `div` overlay (top-left) shows the same information in HTML text for desktop development and presentation use.

---

## 8. Scenario Mode System

Five scenarios are accessed by cycling with left grip (VR) or S key (desktop):

| Mode | Name | Input Space | Output Space | Key Visual |
|---|---|---|---|---|
| 0 | 3×3 SVD | R³ | R³ | Deforming grid, cube, trails |
| 1 | 2×3 Projection | R³ | R² | Blue image plane, collapsing cube |
| 2 | 3×2 Lifting | R² | R³ | Red domain plane, lifting square, 3D cube |
| 3 | 3D PCA | R³ | R³→R²→R¹ | PC arrows, best-fit plane |
| 4 | Least Squares | — | R³ point | 4 planes, residuals, LS marker |

Mode 0 also has three presets cycled by right grip (VR) or 1/2/3/G keys (desktop):
1. **Symmetric** — `[[1.0, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 0.8]]`
2. **Rotation+Scale** — R(36°) × diag(1.8, 0.7, 1.2)
3. **Shear+Scale** — `[[1.5, 0.8, 0.2], [0.0, 1.0, 0.5], [0.1, 0.0, 0.6]]`

---

## 9. Development Challenges and Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Scene shaking violently during grab | Quaternion self-aliasing: `_dQuat` overwritten before being read in `multiply()` | Pre-allocate dedicated `_invQ` for the inversion step |
| Zoom not working on Quest | `axes[1]` is touchpad slot in xr-standard; thumbstick Y is `axes[3]` | Use `axes[3] ?? axes[1] ?? 0` with fallback |
| Left controller not vibrating | `hapticActuators` API not exposed consistently | Added `vibrationActuator.playEffect('dual-rumble',...)` fallback |
| Speech silent in VR | `speechSynthesis` audio not routed through WebXR pipeline on Quest | Documented as known limitation; desktop speech still functional |
| SVD for non-square matrices | Only had 3×3 Jacobi implementation | Extended to `svd2x3` and `svd3x2` via eigendecomposition of AᵀA and AAᵀ respectively |

---

## 10. Performance Considerations

- **InstancedMesh** ensures all N points render in a single draw call. With 60 points (PCA), this is trivial; it scales well to hundreds of points.
- **Pre-allocated Float32Arrays** (`gridBaseX/Y/Z`, `gridTmpX/Y/Z`) avoid GC pressure from the deforming grid, which otherwise would allocate fresh arrays every frame.
- **Pre-allocated Three.js temporaries** (`_cPos`, `_cQuat`, `_dPos`, `_dQuat`, `_invQ`, `_rp`) in the grab system serve the same purpose.
- **Bloom is disabled in VR** — `UnrealBloomPass` requires a render target that is incompatible with the WebXR framebuffer. Desktop gets full post-processing; VR prioritizes frame rate.
- The built JS bundle is ~590 kB gzipped to ~151 kB — dominated by Three.js (~500 kB uncompressed). There is no tree-shaking penalty because Three.js is already modular and only the used classes are imported.

---

## 11. Deployment

The project uses a git worktree at `C:\temp\gh-pages-work` tracking the `gh-pages` branch. The deployment workflow is:

```bash
npm run build                    # Vite produces dist/
cd C:\temp\gh-pages-work
git rm assets/*.js               # Remove old hashed bundle
cp -r dist/* .                   # Copy new build
git add -A && git commit && git push
```

GitHub Actions (via GitHub Pages) serves the branch automatically. The `base: '/Visualizer2026-WebXR/'` in `vite.config.js` ensures all asset paths resolve correctly at the subdirectory URL.

---

## 12. Summary

Visualizer2026-WebXR demonstrates that a single 1350-line JavaScript file — no game engine, no server, no install — can deliver an immersive, interactive, mathematically rigorous VR experience in a standard browser. The key design decisions that make this work:

1. **Self-contained math:** A Jacobi SVD in plain JS eliminates any server dependency and keeps the entire application stateless.
2. **Axis-angle interpolation for smooth animation:** Decomposing rotation matrices into axis+angle gives geodesic paths on SO(3), producing smooth and predictable stage transitions.
3. **InstancedMesh for all point clouds:** Regardless of scenario, all points are a single draw call. The same architecture scales from 30 to hundreds of points without refactoring.
4. **Separation of build path and render path:** Bloom and post-processing on desktop, direct render in VR — one codebase, two visual modes, driven by `renderer.xr.isPresenting`.
5. **Scenario mode architecture:** All five mathematical demonstrations (3×3 SVD, 2×3 projection, 3×2 lifting, PCA, LSE) share the same `tParam`, trail system, HUD, haptics, and input handling. Adding a new scenario means writing one `buildScenarioN()` and one `updateScenarioN()` function.

The result is an educational tool that makes abstract linear algebra tangible: the user can stand inside a matrix transformation, scrub through its three geometric stages, grab it and throw it across a Milky Way backdrop, and cycle through five distinct mathematical demonstrations without leaving the headset.

---

*Report generated March 2026. Source: D:\Visualizer2026-WebXR\main.js (~1352 lines), build toolchain: Vite 8, Three.js r183.*

Project Summary
                                                                                                                                    
  Visualizer2026-WebXR is an immersive browser-based VR application that teaches linear algebra by letting a user physically stand
  inside a matrix transformation. It runs on Meta Quest 2/3 with no installation — just a URL.                                        
  What it does                                                                                                                        
  The core experience animates SVD (Singular Value Decomposition) as a continuous three-stage physical deformation of a 3D point      cloud: rotate by V, stretch by Σ, rotate by U. The user scrubs through the stages with VR triggers, grabs and throws the scene,
  teleports, and zooms — all while floating inside a Milky Way environment with ambient music.                                      
  
  Five mathematical scenarios are accessible by cycling with the left grip button:                                                    
  ┌─────┬────────────────┬────────────────────────────────────────────────────────────────────────────────┐                           │  #  │    Scenario    │                              Concept demonstrated                              │
  ├─────┼────────────────┼────────────────────────────────────────────────────────────────────────────────┤                         
  │ 0   │ 3×3 SVD        │ Full matrix decomposition with deforming grid                                  │
  ├─────┼────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ 2×3 Projection │ A cube collapses flat onto a tilted image plane (R³→R²)                        │                         
  ├─────┼────────────────┼────────────────────────────────────────────────────────────────────────────────┤                           │ 2   │ 3×2 Lifting    │ A flat square lifts and deforms into 3D space (R²→R³)                          │                         
  ├─────┼────────────────┼────────────────────────────────────────────────────────────────────────────────┤                           │ 3   │ 3D PCA         │ Point cloud aligns to principal axes, then squashes to a plane, then to a line │
  ├─────┼────────────────┼────────────────────────────────────────────────────────────────────────────────┤                           │ 4   │ Least Squares  │ Four planes with a glowing LS solution point and residual lines                │
  └─────┴────────────────┴────────────────────────────────────────────────────────────────────────────────┘                           
  Key technical decisions                                                                                                             
  - Self-contained Jacobi SVD in JavaScript — no math library, no server, fully static                                                - Axis-angle interpolation — smooth geodesic rotation paths on SO(3), no gimbal lock
  - InstancedMesh — all N points in one GPU draw call regardless of count                                                             - Dual render path — bloom post-processing on desktop; direct WebXR render in VR
  - Single shared architecture — all five scenarios reuse the same tParam, trail system, haptics, HUD, and input handlers; each adds   only a buildScenarioN() and updateScenarioN() function                                                                             - 1350 lines, one file, zero backend — deployed as a static GitHub Pages site