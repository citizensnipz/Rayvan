# Math Visualizer / Geometry Lab

Frontend-only research tooling inside the existing Rayvan Research Console. Original 2D modules remain available; spatial modules add an XY/3D switch. No model, training, backend routing, or telemetry collection is changed.

## Dependencies

Runtime: `three`, `@react-three/fiber`, `@react-three/drei`. Types: `@types/three`.
No new mathematical framework or backend service is used.

## Architecture

- `MathVisualizerPage.tsx`: preserves original 2D modules and adds the spatial workspace switch.
- `SpatialVisualizer.tsx`: lazy module selection, not one monolithic scene.
- `components/VisualizationShell.tsx`: shared equation → controls → computed values → geometry → interpretation structure, with scientific-context labels.
- `components/TimelinePlayer.tsx`: previous/next, scrub, play/pause, reset; stops playback when the document is hidden; uses slower discrete steps for reduced motion.
- `components/AnimatedMathStep.tsx`: reusable deterministic transformation/trajectory frames.
- `components/SpatialControls.tsx`: finite bounded 3D vector and row-major 3×3 matrix controls.
- `components/ProjectionControls.tsx`: raw axes, locally computed PCA, and disabled custom/UMAP options when no corresponding data is supplied.
- `renderers/scene.ts`: renderer-independent points, arrows, paths, clouds, surfaces, regions.
- `renderers/GeometryView.tsx`: XY projection and lazy 3D renderer; explicit WebGL2-unavailable fallback.
- `renderers/ThreeDScene.tsx`: shared axes/grid, arrows, point clouds, surface meshes, paths, draggable points and competence regions. Orbit/zoom and scene-fitting reset; mathematical z-up convention.
- `core/geometry.ts`: vectors, matrices, symmetric Jacobi eigensolver, SVD, covariance/PCA, orthogonal interpolation, neighbourhoods and deterministic toy samples.
- `core/differential.ts`: analytic landscapes, gradients, Hessians, graph-surface curvatures and bounded-domain optimization.
- `core/routing.ts`: isolated toy score/affine-transform simulator. It never calls EMC routing code.
- `core/data.ts`: projected-state contracts including original dimensions, projection metadata, IDs/steps, expert scores, loss/regret and prototype metadata.
- `visualizations/representations/`: independent embedding, intrinsic-dimension and neighbourhood experiments; `RepresentationLab` is only a dispatcher, not a title-switched shared experiment.
- `core/representations.ts`: stable-ID metric neighbours, normalization, deterministic clusters/structures, PCA projection and covariance participation ratio.

R3F handles declarative geometry/material disposal. Point clouds use one GPU point buffer per cloud rather than one mesh per sample. Surfaces are capped at 36×36 cells; examples use 80–240 samples and at most six prototype regions. Only the selected lab mounts a Canvas. It uses demand rendering with damping disabled and DPR capped at 1.5. There are no permanent animation loops or per-frame React state updates; playback advances discrete inspectable frames. Geometry changes and camera interactions request rendering.

## Implemented lab families

| Family | Concepts |
| --- | --- |
| Vector | 3D vectors, dot/cosine/distance, addition/subtraction/scaling, linear combinations, projection, Gram–Schmidt, conceptual gradient-direction comparison |
| Matrix | 3×3 transformations, orthonormal change of basis, real XY eigenvectors/eigenvalues, full 3×3 SVD stages/spectrum, rank/null space/volume, manually edited Jacobian neighbourhoods |
| PCA | Raw 3D cloud → mean centring → covariance principal directions → PCA basis → discard PC3; retained/discarded variance |
| Surface | Bowl, anisotropic valley, saddle, nonconvex multi-basin, plane and wave; gradient descent path, vector fields, Hessian eigen-directions, moving tangent plane/normal, local samples, graph-surface curvature |
| Embeddings | Three coloured synthetic clusters, separation/spread controls, sample-pair comparisons, animated unit normalization, Euclidean/angular nearest IDs and explicit PCA 3D→2D reduction |
| Intrinsic dimension | Line, flat sheet, curved sheet and volume; animated unfolding, thickness, covariance axes, variance spectrum and explicitly labelled global linear effective dimension |
| Local neighbourhoods | Editable/draggable query, k, Euclidean/cosine search, ranked distances, local covariance/density and colour-coded neighbour changes after PCA reduction |
| EMC | Normalized need, editable prototypes/widths, Euclidean/cosine/diagonal Mahalanobis comparison, sequential affine routing, exponential refractory penalty, transformation-shape matching, 3D manual latent trajectory and neighbourhood evolution |

All five originally disabled entries—eigenvectors, PCA, SVD, Jacobian and Hessian—now open substantive labs.

## Scientific boundaries and intentional limits

- Every example is synthetic. No actual high-dimensional latent vectors, gradients, expert Jacobians or competence measurements are loaded.
- The covariance participation ratio is a global linear spread proxy, NOT an intrinsic-dimension estimator. A curved two-parameter sheet can have three nonzero covariance eigenvalues. Thickness adds off-structure variation.
- Embedding comparisons use full toy coordinates even when the display drops PC3. Neighbourhood searches explicitly recompute in the selected search space; PCA centring changes the angular origin as well as discarding a component. Undefined zero-vector cosine comparisons are excluded.
- Raw dimension choices can repeat axes deliberately; the UI warns that dropping/duplicating dimensions changes geometry.
- PCA currently operates on toy 3D samples. All three components retain the cloud's geometry; the explicit final 3D→2D stage discards PC3. Low discarded variance does not prove routing information is preserved.
- Eigenvector highlighting is restricted to uncoupled real XY blocks of a 3×3 matrix. Complex XY pairs are reported explicitly; arbitrary 3D nonsymmetric eigenvectors and repeated-power animation are deferred.
- SVD rotation stages preserve lengths for proper orthogonal factors. An improper factor includes an explicitly described reflection; passing continuously through a reflection necessarily crosses a singular intermediate state.
- Jacobians are manually supplied local linear maps; nonlinear differentiation and actual model-state extraction are deferred.
- Principal graph-surface curvature values are computed, but principal-curvature direction drawing is deferred. Hessian eigen-directions are labelled separately: they are not generally principal surface-curvature directions.
- The manifold curve is a sampled candidate path through the surface, NOT a shortest-geodesic solution or a certified distance bound.
- Mahalanobis uses an explicitly stated positive diagonal toy covariance (0.3, 2, 1); learned/full routing covariance is deferred.
- Prototype spheres are intuitive width markers, not a final competence model. Scores are uncalibrated toy scores, not probabilities.
- Inhibition is temporary and exponentially decaying; sequential routing is a sandbox implementation only.
- Neighbourhood evolution currently uses affine transforms, not nonlinear bending. Bounding-ball density is a toy proxy.
- UMAP remains Coming Soon. File import, backend projection extraction, multiple-run comparison and persistent saved sequences are deferred.
- Original 2D and spatial workspaces have separate temporary state; changing concepts resets their sandbox.

## Verification

```sh
npm ci
npm run test:math
npm run build
npm run tauri build -- --no-bundle
```

The implementation environment passes TypeScript/production build and 40 numerical/DOM tests. Tests include SVD reconstruction across dense/degenerate matrices, orthogonality, PCA covariance, analytic derivatives against finite differences, curvature signs, metric-dependent winners, inhibition recovery, all lab families' numeric controls and playback, and explicit WebGL fallback behavior. Representation regressions verify distinct equations/controls/scenes, normalization, projection, structure ranks, zero cosine queries, stable neighbour IDs and reset behaviour. No lint command is configured.

The native build cannot start because Cargo is absent. Browser access to the local app is blocked (`ERR_BLOCKED_BY_CLIENT`). Therefore actual WebGL rendering, orbit/drag/reset behavior, Tauri-window resizing, native Console integration, Three.js console warnings, GPU idle usage and runtime resource-leak checks remain UNVERIFIED. DOM tests exercise the XY fallback, not a real WebGL context.

Build performance note: the lazy 3D chunk is approximately 972 kB / 262 kB gzip; the pre-existing main/ECharts chunk is approximately 980 kB / 321 kB gzip. Vite warns about large chunks. Lazy loading prevents the 3D chunk from loading for the original 2D workspace. GPU responsiveness still needs measurement on the target desktop.

### Required native visual smoke pass

1. Run `npm ci` and `npm run tauri dev`; visit every new category.
2. Orbit, zoom, edit numeric values, drag coloured point handles and reset the camera.
3. Scrub/step/play/pause each animated scene, then leave it paused; check GPU/CPU activity settles.
4. Check SVD factor boundaries, collapse/null space, zero-vector comparisons, PCA flattening, descent leaving domain, and routing score updates.
5. Resize between 1480×940 and the configured 1120×720 minimum; inspect narrower browser layouts separately.
6. Switch repeatedly between 2D/3D and labs; inspect WebGL console warnings and resource counts for leaks.
7. Verify New experiment, Live run, History and reports during a native research run. No training run should be started solely to open Math Visualizer.

Reference for demand rendering: [React Three Fiber performance guidance](https://r3f.docs.pmnd.rs/advanced/scaling-performance).
