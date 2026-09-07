# Math Visualizer

A frontend-only workspace in the desktop Research Console. Select **Math Visualizer** in the primary sidebar; it works independently of the experiment-schema loading gate.

## Modules and responsibilities

- `MathVisualizerPage.tsx`: selects the active concept within the existing app layout.
- `components/MathCategorySidebar.tsx`: Foundational, Advanced and EMC registry; unfinished concepts are disabled and labelled Coming later.
- `components/VisualizationShell.tsx`: consistent equation, controls, geometry, computed values and interpretation layout.
- `components/ParameterControls.tsx`: labelled finite, bounded numeric/range controls.
- `components/VectorCanvas.tsx`: responsive SVG, equal-scale axes, grid, labelled arrows, legends and pointer coordinates. Gradient dragging holds the plot scale steady during a gesture.
- `components/TimelinePlayer.tsx`: generic typed frame playback, scrubbing and step navigation. `useSequence` provides manual snapshots for independent mathematical examples.
- `visualizations/VectorVisualizations.tsx`: vector basics, norm, dot product, cosine, distance and projection share the same two-vector maths and drawing primitives.
- `visualizations/MatrixTransformation.tsx`: column-vector convention, transformed grid/basis/point cloud, coordinate table, determinant and presets.
- `visualizations/GradientVisualization.tsx`: editable positive quadratic weights, contours, true gradient vector and one descent step. Large learning rates deliberately demonstrate overshoot.
- `visualizations/LatentTransformation.tsx`: editable initial state and deltas; later states are derived, never stored independently.
- `math.ts`: pure calculations, including explicit undefined results for zero-vector cosine/projection.
- `types.ts`: vectors, matrices, states and generic `Trajectory<T>` / `Frame<T>` contracts.
- `math.css`: styles scoped to the Math Visualizer using the Console's existing tokens. No graphics dependency was added.

## Manual sequences and extension points

Controls edit the selected frame. Add step copies it for ordinary mathematical examples; Load example sequence replaces the local sequence. Playback stops at the last frame and can restart. Sequences are capped at 24 frames and reset when leaving their concept; saving/importing is a future iteration.

Latent frames represent transitions: selecting step 0 shows z0, delta0 and z1. Add step appends a delta; changing an earlier delta recomputes every later state. The dashed path shows the complete trajectory, including future steps.

`LatentTrajectoryCanvas` accepts a typed `Trajectory<Vec2>` and selected transition. A future adapter can supply projected vectors and run/projection metadata without changing its renderer. File parsing, dimensionality reduction, run comparison and telemetry subscriptions remain outside these components. Adapters must validate finite coordinates and frame ordering before supplying data. If using nonlinear projection, projected endpoint displacement is not generally the projection of the original delta; preserve that distinction in future telemetry labels.

Eigenvectors/eigenvalues, PCA, SVD, Jacobian and Hessian/curvature remain Coming later. No EMC model, routing, training or backend behavior is changed.

## Checks

```sh
npm ci
npm run test:math
npm run build
npm run tauri build -- --no-bundle
```

`test:math` runs Node's test runner with tsx and jsdom (test-only dependencies). It checks numerical identities and finite differences, undefined cases, every implemented module's controls, SVG changes, matrix presets, timeline editing/scrubbing/play/pause and latent propagation. DOM tests do not replace visual or native-window checks.

Verification in the implementation environment:

- Frontend TypeScript + Vite production build: passed; Vite reports a large-bundle warning.
- Math and DOM interaction tests: 13 passed.
- Native Tauri build: blocked before compilation because Cargo is not installed.
- Existing Python Console/projection tests: could not start because pytest is not installed.
- Browser visual/manual interaction and resize checks: blocked because the available browser rejects the local dev-server URL. Not claimed as passed.

Desktop follow-up: check each concept at 1480×940 and the configured minimum 1120×720; verify matrix identity/rotation/reflection/collapse, zero vectors, gradient dragging and overshoot, and timeline playback. Navigate back to New experiment, Live run and History during a run to verify native integration. Review narrow responsive layouts in browser devtools. These native/visual checks remain outstanding.
