import { test, after } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import React, { act } from "react";
const dom = new JSDOM('<!doctype html><div id="root"></div>', {
  url: "http://localhost",
});
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true,
});
const { createRoot } = await import("react-dom/client");
const { VectorLab } =
  await import("../src/research/math/visualizations/spatial/VectorLab.tsx");
const { MatrixLab } =
  await import("../src/research/math/visualizations/spatial/MatrixLab.tsx");
const { PCALab } =
  await import("../src/research/math/visualizations/spatial/PCALab.tsx");
const { SurfaceLab } =
  await import("../src/research/math/visualizations/spatial/SurfaceLab.tsx");
const { RepresentationLab } =
  await import("../src/research/math/visualizations/spatial/RepresentationLab.tsx");
const { RoutingLab } =
  await import("../src/research/math/visualizations/emc/RoutingLab.tsx");
const { ShapeLab } =
  await import("../src/research/math/visualizations/emc/ShapeLab.tsx");
const { LatentLab3D } =
  await import("../src/research/math/visualizations/emc/LatentLab3D.tsx");
const root = createRoot(document.getElementById("root"));
let key = 0;
const render = async (Component, props = {}) =>
  act(() =>
    root.render(React.createElement(Component, { ...props, key: ++key })),
  );
const metrics = () => document.querySelector(".math-values").textContent;
const geometry = () => document.querySelector("svg").outerHTML;
async function fill(label, value) {
  const input = [...document.querySelectorAll("input")].find(
    (n) => n.getAttribute("aria-label") === label,
  );
  assert.ok(input, `Missing ${label}`);
  await act(() => {
    Object.getOwnPropertyDescriptor(
      dom.window.HTMLInputElement.prototype,
      "value",
    ).set.call(input, String(value));
    input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}
async function click(text) {
  const button = [...document.querySelectorAll("button")].find(
    (n) => (n.getAttribute("aria-label") || n.textContent) === text,
  );
  assert.ok(button, `Missing ${text}`);
  await act(() =>
    button.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })),
  );
}
async function select(label, value) {
  const input = [...document.querySelectorAll("label")]
    .find((n) => n.textContent.startsWith(label))
    ?.querySelector("select");
  assert.ok(input, label);
  await act(() => {
    input.value = value;
    input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}
after(async () => {
  await act(() => root.unmount());
  dom.window.close();
});
test("spatial vector labs update their numeric and projected geometry states", async () => {
  for (const mode of [
    "vectors",
    "dot",
    "cosine",
    "projection",
    "operations",
    "basis",
    "gram",
    "gradient-similarity",
  ]) {
    await render(VectorLab, { mode });
    const before = metrics(),
      svg = geometry();
    await fill("u x", 3);
    assert.notEqual(metrics(), before, mode);
    assert.notEqual(geometry(), svg, mode);
  }
  await render(VectorLab, { mode: "operations" });
  await select("Operation", "scalar");
  await fill("Scalar α", 2);
  assert.match(metrics(), /\(4, 2, 1\)/);
  await render(VectorLab, { mode: "gram" });
  await fill("u x", 0);
  await fill("u y", 0);
  await fill("u z", 0);
  assert.match(document.body.textContent, /undefined/);
  assert.doesNotMatch(document.body.textContent, /NaN/);
});
test("every matrix lab exposes working 3×3 controls, animation and singular cases", async () => {
  for (const mode of [
    "matrix",
    "eigen",
    "svd",
    "jacobian",
    "rank",
    "basis-change",
  ]) {
    await render(MatrixLab, { mode });
    const before = metrics(),
      svg = geometry();
    await fill("A row 1 column 1", 2.5);
    assert.notEqual(metrics(), before, mode);
    if (mode !== "basis-change") assert.notEqual(geometry(), svg, mode);
    await fill("Timeline frame", 0);
    await fill("Timeline frame", 60);
    assert.match(document.body.textContent, /frame 61 \/ 61/);
  }
  await click("Collapse");
  assert.match(metrics(), /Rank \(relative tolerance 1e−7\)2/);
  assert.match(metrics(), /Nullity1/);
});
test("PCA stages and synthetic cloud controls are connected", async () => {
  await render(PCALab);
  const before = metrics();
  await fill("Off-structure noise", 0.8);
  assert.notEqual(metrics(), before);
  await fill("Timeline frame", 80);
  assert.match(document.body.textContent, /Discard third component/);
  await select("Cloud shape", "line");
  assert.match(document.body.textContent, /PCA/);
  assert.doesNotMatch(metrics(), /NaN|Infinity/);
});
test("all surface labs support edited landscapes and scrubbed current values", async () => {
  for (const mode of ["descent", "manifold", "curvature", "hessian", "field"]) {
    await render(SurfaceLab, { mode });
    const before = metrics();
    await select("Landscape", mode === "manifold" ? "bowl" : "wave");
    assert.notEqual(metrics(), before, mode);
    await fill("Timeline frame", 2);
    assert.ok(document.querySelector("svg"));
    assert.doesNotMatch(metrics(), /NaN|Infinity/);
  }
  await render(SurfaceLab, { mode: "descent" });
  await select("Landscape", "saddle");
  await fill("Learning rate", 1.2);
  assert.match(document.body.textContent, /Stopped before leaving/);
});
test("routing prototypes, metrics, sequential frames and inhibition all update", async () => {
  for (const mode of [
    "need",
    "basins",
    "metrics",
    "sequential",
    "inhibition",
  ]) {
    await render(RoutingLab, { mode });
    const before = metrics();
    await fill("Raw state z x", 1.5);
    assert.notEqual(metrics(), before, mode);
    await fill("Prototype μ y", -2);
    await click("Add expert");
    assert.match(document.body.textContent, /Expert 4/);
    if (mode === "sequential" || mode === "inhibition") {
      await fill("Timeline frame", 1);
      assert.match(document.body.textContent, /Distance after/);
    }
    if (mode === "inhibition") {
      await fill("Inhibition strength", 1.4);
      assert.match(document.body.textContent, /1.4/);
    }
  }
});
test("shape matching accepts desired translation and neighbourhood presets alter output", async () => {
  await render(ShapeLab);
  await click("Translation");
  assert.match(metrics(), /Best matching toy expertTranslation/);
  assert.match(metrics(), /Match error0/);
  await render(ShapeLab, { mode: "neighbourhood" });
  const before = geometry();
  await click("Contraction");
  assert.notEqual(geometry(), before);
});
test("embedding clusters normalize, compare selected samples and discard PC3", async () => {
  await render(RepresentationLab, { mode: "embeddings" });
  const initial = geometry(),
    initialValues = metrics();
  await fill("Cluster spread", 0.6);
  assert.notEqual(geometry(), initial);
  assert.notEqual(metrics(), initialValues);
  await fill("Reference sample", 20);
  assert.match(metrics(), /c1-20/);
  await fill("Timeline frame", 40);
  assert.match(metrics(), /Vector lengths1 \/ 1/);
  await select("Embedding display", "pca");
  assert.match(document.body.textContent, /PC3 discarded/);
  await click("Reset embeddings");
  assert.equal(metrics(), initialValues);
  assert.equal(geometry(), initial);
});
test("intrinsic dimension has distinct line, sheet, curved and volume spectra", async () => {
  await render(RepresentationLab, { mode: "intrinsic" });
  assert.match(metrics(), /Noiseless generating dimensions2/);
  const sheet = geometry();
  await select("Structure", "line");
  assert.match(metrics(), /Noiseless generating dimensions1/);
  assert.match(metrics(), /Global linear effective dimension1/);
  assert.notEqual(geometry(), sheet);
  await select("Structure", "curved");
  assert.match(metrics(), /Noiseless generating dimensions2/);
  assert.notEqual(geometry(), sheet);
  assert.equal(document.querySelectorAll("meter").length, 3);
  await select("Structure", "volume");
  assert.match(metrics(), /Noiseless generating dimensions3/);
  await fill("Timeline frame", 0);
  assert.match(metrics(), /Noiseless generating dimensions1/);
  await click("Reset dimension example");
  assert.equal(geometry(), sheet);
});
test("neighbourhood query, k, metric and projection recompute actual membership", async () => {
  await render(RepresentationLab, { mode: "neighbours" });
  const initial = metrics(),
    initialGeometry = geometry();
  await fill("Query q x", 1.4);
  assert.notEqual(metrics(), initial);
  assert.notEqual(geometry(), initialGeometry);
  await fill("Nearest neighbours k", 12);
  assert.equal(
    document.querySelectorAll(".math-neighbour-ranking li").length,
    12,
  );
  await select("Neighbour search space", "pca");
  assert.match(document.body.textContent, /PC3 discarded/);
  assert.doesNotMatch(metrics(), /Neighbour overlap with original 3D12 \/ 12/);
  await select("Neighbour search space", "raw");
  await select("Neighbour metric", "cosine");
  await fill("Query q x", 0);
  await fill("Query q y", 0);
  await fill("Query q z", 0);
  assert.match(metrics(), /Undefined for a zero cosine query/);
  assert.doesNotMatch(metrics(), /NaN|Infinity/);
  assert.equal(
    document.querySelectorAll(".math-neighbour-ranking li").length,
    0,
  );
  await click("Reset neighbourhood");
  assert.equal(metrics(), initial);
});
test("representation entries cannot regress to title-only aliases", async () => {
  const equations = [],
    scenes = [],
    controls = [];
  for (const mode of ["embeddings", "intrinsic", "neighbours"]) {
    await render(RepresentationLab, { mode });
    equations.push(document.querySelector(".math-equation").textContent);
    scenes.push(geometry());
    controls.push(
      [...document.querySelectorAll(".math-controls input")]
        .map((n) => n.getAttribute("aria-label"))
        .join(),
    );
  }
  assert.equal(new Set(equations).size, 3);
  assert.equal(new Set(scenes).size, 3);
  assert.equal(new Set(controls).size, 3);
});
test("3D latent frames propagate manual deltas and renderer fallback is explicit", async () => {
  await render(LatentLab3D);
  await fill("Selected delta z", 2);
  await click("Next frame");
  assert.match(metrics(), /Current state\(-1, 0, 2\)/);
  await click("Add step");
  assert.match(document.body.textContent, /frame 4 \/ 4/);
  assert.match(document.body.textContent, /WebGL2 unavailable/);
  await click("2D");
  assert.doesNotMatch(document.body.textContent, /WebGL2 unavailable/);
  await click("3D");
  assert.match(document.body.textContent, /WebGL2 unavailable/);
});
