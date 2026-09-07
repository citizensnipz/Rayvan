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
test("representations preserve sample identity while changing projection and structure", async () => {
  for (const mode of ["intrinsic", "embeddings", "neighbours"]) {
    await render(RepresentationLab, { mode });
    const before = metrics();
    await select("Structure", "line");
    assert.notEqual(metrics(), before);
    await select("Projection method", "pca");
    await fill("Timeline frame", 20);
    assert.match(document.body.textContent, /sample 20/);
    assert.ok(document.querySelector("option[disabled]"));
  }
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
