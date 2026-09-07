import { test, before, after } from "node:test";
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
const { VectorVisualization } =
  await import("../src/research/math/visualizations/VectorVisualizations.tsx");
const { MatrixTransformation } =
  await import("../src/research/math/visualizations/MatrixTransformation.tsx");
const { GradientVisualization } =
  await import("../src/research/math/visualizations/GradientVisualization.tsx");
const { LatentTransformation } =
  await import("../src/research/math/visualizations/LatentTransformation.tsx");
const { MathCategorySidebar } =
  await import("../src/research/math/components/MathCategorySidebar.tsx");
let root;
before(() => {
  root = createRoot(document.getElementById("root"));
});
after(async () => {
  await act(() => root.unmount());
  dom.window.close();
});
let renderKey = 0;
const render = async (Component, props = {}) =>
  act(() =>
    root.render(React.createElement(Component, { ...props, key: ++renderKey })),
  );
const text = () => document.body.textContent;
const metrics = () => document.querySelector(".math-values").textContent;
async function fill(label, value) {
  const input = [...document.querySelectorAll("input")].find(
    (n) => n.getAttribute("aria-label") === label,
  );
  assert.ok(input, `Missing control ${label}`);
  await act(() => {
    Object.getOwnPropertyDescriptor(
      dom.window.HTMLInputElement.prototype,
      "value",
    ).set.call(input, String(value));
    input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}
async function click(label) {
  const button = [...document.querySelectorAll("button")].find(
    (n) => (n.getAttribute("aria-label") || n.textContent) === label,
  );
  assert.ok(button, `Missing button ${label}`);
  assert.equal(button.disabled, false);
  await act(() =>
    button.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })),
  );
}
test("every vector module updates computed values and SVG from numeric inputs", async () => {
  for (const mode of [
    "vectors",
    "norm",
    "dot",
    "cosine",
    "distance",
    "projection",
  ]) {
    await render(VectorVisualization, { mode });
    const previous = metrics(),
      geometry = document.querySelector("svg").outerHTML;
    await fill(mode === "cosine" ? "u angle °" : "u x", 0);
    assert.notEqual(metrics(), previous, mode);
    assert.notEqual(document.querySelector("svg").outerHTML, geometry, mode);
  }
});
test("cosine zero magnitude and projection zero target render explicit undefined states", async () => {
  await render(VectorVisualization, { mode: "cosine" });
  await fill("u magnitude", 0);
  assert.match(metrics(), /Undefined/);
  assert.doesNotMatch(text(), /NaN|Infinity/);
  await render(VectorVisualization, { mode: "projection" });
  await fill("v x", 0);
  await fill("v y", 0);
  assert.match(metrics(), /Undefined/);
  assert.doesNotMatch(text(), /NaN|Infinity/);
});
test("matrix controls and presets update coordinates, basis and determinant", async () => {
  await render(MatrixTransformation);
  const previous = document.querySelector("svg").outerHTML;
  await fill("a", 2);
  assert.match(metrics(), /det\(A\) = ad − bc2/);
  assert.notEqual(document.querySelector("svg").outerHTML, previous);
  await click("Rotate 90°");
  assert.match(metrics(), /\(0, 1\)/);
  assert.match(metrics(), /\(-1, 0\)/);
  await click("Collapse");
  assert.match(text(), /collapses the plane onto a line/);
});
test("gradient controls update geometry and report an overshooting step", async () => {
  await render(GradientVisualization);
  const previous = metrics();
  await fill("Point x", 3);
  await fill("x² weight a", 2);
  await fill("Learning rate η", 1.2);
  assert.notEqual(metrics(), previous);
  assert.match(text(), /overshoots/);
  await fill("Learning rate η slider", 0.1);
  assert.match(text(), /reduces loss/);
});
test("editable timeline supports snapshots, scrub, previous/next, play/pause and removal", async () => {
  await render(VectorVisualization, { mode: "vectors" });
  await click("Add step");
  await fill("u x", 4);
  await click("Previous frame");
  assert.match(metrics(), /\(3, 2\)/);
  await click("Next frame");
  assert.match(metrics(), /\(4, 2\)/);
  await fill("Timeline frame", 0);
  assert.match(metrics(), /\(3, 2\)/);
  await click("Play");
  assert.match(text(), /Pause/);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 850));
  });
  assert.match(text(), /frame 2 \/ 2/);
  assert.match(text(), /Play/);
  await click("Load example sequence");
  assert.match(text(), /frame 1 \/ 5/);
  await click("Play");
  await click("Pause");
  await click("Remove step");
  assert.match(text(), /frame 1 \/ 4/);
});
test("latent controls propagate downstream and sequence operations stay valid", async () => {
  await render(LatentTransformation);
  await fill("Delta step 0 x", 2);
  await click("Next frame");
  assert.match(metrics(), /z1\(0, 1\)/);
  assert.match(metrics(), /z2\(2, 0.5\)/);
  await fill("Initial z₀ y", 0);
  assert.match(metrics(), /z1\(0, 2\)/);
  await click("Add step");
  assert.match(text(), /frame 4 \/ 4/);
  await click("Remove step");
  assert.match(text(), /frame 3 \/ 3/);
  await click("Load example sequence");
  await fill("Timeline frame", 4);
  assert.match(text(), /frame 5 \/ 5/);
  assert.doesNotMatch(text(), /NaN|Infinity|undefined/);
});
test("unimplemented concepts are clearly marked and disabled", async () => {
  await render(MathCategorySidebar, {
    selected: "vectors",
    onSelect: () => {},
  });
  const disabled = [...document.querySelectorAll("button:disabled")];
  assert.equal(disabled.length, 5);
  assert.ok(disabled.every((n) => n.textContent.includes("Coming later")));
});
