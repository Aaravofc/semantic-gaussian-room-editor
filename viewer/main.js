import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const DEFAULT_PATHS = {
  splat: "/data/splat.ply",
  manifest: "/data/semantic_scene_manifest.json",
  boxes: "/data/object_bounding_boxes.json",
  semanticPly: "/data/combined_objects_semantic_colored.ply"
};

const LABEL_COLORS = [
  "#e45756", "#4c78a8", "#f2cf5b", "#54a24b", "#b279a2", "#ff9da6",
  "#9d755d", "#72b7b2", "#bab0ac", "#f58518", "#5f9ed1", "#59a14f"
];

const qs = new URLSearchParams(window.location.search);
let paths = { ...DEFAULT_PATHS };

const els = {
  canvas: document.querySelector("#overlay-canvas"),
  status: document.querySelector("#status"),
  showSplat: document.querySelector("#show-splat"),
  showObjects: document.querySelector("#show-objects"),
  showSemanticPoints: document.querySelector("#show-semantic-points"),
  showBoxes: document.querySelector("#show-boxes"),
  showLabels: document.querySelector("#show-labels"),
  labelFilter: document.querySelector("#label-filter"),
  objectList: document.querySelector("#object-list"),
  metadata: document.querySelector("#metadata"),
  resetObject: document.querySelector("#reset-object"),
  removeObject: document.querySelector("#remove-object"),
  resetSplat: document.querySelector("#reset-splat"),
  saveEdits: document.querySelector("#save-edits"),
  modeButtons: Array.from(document.querySelectorAll(".mode-button")),
  targetButtons: Array.from(document.querySelectorAll(".target-button"))
};

const state = {
  objects: [],
  selectedId: null,
  transformTarget: "object",
  sceneManifest: null,
  sourceMode: null,
  labelColors: new Map(),
  edits: new Map(),
  removedIds: new Set(),
  pickables: [],
  semanticPointCloud: null,
  semanticPointsLoading: false,
  splatEdit: null,
  messages: []
};

const view = createScene();
init().catch((error) => {
  addMessage(`Viewer startup failed: ${error.message}`);
  updateStatus();
  console.error(error);
});

async function init() {
  setStatus("Loading room scene...");
  setupEvents();
  resize();
  window.addEventListener("resize", resize);

  const serverConfig = await fetchServerConfig();
  paths = {
    splat: qs.get("splat") || serverConfig?.files?.splat?.url || DEFAULT_PATHS.splat,
    manifest: qs.get("manifest") || serverConfig?.files?.manifest?.url || DEFAULT_PATHS.manifest,
    boxes: qs.get("boxes") || serverConfig?.files?.boxes?.url || DEFAULT_PATHS.boxes,
    semanticPly: qs.get("semanticPly") || serverConfig?.files?.semanticPly?.url || DEFAULT_PATHS.semanticPly
  };

  await loadGaussianProxies(serverConfig);
  addMessage("Gaussian-centric mode: the room splat is canonical; semantic geometry is an editable proxy overlay.");

  buildObjectControls();
  if (els.showSemanticPoints.checked) await loadSemanticPointCloud();
  if (els.showSplat.checked) await loadGaussianSplat(serverConfig);
  await loadSavedEditState();
  frameLoadedScene();
  updateVisibility();
  animate();
  updateStatus();
}

function createScene() {
  const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0d0f10, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(58, 1, 0.01, 2000);
  camera.position.set(2.8, 2.2, 4.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.8, 0);
  controls.enableDamping = true;

  const transform = new TransformControls(camera, renderer.domElement);
  transform.setMode("translate");
  transform.addEventListener("dragging-changed", (event) => {
    controls.enabled = !event.value;
  });
  transform.addEventListener("objectChange", captureActiveEdit);
  scene.add(transform);

  const objectRoot = new THREE.Group();
  objectRoot.name = "semantic-proxies";
  const baselineRoot = new THREE.Group();
  baselineRoot.name = "semantic-points";
  scene.add(objectRoot, baselineRoot);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x282826, 1.45));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
  keyLight.position.set(4, 8, 5);
  scene.add(keyLight);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  renderer.domElement.addEventListener("pointerdown", (event) => {
    if (transform.dragging || state.transformTarget !== "object") return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(state.pickables, true)[0];
    if (!hit) return;
    let target = hit.object;
    while (target && !target.userData.objectId) target = target.parent;
    if (target?.userData.objectId) selectObject(target.userData.objectId);
  });

  return { renderer, scene, camera, controls, transform, objectRoot, baselineRoot };
}

function decorateObject(group, object) {
  const geometry = new THREE.BoxGeometry(
    Math.max(object.size[0], 0.001),
    Math.max(object.size[1], 0.001),
    Math.max(object.size[2], 0.001)
  );
  const edges = new THREE.EdgesGeometry(geometry);
  const box = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color: colorForObject(object), transparent: true, opacity: 0.95, depthTest: false })
  );
  box.position.fromArray(object.boxOffset || [0, 0, 0]);
  box.userData.objectId = object.id;
  group.add(box);
  state.pickables.push(box);
  object.boxObject = box;

  const label = makeLabelSprite(`${object.label} ${object.id}`, colorForObject(object));
  label.position.fromArray(labelOffset(object));
  label.userData.objectId = object.id;
  group.add(label);
  object.labelObject = label;
}

function labelOffset(object) {
  const up = vectorFrom(state.sceneManifest?.world_up) || [0, 1, 0];
  const offset = Math.max(...object.size) * 0.55 + 0.04;
  const origin = object.boxOffset || [0, 0, 0];
  return up.map((value, axis) => origin[axis] + value * offset);
}

async function loadGaussianProxies(serverConfig) {
  const [manifestResult, boxesResult] = await Promise.allSettled([
    fetchJson(paths.manifest),
    fetchJson(paths.boxes)
  ]);
  if (boxesResult.status === "rejected") {
    const checked = serverConfig?.files?.boxes?.candidates?.join("\n") || paths.boxes;
    addMessage(`Semantic object proxies are unavailable. Checked:\n${checked}`);
    return;
  }
  const manifest = manifestResult.status === "fulfilled" ? manifestResult.value : null;
  state.sceneManifest = manifest;
  state.objects = normalizeProxyObjects(boxesResult.value, manifest);
  state.sourceMode = "gaussian_semantic_proxy";
  for (const object of state.objects) {
    const group = new THREE.Group();
    group.name = object.id;
    group.userData.objectId = object.id;
    group.position.fromArray(object.center);
    object.basePosition = [...object.center];
    decorateObject(group, object);
    view.objectRoot.add(group);
    object.threeObject = group;
  }
  addMessage(`Loaded ${state.objects.length} Gaussian semantic proxies from ${paths.boxes}.`);
}

async function loadGaussianSplat(serverConfig) {
  try {
    if (!(await headOrRange(paths.splat))) throw new Error("file not found");
    const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
    const dropInViewer = new GaussianSplats3D.DropInViewer({
      dynamicScene: true,
      sharedMemoryForWorkers: false,
      gpuAcceleratedSort: false
    });
    view.scene.add(dropInViewer);
    await dropInViewer.addSplatScene(paths.splat, { splatAlphaRemovalThreshold: 5, showLoadingUI: false });
    state.splatDropInViewer = dropInViewer;
    state.splatViewer = dropInViewer.viewer;
    state.splatScene = dropInViewer.getSplatScene?.(0);
    if (state.splatScene && !state.splatScene.parent) view.scene.add(state.splatScene);
    addMessage(`Loaded canonical Gaussian splat scene from ${paths.splat}.`);
    if (state.transformTarget === "splat") attachTransformTarget();
  } catch (error) {
    const checked = serverConfig?.files?.splat?.candidates?.join("\n") || paths.splat;
    addMessage(`Gaussian splat scene unavailable: ${error.message}\n${checked}`);
  }
}

async function loadSemanticPointCloud() {
  if (state.semanticPointCloud || state.semanticPointsLoading) return;
  state.semanticPointsLoading = true;
  try {
    if (!(await headOrRange(paths.semanticPly))) throw new Error("file not found");
    const geometry = await new Promise((resolve, reject) => {
      new PLYLoader().load(paths.semanticPly, resolve, undefined, reject);
    });
    const hasColors = Boolean(geometry.getAttribute("color"));
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.025,
        sizeAttenuation: true,
        vertexColors: hasColors,
        color: hasColors ? 0xffffff : 0x89c7ff,
        opacity: 0.72,
        transparent: true,
        depthWrite: false
      })
    );
    view.baselineRoot.add(points);
    state.semanticPointCloud = points;
    updateVisibility();
  } catch (error) {
    addMessage(`Baseline semantic point cloud unavailable: ${error.message}`);
    els.showSemanticPoints.checked = false;
  } finally {
    state.semanticPointsLoading = false;
    updateStatus();
  }
}

function buildObjectControls() {
  const labels = ["All labels", ...new Set(state.objects.map((object) => object.label).sort())];
  els.labelFilter.innerHTML = labels
    .map((label) => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`)
    .join("");
  renderObjectList();
}

function renderObjectList() {
  const filter = els.labelFilter.value || "All labels";
  const objects = state.objects.filter((object) => filter === "All labels" || object.label === filter);
  els.objectList.innerHTML = "";
  for (const object of objects) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = [
      "object-button",
      object.id === state.selectedId ? "active" : "",
      state.removedIds.has(object.id) ? "removed" : ""
    ].filter(Boolean).join(" ");
    button.innerHTML = `
      <span class="swatch" style="background:${colorForObject(object)}"></span>
      <span class="object-name">${escapeHtml(object.label)}</span>
      <span class="object-id">${escapeHtml(object.id)}</span>
    `;
    button.addEventListener("click", () => selectObject(object.id));
    els.objectList.appendChild(button);
  }
  updateVisibility();
}

function selectObject(objectId) {
  state.selectedId = objectId;
  if (state.transformTarget !== "object") setTransformTarget("object");
  attachTransformTarget();
  renderObjectList();
  renderMetadata(selectedObject());
}

function selectedObject() {
  return state.objects.find((object) => object.id === state.selectedId) || null;
}

function renderMetadata(object) {
  if (!object) {
    els.metadata.innerHTML = "<dt>Status</dt><dd>No object selected</dd>";
    return;
  }
  const confidence = object.confidence?.mean_multiview_support_ratio ?? object.confidence;
  const fields = [
    ["Object ID", object.id],
    ["Label", object.label],
    ["Representation", "Gaussian semantic proxy"],
    ["Status", state.removedIds.has(object.id) ? "Removed from scene" : "Active"],
    ["Confidence", formatNullable(confidence)],
    ["Support views", formatNullable(object.supportViews)],
    ["Point count", formatNullable(object.pointCount)],
    ["BBox size", object.size.map(formatNumber).join(" x ")],
    ["Edit scope", "Proxy transform only"]
  ];
  els.metadata.innerHTML = fields
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");
  els.removeObject.textContent = state.removedIds.has(object.id) ? "Restore Object" : "Remove Object";
}

function setupEvents() {
  els.showSplat.addEventListener("change", updateVisibility);
  els.showObjects.addEventListener("change", updateVisibility);
  els.showSemanticPoints.addEventListener("change", async () => {
    if (els.showSemanticPoints.checked && !state.semanticPointCloud) await loadSemanticPointCloud();
    updateVisibility();
  });
  els.showBoxes.addEventListener("change", updateVisibility);
  els.showLabels.addEventListener("change", updateVisibility);
  els.labelFilter.addEventListener("change", renderObjectList);
  els.resetObject.addEventListener("click", resetSelectedObject);
  els.removeObject.addEventListener("click", toggleSelectedObjectRemoval);
  els.resetSplat.addEventListener("click", resetSplatTransform);
  els.saveEdits.addEventListener("click", () => saveEditState().catch((error) => {
    addMessage(`Save failed: ${error.message}`);
    updateStatus();
  }));
  for (const button of els.modeButtons) {
    button.addEventListener("click", () => {
      els.modeButtons.forEach((item) => item.classList.toggle("active", item === button));
      view.transform.setMode(button.dataset.mode);
    });
  }
  for (const button of els.targetButtons) {
    button.addEventListener("click", () => setTransformTarget(button.dataset.target));
  }
}

function setTransformTarget(target) {
  state.transformTarget = target;
  els.targetButtons.forEach((button) => button.classList.toggle("active", button.dataset.target === target));
  attachTransformTarget();
  if (target === "splat") renderSplatMetadata();
  else renderMetadata(selectedObject());
}

function attachTransformTarget() {
  view.transform.detach();
  if (state.transformTarget === "splat") {
    if (state.splatScene) view.transform.attach(state.splatScene);
    return;
  }
  const object = selectedObject();
  if (object?.threeObject && !state.removedIds.has(object.id)) view.transform.attach(object.threeObject);
}

function captureActiveEdit() {
  if (state.transformTarget === "splat") {
    captureSplatEdit();
    renderSplatMetadata();
    return;
  }
  captureObjectEdit(selectedObject());
  renderMetadata(selectedObject());
}

function captureObjectEdit(object) {
  if (!object?.threeObject) return;
  state.edits.set(object.id, {
    object_id: object.id,
    label: object.label,
    removed: state.removedIds.has(object.id),
    pivot_position_world: object.threeObject.position.toArray(),
    translation_delta_world: object.threeObject.position.toArray().map(
      (value, axis) => value - object.basePosition[axis]
    ),
    rotation_euler: [object.threeObject.rotation.x, object.threeObject.rotation.y, object.threeObject.rotation.z],
    quaternion: object.threeObject.quaternion.toArray(),
    scale: object.threeObject.scale.toArray()
  });
}

function captureSplatEdit() {
  if (!state.splatScene) return;
  state.splatScene.updateMatrixWorld(true);
  state.splatDropInViewer?.viewer?.getSplatMesh?.()?.updateTransforms?.();
  state.splatViewer?.forceRenderNextFrame?.();
  state.splatEdit = {
    scene_index: 0,
    translation: state.splatScene.position.toArray(),
    quaternion: state.splatScene.quaternion.toArray(),
    scale: state.splatScene.scale.toArray()
  };
}

function resetSelectedObject() {
  const object = selectedObject();
  if (!object?.threeObject) return;
  object.threeObject.position.fromArray(object.basePosition);
  object.threeObject.rotation.set(0, 0, 0);
  object.threeObject.scale.set(1, 1, 1);
  captureObjectEdit(object);
  attachTransformTarget();
  renderMetadata(object);
}

function toggleSelectedObjectRemoval() {
  const object = selectedObject();
  if (!object) return;
  if (state.removedIds.has(object.id)) state.removedIds.delete(object.id);
  else state.removedIds.add(object.id);
  captureObjectEdit(object);
  attachTransformTarget();
  renderObjectList();
  renderMetadata(object);
}

function resetSplatTransform() {
  if (!state.splatScene) return;
  state.splatScene.position.set(0, 0, 0);
  state.splatScene.rotation.set(0, 0, 0);
  state.splatScene.scale.set(1, 1, 1);
  captureSplatEdit();
  attachTransformTarget();
  renderSplatMetadata();
}

function renderSplatMetadata() {
  if (!state.splatScene) {
    els.metadata.innerHTML = "<dt>Status</dt><dd>Gaussian splat unavailable</dd>";
    return;
  }
  const fields = [
    ["Target", "Full-room Gaussian splat"],
    ["Geometry role", "Canonical appearance and coordinates"],
    ["Source", paths.splat],
    ["Position", state.splatScene.position.toArray().map(formatNumber).join(", ")],
    ["Rotation", [state.splatScene.rotation.x, state.splatScene.rotation.y, state.splatScene.rotation.z].map(formatNumber).join(", ")],
    ["Scale", state.splatScene.scale.toArray().map(formatNumber).join(", ")]
  ];
  els.metadata.innerHTML = fields
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");
}

async function loadSavedEditState() {
  let result;
  try {
    const response = await fetch("/api/edit-state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    result = await response.json();
  } catch (error) {
    addMessage(`Saved edit state unavailable: ${error.message}`);
    return;
  }
  if (!result.found || !result.state) return;

  const saved = result.state;
  if (saved.schema !== "semantic_proxy_edit_state.v3") {
    addMessage(`Ignored edit state with incompatible schema ${saved.schema || "unknown"}.`);
    return;
  }

  let applied = 0;
  for (const edit of saved.object_edits || []) {
    const object = state.objects.find((candidate) => candidate.id === String(edit.object_id));
    if (!object?.threeObject) continue;
    const position = vectorFrom(edit.pivot_position_world);
    const scale = vectorFrom(edit.scale);
    const quaternion = Array.isArray(edit.quaternion)
      ? edit.quaternion.slice(0, 4).map(Number)
      : null;
    if (position) object.threeObject.position.fromArray(position);
    if (scale) object.threeObject.scale.fromArray(scale);
    if (quaternion?.length === 4 && quaternion.every(Number.isFinite)) {
      object.threeObject.quaternion.fromArray(quaternion).normalize();
    }
    if (edit.removed) state.removedIds.add(object.id);
    else state.removedIds.delete(object.id);
    captureObjectEdit(object);
    applied += 1;
  }

  const splat = saved.splat_transform;
  if (state.splatScene && splat) {
    const translation = vectorFrom(splat.translation);
    const scale = vectorFrom(splat.scale);
    const quaternion = Array.isArray(splat.quaternion)
      ? splat.quaternion.slice(0, 4).map(Number)
      : null;
    if (translation) state.splatScene.position.fromArray(translation);
    if (scale) state.splatScene.scale.fromArray(scale);
    if (quaternion?.length === 4 && quaternion.every(Number.isFinite)) {
      state.splatScene.quaternion.fromArray(quaternion).normalize();
    }
    captureSplatEdit();
  }

  renderObjectList();
  renderMetadata(selectedObject());
  addMessage(`Restored ${applied} object edit(s) from ${result.path}.`);
}
async function saveEditState() {
  for (const object of state.objects) captureObjectEdit(object);
  captureSplatEdit();
  const payload = {
    schema: "semantic_proxy_edit_state.v3",
    saved_at: new Date().toISOString(),
    source_mode: state.sourceMode,
    source_files: paths,
    object_edits: Array.from(state.edits.values()),
    splat_transform: state.splatEdit
  };
  const response = await fetch("/api/save-edit-state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload, null, 2)
  });
  if (!response.ok) throw new Error(await response.text());
  const result = await response.json();
  addMessage(`Saved ${payload.object_edits.length} object state(s) to ${result.path}.`);
  updateStatus();
}

function updateVisibility() {
  const filter = els.labelFilter.value || "All labels";
  if (state.splatDropInViewer) state.splatDropInViewer.visible = els.showSplat.checked;
  if (state.semanticPointCloud) state.semanticPointCloud.visible = els.showSemanticPoints.checked;
  for (const object of state.objects) {
    const filtered = filter === "All labels" || object.label === filter;
    const removed = state.removedIds.has(object.id);
    if (object.threeObject) object.threeObject.visible = els.showObjects.checked && filtered && !removed;
    if (object.boxObject) object.boxObject.visible = els.showBoxes.checked;
    if (object.labelObject) object.labelObject.visible = els.showLabels.checked;
  }
}

function frameLoadedScene() {
  const bounds = new THREE.Box3();
  bounds.expandByObject(view.objectRoot);
  bounds.expandByObject(view.baselineRoot);
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  view.controls.target.copy(center);
  view.camera.position.copy(center).add(new THREE.Vector3(radius * 0.9, radius * 0.65, radius * 1.15));
  view.camera.near = Math.max(radius / 10000, 0.001);
  view.camera.far = Math.max(radius * 100, 100);
  view.camera.updateProjectionMatrix();
  view.controls.update();
}

function normalizeProxyObjects(boxesData, manifestData) {
  const manifestById = buildManifestMap(manifestData);
  const rawObjects = Array.isArray(boxesData)
    ? boxesData
    : boxesData.objects || boxesData.bounding_boxes || boxesData.boxes || Object.values(boxesData);
  return rawObjects.map((raw, index) => normalizeProxyObject(raw, manifestById, index)).filter(Boolean);
}

function normalizeProxyObject(raw, manifestById, index) {
  if (!raw || typeof raw !== "object") return null;
  const id = String(raw.object_id ?? raw.id ?? raw.instance_id ?? raw.name ?? `object_${index + 1}`);
  const meta = manifestById.get(id) || {};
  const label = String(raw.label ?? raw.class ?? raw.class_name ?? raw.semantic_label ?? meta.label ?? "unknown");
  let center = vectorFrom(raw.center ?? raw.centroid ?? raw.position ?? raw.bbox_center);
  let size = vectorFrom(raw.size ?? raw.dimensions ?? raw.extents ?? raw.bbox_size);
  const min = vectorFrom(raw.min ?? raw.bbox?.min);
  const max = vectorFrom(raw.max ?? raw.bbox?.max);
  if ((!center || !size) && min && max) {
    center = min.map((value, axis) => (value + max[axis]) / 2);
    size = min.map((value, axis) => max[axis] - value);
  }
  if (!center || !size) return null;
  return {
    id,
    label,
    center,
    size: size.map((value) => Math.abs(value)),
    confidence: raw.confidence ?? meta.confidence,
    supportViews: raw.support_views ?? meta.support_views,
    pointCount: raw.point_count ?? meta.point_count,
    color: colorFromRaw(raw.color_rgb ?? raw.color ?? meta.color_rgb),
    source: "proxy",
    raw
  };
}

function buildManifestMap(data) {
  const map = new Map();
  if (!data) return map;
  const objects = Array.isArray(data) ? data : data.objects || data.instances || Object.values(data);
  if (!Array.isArray(objects)) return map;
  for (const object of objects) {
    const id = String(object?.object_id ?? object?.id ?? object?.instance_id ?? "");
    if (id) map.set(id, object);
  }
  return map;
}

async function fetchServerConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    return response.json();
  } catch (error) {
    addMessage(`Server config unavailable; using default paths. ${error.message}`);
    return null;
  }
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
  return response.json();
}

async function headOrRange(path) {
  let response = await fetch(path, { method: "HEAD", cache: "no-store" });
  if (response.ok) return true;
  response = await fetch(path, { headers: { Range: "bytes=0-0" }, cache: "no-store" });
  return response.ok || response.status === 206;
}

function vectorFrom(value) {
  if (Array.isArray(value) && value.length >= 3) return value.slice(0, 3).map(Number);
  if (value && typeof value === "object") {
    const result = [value.x ?? value[0], value.y ?? value[1], value.z ?? value[2]].map(Number);
    if (result.every(Number.isFinite)) return result;
  }
  return null;
}

function colorForLabel(label) {
  if (!state.labelColors.has(label)) {
    state.labelColors.set(label, LABEL_COLORS[state.labelColors.size % LABEL_COLORS.length]);
  }
  return state.labelColors.get(label);
}

function colorForObject(object) {
  return object?.color || colorForLabel(object?.label || "unknown");
}

function colorFromRaw(value) {
  const vector = vectorFrom(value);
  if (!vector) return null;
  const [red, green, blue] = vector.map((channel) => Math.max(0, Math.min(255, Math.round(channel))));
  return `rgb(${red}, ${green}, ${blue})`;
}

function makeLabelSprite(text, color) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const fontSize = 28;
  context.font = `700 ${fontSize}px Inter, sans-serif`;
  canvas.width = Math.ceil(context.measureText(text).width + 28);
  canvas.height = 48;
  context.font = `700 ${fontSize}px Inter, sans-serif`;
  context.fillStyle = "rgba(17, 19, 20, 0.86)";
  context.beginPath();
  context.roundRect(0, 0, canvas.width, canvas.height, 7);
  context.fill();
  context.fillStyle = color;
  context.fillRect(10, 14, 12, 20);
  context.fillStyle = "#f4f1e8";
  context.fillText(text, 28, 33);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false }));
  sprite.scale.set(canvas.width / 180, canvas.height / 180, 1);
  return sprite;
}

function resize() {
  const rect = els.canvas.parentElement.getBoundingClientRect();
  view.camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
  view.camera.updateProjectionMatrix();
  view.renderer.setSize(rect.width, rect.height, false);
  state.splatViewer?.forceRenderNextFrame?.();
}

function animate() {
  requestAnimationFrame(animate);
  view.controls.update();
  state.splatDropInViewer?.viewer?.update?.(view.renderer, view.camera);
  view.renderer.render(view.scene, view.camera);
}

function addMessage(message) {
  state.messages.push(message);
}

function updateStatus() {
  setStatus(state.messages.slice(-7).join("\n\n"));
}

function setStatus(message) {
  els.status.textContent = message;
}

function formatNullable(value) {
  if (value === undefined || value === null || value === "") return "unknown";
  return typeof value === "number" ? formatNumber(value) : String(value);
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "unknown";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
