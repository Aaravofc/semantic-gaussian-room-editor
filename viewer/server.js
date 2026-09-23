const fs = require("fs");
const http = require("http");
const path = require("path");
const { URL } = require("url");

const projectRoot = path.resolve(__dirname, "..");
const viewerRoot = __dirname;
const options = parseArgs(process.argv.slice(2));
const port = Number(process.env.PORT || options.port || 5177);
const host = process.env.HOST || options.host || "127.0.0.1";
const dataRoot = path.resolve(options.dataRoot || options.projectRoot || projectRoot);
const scan = options.scan || "scan_001";
const fileConfig = buildFileConfig();
const editStatePath = path.resolve(
  options.output || path.join(dataRoot, "scans", scan, "scene_edit_state.json")
);
let fallbackSemanticScene = null;

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".bin": "application/octet-stream",
  ".ply": "application/octet-stream",
  ".wasm": "application/wasm"
};

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (req.method === "GET" && url.pathname === "/api/config") {
      sendJson(res, {
        data_root: dataRoot,
        scan,
        generated_fallbacks: {
          semantic_proxies_from_ply: Boolean((fileConfig.boxesPly.found || fileConfig.semanticPly.found) && !fileConfig.boxes.found),
          source: getFallbackPlySource()?.path || null
        },
        files: Object.fromEntries(
          Object.entries(fileConfig).map(([key, config]) => [
            key,
            {
              url: config.url,
              found: config.found,
              path: config.path,
              candidates: config.candidates
            }
          ])
        )
      });
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/edit-state") {
      if (!fs.existsSync(editStatePath)) {
        sendJson(res, { found: false, path: editStatePath, state: null });
        return;
      }
      const saved = JSON.parse(fs.readFileSync(editStatePath, "utf8"));
      sendJson(res, { found: true, path: editStatePath, state: saved });
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/save-edit-state") {
      const body = await readBody(req);
      const parsed = JSON.parse(body);
      fs.mkdirSync(path.dirname(editStatePath), { recursive: true });
      fs.writeFileSync(editStatePath, JSON.stringify(parsed, null, 2) + "\n");
      sendJson(res, { ok: true, path: editStatePath });
      return;
    }

    if (!["GET", "HEAD"].includes(req.method)) {
      res.writeHead(405);
      res.end("Method not allowed");
      return;
    }

    const filePath = resolveRequestPath(url.pathname);
    const generatedJson = maybeGenerateJson(url.pathname);
    if (generatedJson) {
      sendJson(res, generatedJson);
      return;
    }

    if (!filePath) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.writeHead(404);
      res.end(`File not found: ${url.pathname}`);
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": mimeTypes[ext] || "application/octet-stream",
      "Cache-Control": "no-store",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    fs.createReadStream(filePath).pipe(res);
  } catch (error) {
    res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(error.stack || String(error));
  }
});

server.listen(port, host, () => {
  console.log(`Semantic Gaussian editor: http://${host}:${port}/viewer/`);
  console.log(`Listening on: ${host}:${port}`);
  console.log(`Serving project root: ${projectRoot}`);
  console.log(`Serving data root: ${dataRoot}`);
  console.log(`Scan: ${scan}`);
  console.log(`Edit state output: ${editStatePath}`);
  for (const [key, config] of Object.entries(fileConfig)) {
    const status = config.found ? "found" : "missing";
    console.log(`${key}: ${status} ${config.path || config.candidates.join(" | ")}`);
  }
  if (fileConfig.semanticPly.found && !fileConfig.boxes.found) {
    console.log(`fallback: semantic proxy boxes will be generated from ${getFallbackPlySource().key}`);
  }
});

function resolveRequestPath(requestPath) {
  const normalized = decodeURIComponent(requestPath.split("?")[0]);
  const relative = normalized === "/" ? "/viewer/index.html" : normalized;

  if (relative.startsWith("/data/")) {
    const registered = Object.values(fileConfig).find((config) => config.url === relative);
    if (registered) return registered.path || registered.candidates[0];
    const strippedDataPath = relative.replace(/^\/data\//, "");
    const dataPath = path.resolve(dataRoot, strippedDataPath);
    return isWithin(dataPath, dataRoot) ? dataPath : null;
  }

  const base = relative.startsWith("/viewer/") ? viewerRoot : projectRoot;
  const stripped = relative.startsWith("/viewer/")
    ? relative.replace(/^\/viewer\//, "")
    : relative.replace(/^\//, "");
  const filePath = path.resolve(base, stripped || "index.html");
  return isWithin(filePath, base) ? filePath : null;
}

function parseArgs(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (!arg.startsWith("--")) {
      parsed.port = parsed.port || arg;
      continue;
    }

    const [rawKey, inlineValue] = arg.slice(2).split("=", 2);
    const value = inlineValue ?? args[index + 1];
    if (inlineValue === undefined) index += 1;

    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    parsed[key] = value;
  }
  return parsed;
}

function buildFileConfig() {
  return {
    splat: resolveConfiguredFile("splat", [
      path.join(dataRoot, "splat_labeled.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "splat_labeled.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "splat_labeled.ply"),
      path.join(dataRoot, "scans", scan, "splat_labeled.ply"),
      path.join(dataRoot, "splat.ply"),
      path.join(dataRoot, "scans", scan, "splat.ply"),
      path.join(dataRoot, "scans", scan, "gaussians", "splat.ply"),
      path.join(dataRoot, "scans", scan, "gaussians", "model.ply")
    ], "/data/splat.ply"),
    manifest: resolveConfiguredFile("manifest", [
      path.join(dataRoot, "semantic_scene_manifest.json"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "semantic_scene_manifest.json"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "semantic_scene_manifest.json"),
      path.join(dataRoot, "scans", scan, "semantic_scene_manifest.json")
    ], "/data/semantic_scene_manifest.json"),
    boxes: resolveConfiguredFile("boxes", [
      path.join(dataRoot, "object_bounding_boxes.json"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "object_bounding_boxes.json"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "object_bounding_boxes.json"),
      path.join(dataRoot, "scans", scan, "object_bounding_boxes.json")
    ], "/data/object_bounding_boxes.json"),
    boxesPly: resolveConfiguredFile("boxesPly", [
      path.join(dataRoot, "object_bounding_boxes.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "object_bounding_boxes.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "object_bounding_boxes.ply"),
      path.join(dataRoot, "scans", scan, "object_bounding_boxes.ply")
    ], "/data/object_bounding_boxes.ply"),
    semanticPly: resolveConfiguredFile("semanticPly", [
      path.join(dataRoot, "combined_objects_preview.ply"),
      path.join(dataRoot, "combined_objects_semantic_colored.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "combined_objects_preview.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "projection", "combined_objects_semantic_colored.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "combined_objects_preview.ply"),
      path.join(dataRoot, "scans", scan, "semantic_recon", "outputs", "combined_objects_semantic_colored.ply"),
      path.join(dataRoot, "scans", scan, "combined_objects_preview.ply"),
      path.join(dataRoot, "scans", scan, "combined_objects_semantic_colored.ply")
    ], "/data/combined_objects_semantic_colored.ply")
  };
}

function resolveConfiguredFile(key, candidates, url) {
  const explicit = options[key] ? [path.resolve(options[key])] : [];
  const allCandidates = [...explicit, ...candidates];
  const foundPath = allCandidates.find((candidate) => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
  return {
    url,
    found: Boolean(foundPath),
    path: foundPath || null,
    candidates: allCandidates
  };
}

function isWithin(filePath, rootPath) {
  const relative = path.relative(rootPath, filePath);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function maybeGenerateJson(requestPath) {
  if (fileConfig.boxes.found) return null;
  if (!["/data/object_bounding_boxes.json", "/data/semantic_scene_manifest.json"].includes(requestPath)) return null;

  const scene = getFallbackSemanticScene();
  if (!scene) return null;

  if (requestPath === "/data/object_bounding_boxes.json") {
    return {
      schema: "generated_color_group_bounding_boxes.v1",
      source_ply: scene.source_ply,
      generated_from: "semantic_ply_color_groups",
      objects: scene.objects
    };
  }

  return {
    schema: "generated_semantic_scene_manifest.v1",
    source_ply: scene.source_ply,
    generated_from: "semantic_ply_color_groups",
    objects: scene.objects.map((object) => ({
      object_id: object.object_id,
      label: object.label,
      confidence: object.confidence,
      point_count: object.point_count,
      color_rgb: object.color_rgb
    }))
  };
}

function getFallbackSemanticScene() {
  if (fallbackSemanticScene) return fallbackSemanticScene;
  const source = getFallbackPlySource();
  if (!source) return null;

  const minPoints = Number(options.minProxyPoints || 100);
  const groups = readPlyColorGroups(source.path);
  const objects = [];

  for (const group of groups) {
    if (group.count < minPoints) continue;

    const center = [
      (group.min[0] + group.max[0]) / 2,
      (group.min[1] + group.max[1]) / 2,
      (group.min[2] + group.max[2]) / 2
    ];
    const size = [
      Math.max(group.max[0] - group.min[0], 0.001),
      Math.max(group.max[1] - group.min[1], 0.001),
      Math.max(group.max[2] - group.min[2], 0.001)
    ];
    const colorText = group.color.join("_");

    objects.push({
      object_id: `color_${colorText}`,
      id: `color_${colorText}`,
      label: `semantic_${colorText}`,
      confidence: 1,
      support_views: "generated_from_ply",
      point_count: group.count,
      center,
      size,
      min: group.min,
      max: group.max,
      color_rgb: group.color
    });
  }

  objects.sort((a, b) => b.point_count - a.point_count);
  fallbackSemanticScene = {
    source_ply: source.path,
    source_kind: source.key,
    objects
  };
  console.log(`fallback: generated ${objects.length} proxy boxes from ${source.path}`);
  return fallbackSemanticScene;
}

function getFallbackPlySource() {
  if (fileConfig.boxesPly.found) {
    return { key: "object_bounding_boxes.ply", path: fileConfig.boxesPly.path };
  }
  if (fileConfig.semanticPly.found) {
    return { key: "semantic PLY color groups", path: fileConfig.semanticPly.path };
  }
  return null;
}

function readPlyColorGroups(plyPath) {
  const buffer = fs.readFileSync(plyPath);
  const headerEndMarker = Buffer.from("end_header");
  const headerEnd = buffer.indexOf(headerEndMarker);
  if (headerEnd === -1) throw new Error(`PLY header not found: ${plyPath}`);

  let dataStart = headerEnd + headerEndMarker.length;
  if (buffer[dataStart] === 13) dataStart += 1;
  if (buffer[dataStart] === 10) dataStart += 1;

  const header = buffer.slice(0, headerEnd).toString("utf8");
  const formatMatch = header.match(/^format\s+(\S+)/m);
  const vertexMatch = header.match(/^element\s+vertex\s+(\d+)/m);
  if (!formatMatch || !vertexMatch) throw new Error(`Unsupported PLY header: ${plyPath}`);

  const format = formatMatch[1];
  const vertexCount = Number(vertexMatch[1]);

  if (format === "ascii") {
    return readAsciiPlyGroups(buffer.slice(dataStart).toString("utf8"));
  }

  if (format === "binary_little_endian") {
    return readBinaryPlyGroups(buffer, dataStart, vertexCount);
  }

  throw new Error(`Unsupported PLY format ${format}: ${plyPath}`);
}

function readAsciiPlyGroups(data) {
  const groups = new Map();
  for (const line of data.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 6) continue;
    addPointToGroup(groups, Number(parts[0]), Number(parts[1]), Number(parts[2]), Number(parts[3]), Number(parts[4]), Number(parts[5]));
  }
  return Array.from(groups.values());
}

function readBinaryPlyGroups(buffer, dataStart, vertexCount) {
  const groups = new Map();
  const stride = 15;
  for (let index = 0; index < vertexCount; index += 1) {
    const offset = dataStart + index * stride;
    if (offset + stride > buffer.length) break;
    addPointToGroup(
      groups,
      buffer.readFloatLE(offset),
      buffer.readFloatLE(offset + 4),
      buffer.readFloatLE(offset + 8),
      buffer.readUInt8(offset + 12),
      buffer.readUInt8(offset + 13),
      buffer.readUInt8(offset + 14)
    );
  }
  return Array.from(groups.values());
}

function addPointToGroup(groups, x, y, z, red, green, blue) {
  if (![x, y, z, red, green, blue].every(Number.isFinite)) return;
  const key = `${red},${green},${blue}`;
  let group = groups.get(key);
  if (!group) {
    group = {
      color: [red, green, blue],
      count: 0,
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity]
    };
    groups.set(key, group);
  }

  group.count += 1;
  group.min[0] = Math.min(group.min[0], x);
  group.min[1] = Math.min(group.min[1], y);
  group.min[2] = Math.min(group.min[2], z);
  group.max[0] = Math.max(group.max[0], x);
  group.max[1] = Math.max(group.max[1], y);
  group.max[2] = Math.max(group.max[2], z);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 10 * 1024 * 1024) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function sendJson(res, payload) {
  res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}
