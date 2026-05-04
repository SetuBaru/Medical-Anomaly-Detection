function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: ${id}`);
  return el;
}

const modelSelect = $("modelSelect");
const breastPanel = $("breastPanel");
const brainPanel = $("brainPanel");
const resetBtn = $("resetBtn");
const runBreastBtn = $("runBreastBtn");
const runBrainBtn = $("runBrainBtn");
const breastCsvRow = $("breastCsvRow");
const brainImage = $("brainImage");
const resultBox = $("resultBox");

const BREAST_EXPECTED_COLUMNS = 32; // id + diagnosis + 30 features

function setResult(text, kind = "info") {
  const prefix =
    kind === "error" ? "Error\n-----\n" : kind === "ok" ? "OK\n--\n" : "";
  resultBox.textContent = `${prefix}${text}`;
}

function formatPercent(x) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "n/a";
  return `${x.toFixed(2)}%`;
}

function formatProbability01(x) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "n/a";
  return `${(x * 100).toFixed(2)}%`;
}

function tryParseJson(text) {
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch {
    return { ok: false, value: null };
  }
}

function formatBreastResult(data) {
  const label = data?.label ?? (data?.predicted_malignant === 1 ? "malignant" : "benign");
  const prob =
    typeof data?.prob_malignant === "number" ? formatProbability01(data.prob_malignant) : "n/a";
  return `Prediction: ${String(label)}\nProbability (malignant): ${prob}`;
}

function formatBrainResult(data) {
  const label = data?.label ?? "n/a";
  const conf =
    typeof data?.confidence === "number" ? formatPercent(data.confidence) : "n/a";
  return `Prediction: ${String(label)}\nConfidence: ${conf}`;
}

function setLoading(isLoading) {
  runBreastBtn.disabled = isLoading;
  runBrainBtn.disabled = isLoading;
  resetBtn.disabled = isLoading;
  modelSelect.disabled = isLoading;
  runBreastBtn.textContent = isLoading
    ? "Running…"
    : "Run breast cancer prediction";
  runBrainBtn.textContent = isLoading ? "Running… " : "Run brain tumor detection";
}

function setModelUI(value) {
  const isBreast = value === "breast";
  breastPanel.classList.toggle("hidden", !isBreast);
  brainPanel.classList.toggle("hidden", isBreast);
  setResult("Waiting for input…");
}

function parseCsvRow(row) {
  const trimmed = (row ?? "").trim();
  if (!trimmed) return { ok: false, error: "Paste a CSV row first." };

  // Simple CSV splitting (dataset has no quoted commas for these fields).
  const parts = trimmed.split(",").map((s) => s.trim());
  if (parts.length !== BREAST_EXPECTED_COLUMNS) {
    return {
      ok: false,
      error: `Expected ${BREAST_EXPECTED_COLUMNS} comma-separated values (id + diagnosis + 30 features), but got ${parts.length}.`,
    };
  }

  const [id, diagnosis, ...featuresRaw] = parts;
  if (!id) return { ok: false, error: "Missing id (first column)." };
  if (!diagnosis) return { ok: false, error: "Missing diagnosis (second column)." };

  const diagnosisNorm = diagnosis.toUpperCase();
  if (diagnosisNorm !== "M" && diagnosisNorm !== "B") {
    return {
      ok: false,
      error: `Diagnosis must be "M" or "B" (got "${diagnosis}").`,
    };
  }

  const features = [];
  for (let i = 0; i < featuresRaw.length; i++) {
    const val = Number(featuresRaw[i]);
    if (!Number.isFinite(val)) {
      return {
        ok: false,
        error: `Feature #${i + 1} is not a number ("${featuresRaw[i]}").`,
      };
    }
    features.push(val);
  }

  return { ok: true, id, diagnosis: diagnosisNorm, features, raw: trimmed };
}

async function runBreast() {
  const parsed = parseCsvRow(breastCsvRow.value);
  if (!parsed.ok) {
    setResult(parsed.error, "error");
    return;
  }

  setLoading(true);
  setResult("Sending request to POST /api/breast-cancer …");
  try {
    const res = await fetch("/api/breast-cancer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: parsed.id,
        diagnosis: parsed.diagnosis, // optional for server; your Python currently drops it
        features: parsed.features, // 30 floats
        csv_row: parsed.raw, // convenient if you prefer parsing server-side
      }),
    });

    const text = await res.text();
    if (!res.ok) {
      setResult(
        `Server returned HTTP ${res.status}.\n\nResponse:\n${text || "(empty)"}`,
        "error",
      );
      return;
    }
    const parsedJson = tryParseJson(text);
    if (parsedJson.ok) {
      setResult(formatBreastResult(parsedJson.value), "ok");
    } else {
      setResult(text || "(empty response)", "ok");
    }
  } catch (e) {
    setResult(
      `Could not reach backend.\n\nIf you opened this file directly (file://), fetch() will fail.\nRun a local server (or I can add one) so /api/breast-cancer exists.\n\nDetails: ${String(
        e?.message ?? e,
      )}`,
      "error",
    );
  } finally {
    setLoading(false);
  }
}

async function runBrain() {
  const file = brainImage.files?.[0];
  if (!file) {
    setResult("Choose an image file first.", "error");
    return;
  }

  const form = new FormData();
  form.append("image", file);

  setLoading(true);
  setResult("Uploading image to POST /api/brain-tumor …");
  try {
    const res = await fetch("/api/brain-tumor", { method: "POST", body: form });
    const text = await res.text();
    if (!res.ok) {
      setResult(
        `Server returned HTTP ${res.status}.\n\nResponse:\n${text || "(empty)"}`,
        "error",
      );
      return;
    }
    const parsedJson = tryParseJson(text);
    if (parsedJson.ok) {
      setResult(formatBrainResult(parsedJson.value), "ok");
    } else {
      setResult(text || "(empty response)", "ok");
    }
  } catch (e) {
    setResult(
      `Could not reach backend.\n\nRun a local server (or I can add one) so /api/brain-tumor exists.\n\nDetails: ${String(
        e?.message ?? e,
      )}`,
      "error",
    );
  } finally {
    setLoading(false);
  }
}

function resetAll() {
  breastCsvRow.value = "";
  brainImage.value = "";
  setResult("Waiting for input…");
}

modelSelect.addEventListener("change", () => setModelUI(modelSelect.value));
resetBtn.addEventListener("click", resetAll);
runBreastBtn.addEventListener("click", runBreast);
runBrainBtn.addEventListener("click", runBrain);

setModelUI(modelSelect.value);

