const report = window.REPORT_DATA || null;
let visualMode = "heatmap";
let selectedPredictionIndex = 0;

const fmt = (value) => {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
};

const byId = (id) => document.getElementById(id);
const hasId = (id) => Boolean(byId(id));

const escapeHtml = (value) => String(value ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const pct = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
};

const prototypeLookup = () => new Map((report?.prototypes || []).map((p) => [p.prototype_id, p]));

function voteSummary(p) {
  const votes = p.annotation_votes || [];
  const top = votes[0] || null;
  if (top) {
    const count = top.count ?? top.hits ?? "n/a";
    const share = typeof top.share === "number" ? top.share : p.annotation_confidence;
    return `${top.label || p.annotation_label || "n/a"} - ${pct(share)} - ${count} votes`;
  }

  const evidence = p.annotation_evidence || [];
  const hitCount = evidence.reduce((sum, item) => sum + (item.matched_parts || []).filter((name) => name === p.annotation_label).length, 0);
  return `${p.annotation_label || "n/a"} - ${pct(p.annotation_confidence)} - ${hitCount || "n/a"} hits`;
}

function voteRows(p) {
  const processVotes = p.part_vote_process?.votes || p.gaussian_part_votes || [];
  const exportedVotes = p.annotation_votes || [];
  return (processVotes.length ? processVotes : exportedVotes).map((v) => ({
    label: v.label || v.part || v.part_label || "unknown",
    share: Number(v.share ?? v.vote_share ?? 0),
    weighted: Number(v.weighted_score ?? v.weighted_vote ?? v.vote ?? 0),
    count: v.count ?? v.hits ?? v.image_count ?? "n/a",
  })).sort((a, b) => b.weighted - a.weighted);
}

function partVoteEvidence(p) {
  const processEvidence = p.part_vote_process?.evidence || p.gaussian_vote_evidence || [];
  const legacyEvidence = p.annotation_evidence || [];
  const examples = p.examples || [];
  return (processEvidence.length ? processEvidence : legacyEvidence).map((ev, idx) => ({
    ...ev,
    overlay: ev.overlay || examples[idx]?.overlay,
    original: ev.original || examples[idx]?.original,
    activation: ev.activation || examples[idx]?.activation,
    crop: ev.crop || examples[idx]?.crop,
    image_path: ev.image_path || examples[idx]?.image_path,
    activation_score: ev.activation_score ?? examples[idx]?.activation_score,
  }));
}

function pieGradient(rows) {
  const palette = ["#111111", "#555555", "#888888", "#b5b5b5", "#d4d4d4", "#efefef"];
  const total = rows.reduce((sum, row) => sum + Math.max(0, row.weighted), 0) || 1;
  let cursor = 0;
  const parts = rows.map((row, idx) => {
    const start = cursor;
    cursor += (Math.max(0, row.weighted) / total) * 360;
    return `${palette[idx % palette.length]} ${start.toFixed(2)}deg ${cursor.toFixed(2)}deg`;
  });
  return `conic-gradient(${parts.join(", ")})`;
}

function renderVoteBreakdown(p) {
  const rows = voteRows(p);
  if (!rows.length) return `<div class="detail-panel empty">No part-vote rows were exported for this prototype.</div>`;
  const total = rows.reduce((sum, row) => sum + Math.max(0, row.weighted), 0) || 1;
  const method = p.part_vote_process?.method || p.vote_method || "activation-weighted part voting";
  const cutoff = p.part_vote_process?.activation_cutoff || p.part_vote_process?.percentile || p.vote_percentile || "exported evidence";
  const sigma = p.part_vote_process?.sigma_rule || p.sigma_rule || "activation box diagonal / 2";
  return `
    <section class="vote-breakdown">
      <div>
        <p class="eyebrow">Part-Vote Breakdown</p>
        <h3>${escapeHtml(p.annotation_label || rows[0].label)} wins with ${escapeHtml(pct(rows[0].weighted / total))}</h3>
        <p class="small">Method: <span class="mono">${escapeHtml(method)}</span></p>
        <p class="small">Activation cutoff: <span class="mono">${escapeHtml(fmt(cutoff))}</span>; Gaussian sigma: <span class="mono">${escapeHtml(sigma)}</span></p>
      </div>
      <div class="vote-chart-wrap">
        <div class="vote-pie" style="background:${escapeHtml(pieGradient(rows))}" role="img" aria-label="Weighted part vote pie chart"></div>
        <div class="vote-legend">
          ${rows.map((row, idx) => `<div class="vote-legend-row">
            <span class="legend-swatch swatch-${idx % 6}"></span>
            <strong>${escapeHtml(row.label)}</strong>
            <span class="mono">${escapeHtml(fmt(row.weighted))}</span>
            <span>${escapeHtml(pct(row.weighted / total))}</span>
            <span class="muted">count ${escapeHtml(fmt(row.count))}</span>
          </div>`).join("")}
        </div>
      </div>
    </section>
  `;
}

function renderVoteEvidence(p) {
  const rows = partVoteEvidence(p).slice(0, 8);
  if (!rows.length) return "";
  return `
    <section class="vote-evidence-section">
      <div class="section-heading compact-heading">
        <p class="eyebrow">Voting Evidence</p>
        <h3>How Individual Activations Vote</h3>
      </div>
      <p class="small">Each retained activation has a vote budget scaled by its activation score. The vote is split across visible part landmarks by normalized Gaussian proximity to the activation center.</p>
      <div class="vote-evidence-grid">
        ${rows.map((ev, idx) => {
          const partRows = ev.part_votes || ev.gaussian_votes || (ev.matched_parts || []).map((label) => ({ label, share: null, weight: null }));
          return `<article class="vote-evidence-card">
            ${imageFrame(ev, `${p.prototype_id} vote example ${idx + 1}`)}
            <div class="vote-evidence-body">
              <strong>Example ${idx + 1}</strong>
              <p class="small mono">activation ${escapeHtml(fmt(ev.activation_score))}</p>
              <p class="small path mono">${escapeHtml(ev.image_path || "")}</p>
              <table class="mini-vote-table">
                <thead><tr><th>Part</th><th>Distance</th><th>Gaussian</th><th>Vote</th></tr></thead>
                <tbody>
                  ${partRows.map((row) => `<tr>
                    <td>${escapeHtml(row.label || row.part || "n/a")}</td>
                    <td class="mono">${escapeHtml(fmt(row.distance ?? row.d))}</td>
                    <td class="mono">${escapeHtml(fmt(row.gaussian ?? row.proximity ?? row.r))}</td>
                    <td class="mono">${escapeHtml(fmt(row.weight ?? row.vote ?? row.weighted_vote ?? row.share))}</td>
                  </tr>`).join("") || `<tr><td colspan="4">No per-part vote rows exported.</td></tr>`}
                </tbody>
              </table>
            </div>
          </article>`;
        }).join("")}
      </div>
    </section>
  `;
}

function imageFor(item) {
  if (visualMode === "original") return item.original || item.image || item.crop || item.overlay || "";
  if (visualMode === "activation") return item.activation || item.overlay || item.original || item.image || "";
  return item.overlay || item.image || item.original || "";
}

function prototypeThumbnail(p) {
  if (visualMode === "original") return p.thumbnail_original || p.examples?.[0]?.original || "";
  if (visualMode === "activation") return p.thumbnail_activation || p.examples?.[0]?.activation || "";
  return p.thumbnail_overlay || p.thumbnail || p.examples?.[0]?.overlay || "";
}

function imageModeLabel() {
  if (visualMode === "original") return "original";
  if (visualMode === "activation") return "activation grid";
  return "heatmap + box";
}

function imageFrame(item, alt = "") {
  const img = imageFor(item);
  if (!img) return `<div class="image-missing">visual not saved</div>`;
  const crisp = visualMode === "activation" ? " activation-frame" : "";
  return `<div class="image-frame${crisp}"><img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}"><span class="image-mode-note">${escapeHtml(imageModeLabel())}</span></div>`;
}

function card(label, value, note = "") {
  const text = String(value ?? "");
  const mono = /path|root|checkpoint|tensor|prototype|id|generated/i.test(`${label} ${text}`) ? " mono" : "";
  const path = text.includes("/") || text.includes("\\") ? " path" : "";
  return `<div class="metric-card"><span class="eyebrow">${escapeHtml(label)}</span><span class="metric-value${mono}${path}">${escapeHtml(fmt(value))}</span>${note ? `<p class="small">${escapeHtml(note)}</p>` : ""}</div>`;
}

function initMissingData() {
  if (hasId("summary-grid")) byId("summary-grid").innerHTML = card("Generated Data", "Missing", "Run the Lightning generator and copy its output folders next to this HTML.");
  if (hasId("metrics-grid")) byId("metrics-grid").innerHTML = card("Status", "Not Ready", "Expected data/report_data.js");
}

function initSummary(data) {
  const s = data.model_summary || {};
  if (hasId("run-status")) byId("run-status").innerHTML = `<span class="status-dot"></span><strong>Report data loaded</strong><p><span class="mono">${escapeHtml(fmt(s.generated_at))}</span><br><span class="path mono">${escapeHtml(fmt(s.checkpoint_path))}</span></p>`;
  if (hasId("summary-grid")) byId("summary-grid").innerHTML = [
    card("Backbone", s.backbone),
    card("Classes", s.num_classes),
    card("Prototypes / Class", s.num_prototypes_per_class),
    card("Foreground Prototypes", s.num_foreground_prototypes),
    card("Tensor Shape", s.prototype_tensor_shape),
    card("Dataset Root", s.dataset_root),
    card("Images Scanned", s.images_scanned),
    card("Checkpoint", s.checkpoint_name),
  ].join("");

  const m = data.metrics || {};
  if (hasId("metrics-grid")) byId("metrics-grid").innerHTML = [
    card("Accuracy", m.accuracy),
    card("Consistency", m.consistency),
    card("Stability", m.stability),
    card("Distinctiveness", m.distinctiveness),
    card("Comprehensiveness", m.comprehensiveness),
  ].join("");
}

function unique(arr) {
  return Array.from(new Set(arr.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function populateFilters(prototypes) {
  if (!hasId("class-filter") || !hasId("label-filter")) return;
  const classFilter = byId("class-filter");
  const labelFilter = byId("label-filter");
  for (const name of unique(prototypes.map((p) => p.class_name))) {
    classFilter.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`);
  }
  for (const label of unique(prototypes.map((p) => p.primary_label || p.annotation_label || p.clip_label))) {
    labelFilter.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`);
  }
}

function renderPredictionSelector(predictions) {
  if (!hasId("prediction-selector")) return;
  const items = predictions || [];
  byId("prediction-selector").innerHTML = items.map((p, idx) => `
    <button type="button" class="prediction-pick ${idx === selectedPredictionIndex ? "active" : ""}" data-prediction-index="${idx}">
      <img src="${escapeHtml(p.image || "")}" alt="${escapeHtml(p.true_class || "prediction")}">
      <span>${escapeHtml(p.true_class || "example")}</span>
      <strong>${escapeHtml(p.predicted_class || "n/a")}</strong>
    </button>
  `).join("");
  document.querySelectorAll("[data-prediction-index]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedPredictionIndex = Number(button.dataset.predictionIndex || 0);
      renderPredictionExplorer();
    });
  });
}

function evidenceRowsForPrediction(prediction) {
  if (!hasId("evidence-scope")) return [];
  const scope = byId("evidence-scope").value;
  const all = prediction.prototype_evidence || [];
  if (scope === "predicted") return prediction.predicted_class_evidence || all.filter((item) => item.contributes_to_prediction);
  if (scope === "top_activation") return prediction.top_activation_evidence || [...all].sort((a, b) => b.activation_score - a.activation_score);
  return [...all].sort((a, b) => b.activation_score - a.activation_score);
}

function applyEvidenceLimit(rows) {
  if (!hasId("evidence-limit")) return rows;
  const limit = byId("evidence-limit").value;
  if (limit === "all") return rows;
  return rows.slice(0, Number(limit));
}

function classScoresForPrediction(prediction) {
  const grouped = new Map();
  for (const item of prediction.prototype_evidence || []) {
    const key = item.class_idx;
    const current = grouped.get(key) || {
      class_idx: item.class_idx,
      class_name: item.owner_class || `class ${item.class_idx}`,
      weighted_sum: 0,
      prototype_count: 0,
    };
    current.weighted_sum += Number(item.owner_logit_contribution || 0);
    current.prototype_count += 1;
    grouped.set(key, current);
  }
  return Array.from(grouped.values()).sort((a, b) => b.weighted_sum - a.weighted_sum);
}

function renderClassScoreTable(prediction) {
  if (!hasId("class-score-table")) return;
  const limitValue = hasId("class-score-limit") ? byId("class-score-limit").value : "5";
  const scores = classScoresForPrediction(prediction);
  const limited = limitValue === "all" ? scores : scores.slice(0, Number(limitValue));
  byId("class-score-table").innerHTML = limited.map((row, idx) => {
    const isPred = row.class_idx === prediction.predicted_class_idx;
    const isTrue = row.class_idx === prediction.true_class_idx;
    const marker = [isPred ? "predicted" : "", isTrue ? "true" : ""].filter(Boolean).join(" / ");
    return `<tr class="${isPred ? "predicted-row" : ""}">
      <td class="mono">#${idx + 1}</td>
      <td>${escapeHtml(row.class_name)}<span class="evidence-sub mono">class ${escapeHtml(row.class_idx)}</span></td>
      <td class="mono score-number">${escapeHtml(fmt(row.weighted_sum))}</td>
      <td>${escapeHtml(marker || "-")}</td>
    </tr>`;
  }).join("");
}

function topContributorSummary(prediction) {
  const rows = prediction.predicted_class_evidence || [];
  const top = [...rows].sort((a, b) => b.owner_logit_contribution - a.owner_logit_contribution)[0];
  if (!top) return "No contributor metadata exported.";
  const label = top.annotation_label || top.primary_label || top.clip_label || top.prototype_id;
  return `${top.prototype_id} (${label}) contributes ${fmt(top.owner_logit_contribution)} to the predicted-class logit.`;
}

function topClassSummary(prediction) {
  const scores = classScoresForPrediction(prediction);
  const top = scores[0];
  const second = scores[1];
  if (!top) return "No class-score rows available.";
  if (!second) return `${top.class_name} has weighted sum ${fmt(top.weighted_sum)}.`;
  return `${top.class_name} ranks first with ${fmt(top.weighted_sum)}; next is ${second.class_name} at ${fmt(second.weighted_sum)}.`;
}

function renderInferenceWalkthrough(prediction) {
  if (!hasId("inference-walkthrough")) return;
  const contributorCount = (prediction.predicted_class_evidence || []).length;
  const allCount = (prediction.prototype_evidence || []).length;
  const firstVisual = (prediction.predicted_class_evidence || []).find((item) => item.overlay) || (prediction.top_activation_evidence || []).find((item) => item.overlay) || {};
  byId("inference-walkthrough").innerHTML = `
    <article class="walkthrough-step">
      <span class="step-number">1</span>
      <div>
        <p class="eyebrow">Input</p>
        <h3>Cropped test image</h3>
        ${imageFrame(prediction, prediction.true_class)}
        <p class="small">True class: <strong>${escapeHtml(prediction.true_class || "n/a")}</strong></p>
      </div>
    </article>
    <article class="walkthrough-step">
      <span class="step-number">2</span>
      <div>
        <p class="eyebrow">Patch Encoding</p>
        <h3>DINOv2 patch-token grid</h3>
        <div class="patch-grid-demo" aria-label="16 by 16 patch grid">${Array.from({ length: 256 }, (_, i) => `<span style="--d:${(i % 16) + Math.floor(i / 16)}"></span>`).join("")}</div>
        <p class="small">The image is represented as local patch descriptors rather than one opaque global vector.</p>
      </div>
    </article>
    <article class="walkthrough-step">
      <span class="step-number">3</span>
      <div>
        <p class="eyebrow">Prototype Matching</p>
        <h3>${escapeHtml(allCount)} prototype activations</h3>
        ${imageFrame(firstVisual, firstVisual.prototype_id || "top prototype")}
        <p class="small">Each learned prototype searches the patch grid; the strongest patch match becomes its activation score.</p>
      </div>
    </article>
    <article class="walkthrough-step">
      <span class="step-number">4</span>
      <div>
        <p class="eyebrow">Score Aggregation</p>
        <h3>${escapeHtml(contributorCount)} decision contributors</h3>
        <p class="walkthrough-stat mono">${escapeHtml(topContributorSummary(prediction))}</p>
        <p class="small">For each class, only that class's five prototype scores are weighted and summed.</p>
      </div>
    </article>
    <article class="walkthrough-step">
      <span class="step-number">5</span>
      <div>
        <p class="eyebrow">Prediction</p>
        <h3>${escapeHtml(prediction.predicted_class || "n/a")}</h3>
        <p class="walkthrough-stat mono">${escapeHtml(topClassSummary(prediction))}</p>
        <p class="small">The class with the highest weighted sum is returned as the final prediction.</p>
      </div>
    </article>
  `;
}

function evidenceMetricList(item) {
  return `<div class="evidence-list compact">
    <div class="evidence-row"><span class="evidence-key">Prototype</span><span class="evidence-value mono">${escapeHtml(item.prototype_id)}</span></div>
    <div class="evidence-row"><span class="evidence-key">Owner</span><span class="evidence-value">${escapeHtml(item.owner_class || "n/a")}</span></div>
    <div class="evidence-row"><span class="evidence-key">Part Vote</span><span class="evidence-value">${escapeHtml(item.annotation_label || item.primary_label || "n/a")}<span class="evidence-sub">${escapeHtml(pct(item.annotation_confidence))}</span></span></div>
    <div class="evidence-row"><span class="evidence-key">CLIP</span><span class="evidence-value">${escapeHtml(item.clip_label || "n/a")}</span></div>
    <div class="evidence-row"><span class="evidence-key">Weight</span><span class="evidence-value mono">${escapeHtml(fmt(item.classifier_weight))}</span></div>
    <div class="evidence-row"><span class="evidence-key">Contribution</span><span class="evidence-value mono">${escapeHtml(fmt(item.owner_logit_contribution))}</span></div>
  </div>`;
}

function renderPredictionExplorer() {
  if (!hasId("selected-prediction") || !hasId("local-evidence")) return;
  const predictions = report.predictions || [];
  const p = predictions[selectedPredictionIndex] || predictions[0];
  if (!p) {
    byId("prediction-selector").innerHTML = "";
    if (hasId("selected-prediction")) byId("selected-prediction").innerHTML = `<div class="detail-panel empty">Prediction examples were not generated yet.</div>`;
    if (hasId("inference-walkthrough")) byId("inference-walkthrough").innerHTML = "";
    byId("local-evidence").innerHTML = "";
    return;
  }

  renderPredictionSelector(predictions);
  renderInferenceWalkthrough(p);
  const correctness = p.is_correct === undefined ? "" : (p.is_correct ? "correct" : "incorrect");
  byId("selected-prediction").innerHTML = `
    <article class="selected-image-card">
      ${imageFrame(p, p.true_class)}
      <div>
        <p class="eyebrow">Selected Test Image</p>
        <h3>${escapeHtml(p.predicted_class || "Prediction")}</h3>
        <p class="small">true class: <strong>${escapeHtml(p.true_class || "n/a")}</strong></p>
        <p class="small">confidence: <span class="mono">${escapeHtml(fmt(p.score))}</span> ${correctness ? `- ${escapeHtml(correctness)}` : ""}</p>
        <p class="small path mono">${escapeHtml(p.image_path || "")}</p>
      </div>
    </article>
    <div class="decision-note">
      <strong>Interpretation rule:</strong> the predicted class logit is the weighted sum of that class's five prototype evidence scores.
      Other prototypes can activate on the image, but they contribute to their own owner classes rather than directly to this predicted class.
    </div>
  `;

  renderClassScoreTable(p);
  const rows = applyEvidenceLimit(evidenceRowsForPrediction(p));
  byId("local-evidence").innerHTML = rows.map((item, idx) => `
    <article class="local-evidence-card ${item.contributes_to_prediction ? "contributor" : ""}">
      <div class="evidence-hero">
        <span class="rank-pill">#${idx + 1}</span>
        <div>
          <span class="activation-label">Activation</span>
          <strong class="activation-value mono">${escapeHtml(fmt(item.activation_score))}</strong>
        </div>
      </div>
      ${imageFrame(item, item.prototype_id)}
      <div class="local-evidence-body">
        <h3><span class="prototype-id">${escapeHtml(item.prototype_id)}</span></h3>
        ${item.contributes_to_prediction ? `<p class="contribution-flag">contributes to predicted class</p>` : `<p class="small">activation for another owner class</p>`}
        ${evidenceMetricList(item)}
      </div>
    </article>
  `).join("") || `<div class="detail-panel empty">No evidence rows were exported for this image.</div>`;
}

function renderPrototypes(prototypes) {
  if (!hasId("prototype-grid")) return;
  const q = byId("search").value.trim().toLowerCase();
  const classValue = byId("class-filter").value;
  const labelValue = byId("label-filter").value;

  const filtered = prototypes.filter((p) => {
    const mainLabel = p.primary_label || p.annotation_label || p.clip_label || "";
    const haystack = `${p.prototype_id} ${p.class_name} ${mainLabel} ${p.annotation_label} ${p.clip_label} ${p.clip_candidates?.join(" ")}`.toLowerCase();
    return (!q || haystack.includes(q)) && (!classValue || p.class_name === classValue) && (!labelValue || mainLabel === labelValue);
  });

  byId("prototype-grid").innerHTML = filtered.map((p) => {
    const img = prototypeThumbnail(p);
    const mainLabel = p.primary_label || p.annotation_label || p.clip_label || "unlabeled";
    return `<article class="prototype-card" data-prototype="${escapeHtml(p.prototype_id)}">
      ${img ? imageFrame({ overlay: img, original: img, activation: img }, p.prototype_id) : ""}
      <div class="prototype-card-body">
        <h3 class="prototype-id">${escapeHtml(p.prototype_id)}</h3>
        <p class="small">${escapeHtml(p.class_name)} - slot <span class="mono">${escapeHtml(p.part_idx)}</span></p>
        <div class="evidence-list">
          <div class="evidence-row"><span class="evidence-key">Primary</span><span class="evidence-value">${escapeHtml(mainLabel)}</span></div>
          <div class="evidence-row"><span class="evidence-key">Part Vote</span><span class="evidence-value">${escapeHtml(p.annotation_label || "n/a")}<span class="evidence-sub">${escapeHtml(voteSummary(p))}</span></span></div>
          <div class="evidence-row"><span class="evidence-key">CLIP</span><span class="evidence-value">${escapeHtml(p.clip_label || "n/a")}<span class="evidence-sub">score ${escapeHtml(fmt(p.clip_score))}</span></span></div>
        </div>
      </div>
    </article>`;
  }).join("");

  document.querySelectorAll(".prototype-card").forEach((el) => {
    el.addEventListener("click", () => showPrototype(el.dataset.prototype));
  });
}

function showPrototype(id) {
  if (!hasId("prototype-detail")) return;
  const lookup = prototypeLookup();
  const p = lookup.get(id);
  if (!p) return;
  const mainLabel = p.primary_label || p.annotation_label || p.clip_label || "unlabeled";
  const examples = (p.examples || []).map((ex, idx) => `<article class="example-card">
    ${imageFrame(ex, `${p.prototype_id} example ${idx + 1}`)}
    <div>
      <strong>Top ${idx + 1}</strong>
      <p class="small path mono">${escapeHtml(ex.image_path || "")}</p>
      <p class="small">activation ${escapeHtml(fmt(ex.activation_score))}</p>
      ${ex.crop ? `<a href="${escapeHtml(ex.crop)}">activation crop</a>` : ""}
    </div>
  </article>`).join("");

  byId("prototype-detail").classList.remove("empty");
  byId("prototype-detail").innerHTML = `
    <p class="eyebrow">${escapeHtml(p.class_name)}</p>
    <h2><span class="prototype-id">${escapeHtml(p.prototype_id)}</span> - ${escapeHtml(mainLabel)}</h2>
    <p class="small">Prototype slot <span class="mono">${escapeHtml(p.part_idx)}</span>; annotation vote: <strong>${escapeHtml(voteSummary(p))}</strong>; CLIP: <strong>${escapeHtml(p.clip_label || "n/a")}</strong> (${escapeHtml(fmt(p.clip_score))})</p>
    <p class="small">CLIP candidates: ${escapeHtml((p.clip_candidates || []).join(", ") || "n/a")}</p>
    <div class="method-callout">
      <strong>Voting rule:</strong>
      scan owner-class images, keep high-activation examples, compute distances from the activation center to visible part landmarks,
      convert distances to Gaussian scores with <span class="mono">sigma = activation box diagonal / 2</span>,
      normalize them, then add <span class="mono">activation x normalized Gaussian weight</span> to each part.
    </div>
    ${renderVoteBreakdown(p)}
    ${renderVoteEvidence(p)}
    <div class="section-heading compact-heading">
      <p class="eyebrow">Top Activations</p>
      <h3>Saved Prototype Atlas Images</h3>
    </div>
    <div class="example-grid">${examples}</div>
  `;
  byId("prototype-detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

if (!report) {
  initMissingData();
} else {
  initSummary(report);
  populateFilters(report.prototypes || []);
  renderPredictionExplorer();
  renderPrototypes(report.prototypes || []);
  ["search", "class-filter", "label-filter"].forEach((id) => {
    if (hasId(id)) byId(id).addEventListener("input", () => renderPrototypes(report.prototypes || []));
  });
  ["evidence-scope", "evidence-limit", "class-score-limit"].forEach((id) => {
    if (hasId(id)) byId(id).addEventListener("input", renderPredictionExplorer);
  });
  document.querySelectorAll("[data-view-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      visualMode = button.dataset.viewMode || "heatmap";
      document.querySelectorAll("[data-view-mode]").forEach((item) => item.classList.toggle("active", item === button));
      const openId = document.querySelector("#prototype-detail .prototype-id")?.textContent;
      renderPredictionExplorer();
      renderPrototypes(report.prototypes || []);
      if (openId) showPrototype(openId);
    });
  });
}
