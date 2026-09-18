import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowDown, ArrowUp, Info, Trophy, Settings, BarChart2, Zap, AlertCircle, Play, Loader2, ChevronDown, ChevronUp, CheckCircle2, XCircle } from "lucide-react";

const getVal = (obj, key) => obj?.[key] ?? obj?.[key.toUpperCase()] ?? obj?.[key.toLowerCase()];

// null/undefined means genuinely unavailable — NOT the same as 0.000
function DeltaRow({ label, before, after, tooltip }) {
  const isUnavailable = before === null || before === undefined || after === null || after === undefined;

  if (isUnavailable) {
    return (
      <tr className="border-b border-white/[0.04] last:border-0">
        <td className="truncate px-2 py-1.5 min-w-0 font-medium text-textSecondary" title={tooltip}>{label}</td>
        <td className="truncate px-2 py-1.5 min-w-0 text-textSecondary/40 text-right text-xs italic">
          {before !== null && before !== undefined ? Number(before).toFixed(3) : "N/A"}
        </td>
        <td className="truncate px-2 py-1.5 min-w-0 text-textSecondary/40 text-right text-xs italic">N/A</td>
        <td className="truncate px-2 py-1.5 min-w-0 text-textSecondary/40 text-right text-xs">—</td>
      </tr>
    );
  }

  const b = Number(before);
  const a = Number(after);
  const delta = a - b;
  const absDelta = Math.abs(delta);
  let isImprovement = false;
  if (label.toUpperCase() === "DI") {
    isImprovement = Math.abs(1 - a) < Math.abs(1 - b);
  } else {
    isImprovement = Math.abs(a) < Math.abs(b);
  }

  return (
    <tr className="border-b border-white/[0.04] last:border-0">
      <td className="truncate px-2 py-1.5 min-w-0 font-medium text-textSecondary" title={tooltip}>{label}</td>
      <td className="truncate px-2 py-1.5 min-w-0 text-textPrimary text-right">{b.toFixed(3)}</td>
      <td className="truncate px-2 py-1.5 min-w-0 text-textPrimary text-right">{a.toFixed(3)}</td>
      <td className={`truncate px-2 py-1.5 min-w-0 font-medium text-right ${isImprovement ? "text-success" : "text-danger"}`}>
        <div className="flex items-center justify-end gap-1">
          {isImprovement ? <ArrowDown size={14} className="flex-shrink-0" /> : <ArrowUp size={14} className="flex-shrink-0" />}
          <span className="truncate">{absDelta.toFixed(3)}</span>
        </div>
      </td>
    </tr>
  );
}

function MetricCompact({ label, before, after, tooltip }) {
  const isUnavailable = before === null || before === undefined || after === null || after === undefined;

  if (isUnavailable) {
    return (
      <div title={tooltip}>
        <p className="text-[10px] text-textSecondary uppercase tracking-wider mb-1">{label}</p>
        <div className="flex items-baseline gap-1">
          <span className="text-sm text-textSecondary/40 italic">N/A</span>
        </div>
      </div>
    );
  }

  const b = Number(before);
  const a = Number(after);
  const delta = a - b;
  const isDrop = delta < 0;

  return (
    <div title={tooltip}>
       <p className="text-[10px] text-textSecondary uppercase tracking-wider mb-1">{label}</p>
       <div className="flex items-baseline gap-1.5">
          <span className="text-sm font-semibold text-textPrimary">{a.toFixed(3)}</span>
          <span className={`text-[10px] font-medium ${isDrop ? "text-danger" : "text-success"}`}>
             {delta > 0 ? "+" : ""}{delta.toFixed(3)}
          </span>
       </div>
    </div>
  );
}

// Reweighing transparency explanation panel
function ReweighExplainer({ data }) {
  const [open, setOpen] = useState(false);
  const ws = data?.weights_summary || {};
  const modelRetrained = data?.model_retrained === true;
  const swSupported = data?.sample_weight_supported;
  const originalModelType = data?.original_model_type;
  const nTrain = data?.n_train_samples;
  const nEval = data?.n_eval_samples;
  const after = data?.after || {};

  return (
    <div className="mt-4 border border-white/[0.06] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-textSecondary uppercase tracking-widest bg-surface/40 hover:bg-surface/70 transition-colors"
      >
        <span>What changed? What did not?</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="px-4 py-4 space-y-5 bg-surface/20">

          {/* Process flow */}
          <div>
            <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider mb-3">How Reweighing Works — Step by Step</p>
            <div className="space-y-2">
              {[
                { step: "1", text: "Count how often each (group, outcome) combination appears in the dataset." },
                { step: "2", text: "Compute statistical weights: w = P(Group) × P(Outcome) / P(Group, Outcome). Under-represented combinations get a weight > 1; over-represented ones get < 1." },
                { step: "3", text: modelRetrained
                    ? `Clone the original model (${originalModelType || "model"}) and refit it on 70% of the data using these sample weights. Evaluate on the remaining 30% held-out test split.`
                    : "Apply weights to the dataset outcome distribution to compute the expected fairness metrics without retraining a model." },
                { step: "4", text: modelRetrained
                    ? `Evaluate the retrained model on the SAME ${nEval ? nEval + " " : ""}held-out evaluation samples to compute NEW fairness metrics (SPD, DI, EOD, AOD) and NEW performance metrics (Accuracy, Precision, Recall, F1) from actual model predictions.`
                    : "Measure dataset-level SPD and DI from the reweighted outcome distribution (performance metrics are unavailable without a retrained model)." },
              ].map(({ step, text }) => (
                <div key={step} className="flex gap-3 items-start">
                  <span className="w-5 h-5 rounded-full bg-accent/20 border border-accent/30 flex-shrink-0 flex items-center justify-center text-[9px] font-bold text-accent">{step}</span>
                  <p className="text-xs text-textSecondary/90 leading-relaxed">{text}</p>
                </div>
              ))}
            </div>
          </div>

          {/* What changed / What did not */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-success/5 border border-success/15 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 size={13} className="text-success flex-shrink-0" />
                <p className="text-[10px] font-semibold text-success uppercase tracking-wider">What Changed</p>
              </div>
              <ul className="space-y-1">
                {[
                  modelRetrained
                    ? `New fairness metrics (SPD, DI, EOD, AOD) computed from retrained model predictions on held-out test set`
                    : "Statistical outcome distribution (SPD, DI) reweighted across the dataset",
                  modelRetrained ? `A fresh clone of ${originalModelType || "the model"} was trained using sample weights` : null,
                  modelRetrained ? `Performance metrics (Accuracy, Precision, Recall, F1) evaluated on ${nEval} held-out test samples` : null,
                  modelRetrained ? "Learned decision boundary adjusted to balance group outcomes" : null,
                ].filter(Boolean).map((item, i) => (
                  <li key={i} className="text-xs text-success/80 leading-relaxed flex gap-1.5 items-start">
                    <span className="mt-1 flex-shrink-0">•</span><span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <XCircle size={13} className="text-textSecondary flex-shrink-0" />
                <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider">What Did NOT Change</p>
              </div>
              <ul className="space-y-1">
                {[
                  modelRetrained ? `The original deployed model (${originalModelType || "model"}) — only a clone is retrained` : "The original model — no model was modified",
                  "The raw input data and ground-truth labels",
                  "The inference procedure and model architecture (classification threshold and feature structure)",
                  "Feature engineering, preprocessing, and column definitions",
                ].map((item, i) => (
                  <li key={i} className="text-xs text-textSecondary/70 leading-relaxed flex gap-1.5 items-start">
                    <span className="mt-1 flex-shrink-0">•</span><span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Weight summary */}
          {(ws.min != null) && (
            <div>
              <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider mb-2">Sample Weight Statistics</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: "Min Weight", value: ws.min?.toFixed(3) },
                  { label: "Mean Weight", value: ws.mean?.toFixed(3) },
                  { label: "Max Weight", value: ws.max?.toFixed(3) },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-surface/40 rounded p-2 text-center">
                    <p className="text-[9px] text-textSecondary/70">{label}</p>
                    <p className="text-xs font-semibold text-textPrimary">{value ?? "—"}</p>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-textSecondary/50 mt-2">
                Canonical weights computed via w = P(G)·P(Y)/P(G,Y) and bounded to [0.1, 10.0] as a numerical safeguard against extreme gradient updates. Weight = 1.0 means unaffected; &gt;1.0 = upweighted; &lt;1.0 = downweighted.
              </p>
            </div>
          )}

          {/* Model retraining info */}
          {originalModelType && (
            <div className={`rounded-lg p-3 border ${modelRetrained ? "bg-emerald-500/5 border-emerald-500/15" : "bg-amber-500/5 border-amber-500/15"}`}>
              <p className={`text-[10px] font-semibold uppercase tracking-wider mb-1 ${modelRetrained ? "text-emerald-400" : "text-amber-400"}`}>
                Model Retraining — {modelRetrained ? "Performed" : "Not Performed"}
              </p>
              {modelRetrained ? (
                <div className="space-y-1">
                  <p className="text-xs text-emerald-300/80">Original model: <span className="font-semibold">{originalModelType}</span></p>
                  <p className="text-xs text-emerald-300/80">Training samples used: <span className="font-semibold">{nTrain}</span></p>
                  <p className="text-xs text-emerald-300/80">Evaluation samples: <span className="font-semibold">{nEval}</span></p>
                  <p className="text-xs text-emerald-300/60 mt-2">
                    A fresh clone of the model was fitted with the reweighing sample weights. The original model is unchanged.
                  </p>
                </div>
              ) : (
                <p className="text-xs text-amber-300/70">
                  {swSupported === false
                    ? `${originalModelType} does not support sample_weight in fit(). SPD/DI reflect dataset-level reweighted distributions. Performance metrics are not available for this reason.`
                    : "Dataset-level reweighing only. Upload a model with sample_weight support to see performance metrics after retraining."
                  }
                </p>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
}

// Threshold Adjustment transparency explanation panel
function ThresholdExplainer({ data }) {
  const [open, setOpen] = useState(false);
  const thresholds = data?.thresholds || {};
  const isSimulation = data?.is_simulation === true;
  const modelType = data?.model_info?.type || null;
  const hasThresholds = Object.keys(thresholds).length > 0;

  return (
    <div className="mt-4 border border-white/[0.06] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-textSecondary uppercase tracking-widest bg-surface/40 hover:bg-surface/70 transition-colors"
      >
        <span>What changed? What did not?</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="px-4 py-4 space-y-5 bg-surface/20">

          {/* Process flow */}
          <div>
            <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider mb-3">How Threshold Adjustment Works — Step by Step</p>
            <div className="space-y-2">
              {[
                { step: "1", text: isSimulation ? "An internal GBM model generates probability scores for each sample (because no real model was uploaded)." : "The uploaded model's predict_proba() generates a probability score for each sample." },
                { step: "2", text: "Samples are grouped by demographic attribute (e.g. Male/Female)." },
                { step: "3", text: "Per-group decision thresholds are optimised to minimise SPD (the fairness gap). Each group gets its own cut-off instead of a single global 0.5 threshold." },
                { step: "4", text: "Final predictions are made by comparing each sample's probability to its group's threshold. SPD, DI, EOD, AOD and performance are measured." },
              ].map(({ step, text }) => (
                <div key={step} className="flex gap-3 items-start">
                  <span className="w-5 h-5 rounded-full bg-violet-500/20 border border-violet-500/30 flex-shrink-0 flex items-center justify-center text-[9px] font-bold text-violet-400">{step}</span>
                  <p className="text-xs text-textSecondary/90 leading-relaxed">{text}</p>
                </div>
              ))}
            </div>
          </div>

          {/* What changed / What did not */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-success/5 border border-success/15 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 size={13} className="text-success flex-shrink-0" />
                <p className="text-[10px] font-semibold text-success uppercase tracking-wider">What Changed</p>
              </div>
              <ul className="space-y-1">
                {[
                  "The decision threshold — calibrated per demographic group to equalise positive rates",
                  "Which samples receive a positive prediction based on group-specific cutoffs",
                  "SPD, DI, EOD, AOD (measured from new group-specific decisions)",
                  "Accuracy, Precision, Recall, F1 (re-evaluated under adjusted thresholds)",
                ].map((item, i) => (
                  <li key={i} className="text-xs text-success/80 leading-relaxed flex gap-1.5 items-start">
                    <span className="mt-1 flex-shrink-0">•</span><span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <XCircle size={13} className="text-textSecondary flex-shrink-0" />
                <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider">What Did NOT Change</p>
              </div>
              <ul className="space-y-1">
                {[
                  "The model's internal weights, parameters, and decision boundaries (no retraining)",
                  "The model's raw probability scores from predict_proba (probabilities remain identical)",
                  "The training data, features, and model architecture",
                  "The original deployed model artifact",
                ].map((item, i) => (
                  <li key={i} className="text-xs text-textSecondary/70 leading-relaxed flex gap-1.5 items-start">
                    <span className="mt-1 flex-shrink-0">•</span><span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Per-group thresholds */}
          {hasThresholds && (
            <div>
              <p className="text-[10px] font-semibold text-textSecondary uppercase tracking-wider mb-2">Per-Group Decision Thresholds</p>
              <div className="space-y-2">
                {Object.entries(thresholds).map(([group, threshold]) => (
                  <div key={group} className="flex items-center justify-between bg-surface/40 rounded px-3 py-2">
                    <div>
                      <p className="text-xs font-medium text-textPrimary">{group}</p>
                      <p className="text-[10px] text-textSecondary/60">Group-specific cutoff</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-textSecondary/50 line-through">0.500</p>
                      <p className="text-sm font-semibold text-violet-300">{Number(threshold).toFixed(3)}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-textSecondary/50 mt-2">
                Default threshold was 0.5. Adjusted thresholds equalise positive decision rates across groups.
              </p>
            </div>
          )}

          {/* Simulation note */}
          {isSimulation && (
            <div className="bg-amber-500/5 border border-amber-500/15 rounded-lg p-3">
              <p className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider mb-1">Using Internal Simulation</p>
              <p className="text-xs text-amber-300/70">
                No real model was uploaded. An internal GBM simulation generates probability scores to demonstrate threshold adjustment.
                Upload a real sklearn model with predict_proba() to see results from your actual model.
              </p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

const METRIC_TOOLTIPS = {
  SPD: "Statistical Parity Difference: positive outcome rate (unpriv) − (priv). 0 = fair. |>0.1| = bias.",
  DI: "Disparate Impact: rate(unpriv) / rate(priv). Below 0.8 fails the legal 80% rule.",
  EOD: "Equal Opportunity Difference: True Positive Rate gap between groups. 0 = fair.",
  AOD: "Average Odds Difference: average of TPR and FPR gaps between groups. 0 = fair.",
  accuracy: "Fraction of all predictions that are correct.",
  precision: "Of all positive predictions, what fraction were actually positive.",
  recall: "Of all actual positives, what fraction were correctly identified.",
  f1: "Harmonic mean of Precision and Recall. Balances both error types.",
};

export default function TechniqueCard({
  name,
  data,
  isWinner,
  winnerReason,
  onRunSimulation,
  isSimulating = false,
}) {
  if (!data) return null;

  const title = name === "reweigh" ? "Reweighing" : "Threshold Adjustment";
  const desc = name === "reweigh"
    ? "Adjusts training data weights to ensure demographic balance."
    : "Finds per-group decision thresholds to equalise outcome rates.";

  const before = data.before || {};
  const after = data.after;
  const effects = data.effects || {};
  const isSimulation = data.is_simulation === true;
  const simulationNote = data.simulation_note;
  const isModelRequired = data.model_required === true;
  const hasRealModel = Boolean(
    data.has_real_model === true ||
    (!isSimulation && !isModelRequired && (data.model_retrained || (after?.accuracy != null && before?.accuracy != null)))
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card p-5 border ${
        isWinner
          ? "border-accent/30 shadow-lg shadow-accent/5"
          : "border-white/[0.06]"
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
            name === "reweigh" ? "bg-accent/15" : "bg-violet-500/15"
          }`}>
            {name === "reweigh" ? (
              <Settings size={16} className="text-accent" />
            ) : (
              <Zap size={16} className="text-violet-400" />
            )}
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-textPrimary text-sm truncate">{title}</h3>
            <p className="text-xs text-textSecondary truncate">{desc}</p>
          </div>
        </div>

        {isWinner && (
          <div className="flex-shrink-0 flex items-center gap-1.5 bg-accent/15 border border-accent/25 rounded-full px-3 py-1">
            <Trophy size={12} className="text-accent" />
            <span className="text-xs font-semibold text-accent">Recommended</span>
          </div>
        )}
      </div>

      {/* Winner reason */}
      {isWinner && winnerReason && (() => {
        let bannerText = winnerReason;
        if (name === "reweigh" && bannerText.includes("achieved 0.0% bias reduction")) {
          const spdB = before?.SPD != null ? Math.abs(before.SPD) : null;
          const spdA = after?.SPD != null ? Math.abs(after.SPD) : null;
          if (spdB != null && spdA != null && spdA > spdB) {
            bannerText = bannerText
              .replace(/Reweighing achieved 0\.0% bias reduction/g, `Reweighing: absolute SPD gap increased from ${spdB.toFixed(3)} to ${spdA.toFixed(3)}`)
              .replace(/achieved 0\.0% bias reduction/g, `absolute SPD gap increased from ${spdB.toFixed(3)} to ${spdA.toFixed(3)}`);
          }
        }
        return (
          <div className="mb-4 flex items-start gap-2 bg-accent/5 border border-accent/15 rounded-lg px-3 py-2">
            <Info size={13} className="text-accent/70 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-accent/80 leading-relaxed">{bannerText}</p>
          </div>
        );
      })()}

      {/* Model required notice */}
      {isModelRequired && (
        <div className="mb-4 flex flex-col gap-3 bg-white/[0.03] border border-white/10 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertCircle size={14} className="text-textSecondary/70 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-textSecondary/80 leading-relaxed">
              Threshold adjustment requires probability scores from a trained model.
            </p>
          </div>
          {onRunSimulation && (
            <button
              onClick={onRunSimulation}
              disabled={isSimulating}
              className="flex items-center gap-2 text-xs font-medium text-accent border border-accent/30 rounded-lg px-3 py-2 hover:bg-accent/10 transition-colors disabled:opacity-50"
            >
              {isSimulating
                ? <><Loader2 size={13} className="animate-spin" /> Running simulation…</>
                : <><Play size={13} /> Run with internal simulation</>
              }
            </button>
          )}
        </div>
      )}

      {/* Simulation disclosure */}
      {isSimulation && !isModelRequired && (
        <div className="mb-4 flex items-start gap-2 bg-amber-500/6 border border-amber-500/20 rounded-lg px-3 py-2">
          <Info size={13} className="text-amber-400/70 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/70 leading-relaxed">
            Simulation only — not real model performance. An internal simulation model was used to demonstrate threshold adjustment.
          </p>
        </div>
      )}
      {!isSimulation && !isModelRequired && simulationNote && (
        <div className="mb-4 flex items-start gap-2 bg-blue-500/6 border border-blue-500/20 rounded-lg px-3 py-2">
          <Info size={13} className="text-blue-400/60 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-blue-300/60 leading-relaxed">{simulationNote}</p>
        </div>
      )}

      {/* Real Model Failure / Unavailable error state */}
      {(data.error || after?.error) && (
        <div className="mb-4 flex items-start gap-2 bg-red-500/10 border border-red-500/25 rounded-lg px-3 py-2 text-red-400">
          <Info size={14} className="flex-shrink-0 mt-0.5 text-red-400" />
          <div>
            <p className="text-xs font-semibold">Real model evaluation unavailable</p>
            <p className="text-xs text-red-300/80 leading-relaxed mt-0.5">{data.error || after?.error}</p>
          </div>
        </div>
      )}

      {/* Fairness Deltas */}
      <div className="mb-6">
        <h4 className="text-xs font-semibold text-textSecondary uppercase tracking-widest mb-2 border-b border-white/[0.06] pb-2">
          Fairness Impact
        </h4>
        <div className="bg-surface/30 rounded-lg p-2">
          <table className="w-full table-fixed text-xs">
            <thead>
              <tr className="text-textSecondary font-medium border-b border-white/[0.06]">
                <th className="truncate px-2 py-1.5 min-w-0 text-left" style={{ width: "25%" }}>Metric</th>
                <th className="truncate px-2 py-1.5 min-w-0 text-right" style={{ width: "22%" }}>Before</th>
                <th className="truncate px-2 py-1.5 min-w-0 text-right" style={{ width: "22%" }}>After</th>
                <th className="truncate px-2 py-1.5 min-w-0 text-right" style={{ width: "31%" }}>Change</th>
              </tr>
            </thead>
            <tbody>
              {["SPD", "DI", "EOD", "AOD"].map(m => (
                <DeltaRow
                  key={m}
                  label={m}
                  tooltip={METRIC_TOOLTIPS[m]}
                  before={getVal(before, m)}
                  after={isModelRequired ? null : getVal(after, m)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Effects */}
      <div>
        <h4 className="text-xs font-semibold text-textSecondary uppercase tracking-widest mb-2 border-b border-white/[0.06] pb-2">
          Performance Trade-off
        </h4>
        {isSimulation ? (
          <div className="flex items-start gap-1.5 mb-3 bg-amber-500/5 border border-amber-500/12 rounded px-2 py-1.5">
            <Info size={11} className="text-amber-400/60 flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-amber-300/60 leading-relaxed">Simulation only — not real model performance</p>
          </div>
        ) : hasRealModel ? (
          <div className="flex items-start gap-1.5 mb-3 bg-emerald-500/5 border border-emerald-500/15 rounded px-2 py-1.5">
            <Zap size={11} className="text-emerald-400/70 flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-emerald-300/70 leading-relaxed">
              {name === "reweigh" && data.model_retrained
                ? "Real model retrained with sample weights. Performance evaluated on held-out test set."
                : "Real model performance evaluated directly on uploaded model."}
            </p>
          </div>
        ) : (
          <div className="flex items-start gap-1.5 mb-3 bg-white/[0.03] border border-white/10 rounded px-2 py-1.5">
            <Info size={11} className="text-textSecondary/80 flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-textSecondary/90 leading-relaxed">Model-level performance unavailable — no model uploaded</p>
          </div>
        )}
        <div className="grid grid-cols-4 gap-2">
          <MetricCompact
            label="Acc"
            tooltip={METRIC_TOOLTIPS.accuracy}
            before={isModelRequired ? null : getVal(before, "accuracy")}
            after={isModelRequired ? null : getVal(after, "accuracy")}
          />
          <MetricCompact
            label="Pre"
            tooltip={METRIC_TOOLTIPS.precision}
            before={isModelRequired ? null : getVal(before, "precision")}
            after={isModelRequired ? null : getVal(after, "precision")}
          />
          <MetricCompact
            label="Rec"
            tooltip={METRIC_TOOLTIPS.recall}
            before={isModelRequired ? null : getVal(before, "recall")}
            after={isModelRequired ? null : getVal(after, "recall")}
          />
          <MetricCompact
            label="F1"
            tooltip={METRIC_TOOLTIPS.f1}
            before={isModelRequired ? null : getVal(before, "f1")}
            after={isModelRequired ? null : getVal(after, "f1")}
          />
        </div>
      </div>

      {/* Non-winner diagnostic */}
      {!isWinner && !isModelRequired && (
        <div className="mt-5 flex items-start gap-2 text-xs text-textSecondary bg-surface/50 p-3 rounded-lg">
          <Info size={14} className="flex-shrink-0 mt-0.5 opacity-70" />
          <p>
            Not recommended because {
              (effects.bias_reduction_pct || 0) < 30
                ? "it achieved insufficient bias reduction."
                : "it resulted in a more severe accuracy drop compared to the alternative."
            }
          </p>
        </div>
      )}

      {/* Diagnostic box when bias reduction is very low */}
      {!isModelRequired && effects.diagnostic && (
        <div className="mt-4 flex items-start gap-2 text-xs bg-amber-500/10 border border-amber-500/25 p-3 rounded-lg">
          <Info size={14} className="flex-shrink-0 mt-0.5 text-amber-400" />
          <p className="text-amber-300/90 leading-relaxed">{effects.diagnostic}</p>
        </div>
      )}

      {/* Understanding this result */}
      {!isModelRequired && data.explanation && (
        <div className="mt-6 border-t border-white/[0.06] pt-4">
          <h4 className="text-xs font-semibold text-textSecondary uppercase tracking-widest mb-3">
            Understanding this result
          </h4>
          <div className="space-y-4">
            <div className="flex gap-3 items-start">
              <Settings size={16} className="text-textSecondary mt-0.5" />
              <p className="text-xs text-textSecondary leading-relaxed">
                {data.explanation.how_it_works}
              </p>
            </div>

            <div className="flex gap-3 items-start">
              <BarChart2 size={16} className={`${
                (effects.bias_reduction_pct || 0) > 50 ? "text-success" :
                (effects.bias_reduction_pct || 0) >= 10 ? "text-warning" : "text-danger"
              } mt-0.5`} />
              <p className={`text-xs leading-relaxed ${
                (effects.bias_reduction_pct || 0) > 50 ? "text-success-light" :
                (effects.bias_reduction_pct || 0) >= 10 ? "text-warning-light" : "text-danger-light"
              }`}>
                {data.explanation.bias_result}
              </p>
            </div>

            <div className="flex gap-3 items-start">
              <Zap size={16} className={`${
                (effects.accuracy_retained_pct || 0) > 98 ? "text-success" :
                (effects.accuracy_retained_pct || 0) >= 90 ? "text-warning" : "text-danger"
              } mt-0.5`} />
              <p className={`text-xs leading-relaxed ${
                (effects.accuracy_retained_pct || 0) > 98 ? "text-success-light" :
                (effects.accuracy_retained_pct || 0) >= 90 ? "text-warning-light" : "text-danger-light"
              }`}>
                {data.explanation.acc_result}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Technique-specific explainer */}
      {!isModelRequired && (
        name === "reweigh"
          ? <ReweighExplainer data={data} />
          : <ThresholdExplainer data={data} />
      )}
    </motion.div>
  );
}
