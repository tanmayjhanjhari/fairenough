import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Loader2, Sparkles, Tag } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import client from "../../api/client";
import useAnalysisStore from "../../store/analysisStore";
import ValidationCard from "./ValidationCard";

const DTYPE_COLORS = {
  int64:   { label: "numeric",     cls: "bg-accent/20 text-accent" },
  float64: { label: "numeric",     cls: "bg-accent/20 text-accent" },
  int32:   { label: "numeric",     cls: "bg-accent/20 text-accent" },
  float32: { label: "numeric",     cls: "bg-accent/20 text-accent" },
  object:  { label: "categorical", cls: "bg-accent2/20 text-accent2" },
  bool:    { label: "boolean",     cls: "bg-warning/20 text-warning" },
  default: { label: "other",       cls: "bg-surface text-textSecondary" },
};

function dtypeChip(dtype) {
  const entry = DTYPE_COLORS[dtype] ?? DTYPE_COLORS.default;
  return (
    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${entry.cls}`}>
      {entry.label}
    </span>
  );
}

const SCENARIO_OPTIONS = ["income", "hiring", "lending", "healthcare", "education", "other"];

const SCENARIO_COLORS = {
  income:     "bg-accent/20 text-accent border-accent/30",
  hiring:     "bg-accent/20 text-accent border-accent/30",
  lending:    "bg-accent2/20 text-accent2 border-accent2/30",
  healthcare: "bg-success/20 text-success border-success/30",
  education:  "bg-warning/20 text-warning border-warning/30",
  other:      "bg-surface text-textSecondary border-white/10",
};

// ── Main component ─────────────────────────────────────────────────────────────
export default function ColumnSelector() {
  const navigate   = useNavigate();
  const store      = useAnalysisStore();
  const { columns, dtypes, sessionId, modelId, preprocessingReport,
          setTarget, setSensitive, setScenario, setMetrics,
          setStep, setLoading } = store;

  const [targetCol, setTargetCol] = useState(() => {
    if (preprocessingReport?.uci_adult_detected && columns.includes("income_binary")) {
      return "income_binary";
    }
    return "";
  });

  const [sensitiveAttrs, setSensitiveAttrs] = useState(() => {
    if (preprocessingReport?.uci_adult_detected) {
      const uciSensitive = ["sex", "race"].filter(c => columns.includes(c));
      if (uciSensitive.length > 0) return uciSensitive;
    }
    return store.suggestedSensitive || [];
  });
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioData,    setScenarioData]    = useState(() => {
    if (!store.scenario) return null;
    if (typeof store.scenario === "string") {
      return { scenario: store.scenario, confidence_pct: 95, reason: "Auto-detected from dataset features" };
    }
    return store.scenario;
  });
  const [scenarioOverride, setScenarioOverride] = useState(false);
  const [analyzeLoading,  setAnalyzeLoading]  = useState(false);

  useEffect(() => {
    if (store.scenario && !scenarioData) {
      if (typeof store.scenario === "string") {
        setScenarioData({ scenario: store.scenario, confidence_pct: 95, reason: "Auto-detected from dataset features" });
      } else {
        setScenarioData(store.scenario);
      }
    }
  }, [store.scenario]);

  // ── Scenario detect ──────────────────────────────────────────────────────
  const handleDetectScenario = async () => {
    setScenarioLoading(true);
    try {
      const { data } = await client.post("/api/detect-scenario", {
        session_id: sessionId,
        columns,
      });
      setScenarioData(data);
      setScenario(data);
    } catch {
      /* handled by interceptor */
    } finally {
      setScenarioLoading(false);
    }
  };

  // ── Run analysis ─────────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    setAnalyzeLoading(true);
    setLoading(true);
    try {
      const chosenScenario = scenarioData?.scenario || (typeof store.scenario === "string" ? store.scenario : store.scenario?.scenario);
      const { data } = await client.post("/api/analyze", {
        session_id:      sessionId,
        target_col:      targetCol,
        sensitive_attrs: sensitiveAttrs,
        model_id:        modelId || undefined,
        scenario:        chosenScenario || undefined,
      });
      setMetrics(data);
      if (chosenScenario) {
        setScenario({
          scenario: chosenScenario,
          confidence_pct: scenarioData?.confidence_pct || 100,
          reason: scenarioData?.reason || "User-selected scenario"
        });
      }
      setTarget(targetCol);
      setSensitive(sensitiveAttrs);
      setStep(2);
      navigate("/results");
    } catch {
      /* handled by interceptor */
    } finally {
      setAnalyzeLoading(false);
      setLoading(false);
    }
  };

  const canAnalyze = targetCol && sensitiveAttrs.length > 0 && !analyzeLoading;

  // Columns excluding target for sensitive list
  const sensitiveOptions = useMemo(
    () => columns.filter((c) => c !== targetCol),
    [columns, targetCol]
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass-card p-6 space-y-6"
    >
      {/* ── Banner for UCI Adult ────────────────────────────────────────────── */}
      {preprocessingReport?.uci_adult_detected && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-accent/10 border border-accent/20">
          <Sparkles className="text-accent flex-shrink-0 mt-0.5" size={18} />
          <p className="text-sm text-textPrimary leading-relaxed">
            <span className="font-semibold text-accent">FairEnough detected the UCI Adult Income dataset</span> and pre-configured the analysis settings for you. Target variable is set to <code className="text-accent2 px-1 rounded bg-black/20">income_binary</code> and sensitive attributes to <code className="text-accent2 px-1 rounded bg-black/20">sex</code> and <code className="text-accent2 px-1 rounded bg-black/20">race</code>.
          </p>
        </div>
      )}

      {/* ── Target variable ─────────────────────────────────────────────────── */}
      <div>
        <p className="section-label">Target Variable</p>
        <div className="relative">
          <select
            value={targetCol}
            onChange={(e) => {
              setTargetCol(e.target.value);
              setSensitiveAttrs((prev) => prev.filter((a) => a !== e.target.value));
            }}
            className="input-base appearance-none pr-10"
          >
            <option value="">— Select target column —</option>
            {columns.map((col) => (
              <option key={col} value={col}>
                {col} ({dtypes[col] ?? "?"})
              </option>
            ))}
          </select>
          <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-textSecondary pointer-events-none" />
        </div>
        {targetCol && (
          <p className="text-xs text-textSecondary mt-1.5 flex items-center gap-1.5">
            Selected: <span className="text-textPrimary font-medium">{targetCol}</span>
            {dtypeChip(dtypes[targetCol])}
          </p>
        )}
      </div>

      {/* ── Sensitive attributes ─────────────────────────────────────────────── */}
      <div>
        <p className="section-label">Sensitive Attributes</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {sensitiveOptions.map((col) => {
            const checked = sensitiveAttrs.includes(col);
            let highlight = "";
            if (store.suggestedSensitive?.includes(col)) {
              highlight = "bg-warning/10 border-warning/40 text-warning-light";
            } else if (store.blockedSensitive?.includes(col)) {
              highlight = "bg-surface border-white/10 text-textSecondary opacity-70";
            } else {
              highlight = checked 
                ? "bg-accent/10 border-accent/40" 
                : "bg-surface/50 border-white/[0.06] hover:border-white/20";
            }

            return (
              <motion.label
                key={col}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors duration-150 ${highlight}`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(e) =>
                    setSensitiveAttrs((prev) =>
                      e.target.checked ? [...prev, col] : prev.filter((a) => a !== col)
                    )
                  }
                  className="mt-0.5 accent-teal-400 w-3.5 h-3.5 flex-shrink-0"
                />
                <div className="min-w-0">
                  <p className="text-xs font-medium text-textPrimary truncate">{col}</p>
                  <div className="flex items-center gap-1 mt-1 flex-wrap">
                    {dtypeChip(dtypes[col])}
                    {store.suggestedSensitive?.includes(col) && (
                      <span className="text-[10px] flex items-center gap-0.5 text-warning-light bg-warning/10 px-1.5 py-0.5 rounded-full font-medium">
                        <Sparkles size={10} /> suggested
                      </span>
                    )}
                  </div>
                </div>
              </motion.label>
            );
          })}
        </div>
        {sensitiveAttrs.length > 0 && (
          <p className="text-xs text-textSecondary mt-2">
            {sensitiveAttrs.length} attribute{sensitiveAttrs.length > 1 ? "s" : ""} selected
          </p>
        )}
      </div>

      {/* ── Validation card ───────────────────────────────────────────────────── */}
      <ValidationCard targetCol={targetCol} sensitiveAttrs={sensitiveAttrs} />

      {/* ── Detect scenario ───────────────────────────────────────────────────── */}
      {targetCol && sensitiveAttrs.length > 0 && (
        <div>
          <p className="section-label">Dataset Scenario</p>
          <div className="flex flex-wrap items-center gap-3">
            {!scenarioData ? (
              <button
                onClick={handleDetectScenario}
                disabled={scenarioLoading}
                className="btn-secondary flex items-center gap-2 text-sm"
              >
                {scenarioLoading ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Sparkles size={14} className="text-accent2" />
                )}
                {scenarioLoading ? "Detecting…" : "Detect Scenario with Gemini"}
              </button>
            ) : (
              <AnimatePresence>
                {!scenarioOverride ? (
                  <motion.div
                    key="chip"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-2"
                  >
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-sm font-semibold
                        ${SCENARIO_COLORS[scenarioData.scenario] ?? SCENARIO_COLORS.other}`}
                    >
                      <Sparkles size={13} />
                      {scenarioData.scenario?.charAt(0).toUpperCase() + scenarioData.scenario?.slice(1)}
                      <span className="opacity-60 font-normal">
                        · {scenarioData.confidence_pct}%
                      </span>
                    </span>
                    <button
                      onClick={() => setScenarioOverride(true)}
                      className="text-xs text-textSecondary underline underline-offset-2 hover:text-textPrimary"
                    >
                      Override
                    </button>
                  </motion.div>
                ) : (
                  <motion.div
                    key="override"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-center gap-2"
                  >
                    <div className="relative">
                      <select
                        defaultValue={scenarioData.scenario}
                        onChange={(e) => {
                          const updated = { ...scenarioData, scenario: e.target.value, confidence_pct: 100 };
                          setScenarioData(updated);
                          setScenario(updated);
                          setScenarioOverride(false);
                        }}
                        className="input-base py-1.5 pr-8 text-sm"
                      >
                        {SCENARIO_OPTIONS.map((s) => (
                          <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
                        ))}
                      </select>
                      <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-textSecondary pointer-events-none" />
                    </div>
                    <button
                      onClick={() => setScenarioOverride(false)}
                      className="text-xs text-textSecondary hover:text-textPrimary"
                    >
                      Done
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            )}
          </div>
        </div>
      )}

      {/* ── Run analysis button ───────────────────────────────────────────────── */}
      <div className="pt-2">
        <motion.button
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          whileHover={canAnalyze ? { scale: 1.02 } : {}}
          whileTap={canAnalyze   ? { scale: 0.97 } : {}}
          className="btn-primary w-full flex items-center justify-center gap-2 py-3 text-base"
        >
          {analyzeLoading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Running Bias Analysis…
            </>
          ) : (
            "Run Bias Analysis"
          )}
        </motion.button>
        {!canAnalyze && !analyzeLoading && (
          <p className="text-xs text-textSecondary text-center mt-2">
            Select a target variable and at least one sensitive attribute to continue.
          </p>
        )}
      </div>
    </motion.div>
  );
}
