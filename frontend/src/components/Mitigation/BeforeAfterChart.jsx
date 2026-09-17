import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { Info, CheckCircle2, AlertCircle } from "lucide-react";

// Null-safe extractor: preserves null when metric is genuinely unavailable
const getVal = (obj, key) => {
  const v = obj?.[key] ?? obj?.[key?.toUpperCase()] ?? obj?.[key?.toLowerCase()];
  return v !== undefined && v !== null ? Number(v) : null;
};

export default function BeforeAfterChart({ mitigation }) {
  if (!mitigation) return null;

  const rewBefore = mitigation.reweigh?.before || {};
  const rewAfter  = mitigation.reweigh?.after  || {};
  const thrAfter  = mitigation.threshold?.after || {};
  const thrIsSimulation = mitigation.threshold?.is_simulation === true;

  // ── 1. Parity Differences (SPD, EOD, AOD: Ideal = 0.000) ──────────────────
  const diffMetricKeys = ["SPD", "EOD", "AOD"];
  const activeDiffMetrics = diffMetricKeys.filter(m => {
    const vB = getVal(rewBefore, m);
    const vR = getVal(rewAfter, m);
    const vT = getVal(thrAfter, m);
    return vB !== null || vR !== null || vT !== null;
  });

  const diffData = activeDiffMetrics.map(m => {
    const vB = getVal(rewBefore, m);
    const vR = getVal(rewAfter, m);
    const vT = getVal(thrAfter, m);
    return {
      name: m,
      rawBefore: vB,
      rawReweighing: vR,
      rawThreshold: vT,
      // Use absolute gap for bar height so scale is strictly positive and proportional to disparity
      Before: vB !== null ? Math.abs(vB) : undefined,
      Reweighing: vR !== null ? Math.abs(vR) : undefined,
      Threshold: vT !== null ? Math.abs(vT) : undefined,
    };
  });

  // ── 2. Disparate Impact Ratio (DI: Ideal = 1.000, 80% Rule >= 0.800) ────────
  const diBefore = getVal(rewBefore, "DI");
  const diRewAfter = getVal(rewAfter, "DI");
  const diThrAfter = getVal(thrAfter, "DI");
  const hasDi = diBefore !== null || diRewAfter !== null || diThrAfter !== null;

  const diData = hasDi
    ? [
        {
          name: "Disparate Impact",
          Before: diBefore !== null ? diBefore : undefined,
          Reweighing: diRewAfter !== null ? diRewAfter : undefined,
          Threshold: diThrAfter !== null ? diThrAfter : undefined,
        },
      ]
    : [];

  const omittedMetrics = ["SPD", "DI", "EOD", "AOD"].filter(m => {
    if (m === "DI") return !hasDi;
    return !activeDiffMetrics.includes(m);
  });

  // Custom tooltip for Difference Metrics
  const DiffTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const item = payload[0]?.payload;
      return (
        <div className="bg-surface/95 border border-white/10 rounded-lg p-3 shadow-xl backdrop-blur-md text-xs">
          <p className="font-semibold text-textPrimary mb-1 border-b border-white/10 pb-1">
            {label} — Absolute Parity Gap (|Gap| from 0.000)
          </p>
          <div className="space-y-1.5 my-2">
            {item.rawBefore !== null && item.rawBefore !== undefined && (
              <div className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#64748B]" />
                  Before:
                </span>
                <span className="font-mono text-textPrimary font-medium">
                  {item.rawBefore.toFixed(4)} (|gap|: {Math.abs(item.rawBefore).toFixed(4)})
                </span>
              </div>
            )}
            {item.rawReweighing !== null && item.rawReweighing !== undefined && (
              <div className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-teal-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#14B8A6]" />
                  Reweighing:
                </span>
                <span className="font-mono text-textPrimary font-medium">
                  {item.rawReweighing.toFixed(4)} (|gap|: {Math.abs(item.rawReweighing).toFixed(4)})
                </span>
              </div>
            )}
            {item.rawThreshold !== null && item.rawThreshold !== undefined && (
              <div className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-purple-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#A855F7]" />
                  Threshold:
                </span>
                <span className="font-mono text-textPrimary font-medium">
                  {item.rawThreshold.toFixed(4)} (|gap|: {Math.abs(item.rawThreshold).toFixed(4)})
                  {thrIsSimulation ? " *" : ""}
                </span>
              </div>
            )}
          </div>
          <p className="text-[10px] text-textSecondary opacity-75 pt-1 border-t border-white/5">
            Ideal parity gap is 0.000. Bars show |gap| auto-scaled to differences.
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom tooltip for Disparate Impact
  const DiTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-surface/95 border border-white/10 rounded-lg p-3 shadow-xl backdrop-blur-md text-xs">
          <p className="font-semibold text-textPrimary mb-1 border-b border-white/10 pb-1">
            Disparate Impact Ratio (Unprivileged / Privileged)
          </p>
          <div className="space-y-1.5 my-2">
            {payload.map((entry, idx) => {
              const val = entry.value;
              const passes80 = val >= 0.8 && val <= 1.25;
              return (
                <div key={idx} className="flex items-center justify-between gap-4">
                  <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: entry.color }} />
                    {entry.name}:
                  </span>
                  <span className="font-mono text-textPrimary font-medium flex items-center gap-1">
                    {val !== undefined ? val.toFixed(3) : "N/A"}
                    {passes80 ? (
                      <CheckCircle2 size={12} className="text-emerald-400" />
                    ) : (
                      <AlertCircle size={12} className="text-amber-400" />
                    )}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] space-y-0.5 text-textSecondary opacity-80 pt-1 border-t border-white/5">
            <p>• Ideal parity is 1.000</p>
            <p>• Legal 80% Rule requires ratio ≥ 0.800</p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* ── Section 1: Parity Differences (SPD, EOD, AOD) ───────────────── */}
      <div className="rounded-lg bg-surface/30 p-4 border border-white/[0.04]">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h4 className="text-sm font-semibold text-textPrimary">Parity Differences (|Gap|)</h4>
            <p className="text-xs text-textSecondary">
              Differences in selection and error rates. Closer to 0.000 = fairer.
            </p>
          </div>
          <span className="text-[11px] font-mono text-textSecondary bg-white/5 px-2 py-0.5 rounded border border-white/5">
            Ideal: 0.000
          </span>
        </div>

        {diffData.length > 0 ? (
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diffData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#94A3B8", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#94A3B8", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, "auto"]}
                  tickFormatter={(v) => v.toFixed(3)}
                />
                <Tooltip content={<DiffTooltip />} />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "4px" }} />
                <Bar dataKey="Before" fill="#64748B" radius={[3, 3, 0, 0]} maxBarSize={28} />
                <Bar dataKey="Reweighing" fill="#14B8A6" radius={[3, 3, 0, 0]} maxBarSize={28} />
                <Bar dataKey="Threshold" fill="#A855F7" radius={[3, 3, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-xs text-textSecondary/60 py-4 text-center">No parity difference metrics available.</p>
        )}
      </div>

      {/* ── Section 2: Disparate Impact Ratio (DI) ──────────────────────── */}
      {hasDi && (
        <div className="rounded-lg bg-surface/30 p-4 border border-white/[0.04]">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h4 className="text-sm font-semibold text-textPrimary">Selection Rate Ratio (Disparate Impact)</h4>
              <p className="text-xs text-textSecondary">
                Ratio of unprivileged to privileged positive rate. 80% Rule requires ≥ 0.800.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-amber-400/90 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                Rule: ≥ 0.800
              </span>
              <span className="text-[11px] font-mono text-emerald-400/90 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Parity: 1.000
              </span>
            </div>
          </div>

          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#94A3B8", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#94A3B8", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, (dataMax) => Math.max(1.15, Number((dataMax * 1.1).toFixed(2)))]}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <ReferenceLine y={0.8} stroke="#F59E0B" strokeDasharray="3 3" label={{ value: "80% Threshold (0.80)", fill: "#F59E0B", fontSize: 10, position: "insideTopRight" }} />
                <ReferenceLine y={1.0} stroke="#10B981" strokeDasharray="3 3" label={{ value: "Parity (1.00)", fill: "#10B981", fontSize: 10, position: "insideBottomRight" }} />
                <Tooltip content={<DiTooltip />} />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "4px" }} />
                <Bar dataKey="Before" fill="#64748B" radius={[3, 3, 0, 0]} maxBarSize={32} />
                <Bar dataKey="Reweighing" fill="#14B8A6" radius={[3, 3, 0, 0]} maxBarSize={32} />
                <Bar dataKey="Threshold" fill="#A855F7" radius={[3, 3, 0, 0]} maxBarSize={32} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Disclaimers & Notes ─────────────────────────────────────────── */}
      {omittedMetrics.length > 0 && (
        <div className="flex items-start gap-2 bg-surface/40 rounded-lg px-3 py-2 border border-white/[0.04]">
          <Info size={13} className="text-textSecondary/50 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-textSecondary/50">
            {omittedMetrics.join(", ")} omitted from chart — require model predictions which are not available in dataset-only analysis.
          </p>
        </div>
      )}

      {thrIsSimulation && (
        <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <Info size={13} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/80 leading-relaxed">
            <span className="font-semibold text-amber-300">Threshold values are from an internal simulation model.</span>{" "}
            A real model was not provided for threshold adjustment. Reweighing before/after values are from actual dataset outcome distributions.
          </p>
        </div>
      )}

      {mitigation.reweigh?.explanation?.graph_explanation && (
        <div className="flex items-start gap-2 bg-surface/50 rounded-lg p-3 border border-white/[0.04]">
          <Info size={15} className="text-textSecondary flex-shrink-0 mt-0.5" />
          <p className="text-xs text-textSecondary leading-relaxed">
            {mitigation.reweigh.explanation.graph_explanation}
          </p>
        </div>
      )}
    </div>
  );
}
