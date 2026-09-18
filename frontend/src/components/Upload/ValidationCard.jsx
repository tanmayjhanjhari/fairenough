import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle, Info } from "lucide-react";
import { useEffect, useState } from "react";
import client from "../../api/client";
import useAnalysisStore from "../../store/analysisStore";

export default function ValidationCard({ targetCol, sensitiveAttrs }) {
  const sessionId    = useAnalysisStore((s) => s.sessionId);
  const setValidation = useAnalysisStore((s) => s.setValidation);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  // Auto-run validation whenever target or sensitive attrs change
  useEffect(() => {
    if (!sessionId || !targetCol || sensitiveAttrs.length === 0) {
      setResult(null);
      return;
    }

    let cancelled = false;
    const run = async () => {
      setLoading(true);
      try {
        const { data } = await client.post("/api/validate", {
          session_id:      sessionId,
          target_col:      targetCol,
          sensitive_attrs: sensitiveAttrs,
        }, { skipToast: true });
        if (!cancelled) {
          setResult(data);
          setValidation(data);
        }
      } catch {
        /* interceptor handles toast */
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => { cancelled = true; };
  }, [sessionId, targetCol, sensitiveAttrs.join(",")]);

  if (!targetCol || sensitiveAttrs.length === 0) return null;

  return (
    <AnimatePresence mode="wait">
      {loading && (
        <motion.div
          key="loading"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="flex items-center gap-2 text-textSecondary text-sm mt-4"
        >
          <span className="w-3 h-3 rounded-full border-2 border-accent border-t-transparent animate-spin" />
          Validating dataset…
        </motion.div>
      )}

      {result && !loading && (
        <motion.div
          key="result"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35 }}
          className="mt-4 rounded-xl border overflow-hidden"
          style={{
            borderColor: result.supported
              ? "rgba(34,197,94,0.35)"
              : "rgba(245,158,11,0.35)",
            background: result.supported
              ? "rgba(34,197,94,0.06)"
              : "rgba(245,158,11,0.06)",
          }}
        >
          {/* Header */}
          <div className="flex items-start gap-3 px-5 py-4">
            {result.supported ? (
              <CheckCircle size={20} className="text-success mt-0.5 flex-shrink-0" />
            ) : (
              <AlertTriangle size={20} className="text-warning mt-0.5 flex-shrink-0" />
            )}
            <div>
              <p className={`font-semibold text-sm ${result.supported ? "text-success" : "text-warning"}`}>
                {result.supported
                  ? "Full analysis supported. Binary classification dataset detected."
                  : result.fallback_needed
                  ? `Extended mode: Fairlearn analysis will be used for ${result.target_type} target.`
                  : "Partial analysis only — some checks are unavailable."}
              </p>
              <p className="text-xs text-textSecondary mt-0.5">
                Engine: <span className="text-textPrimary font-medium">{result.engine}</span>
                &nbsp;·&nbsp;
                {result.row_count?.toLocaleString()} rows detected
              </p>
            </div>
          </div>

          {/* Row-count reliability warning */}
          {result.row_count < 200 && (
            <div className="px-5 pb-3 flex items-start gap-2">
              <Info size={14} className="text-warning mt-0.5 flex-shrink-0" />
              <p className="text-xs text-warning">
                Statistical reliability warning: {result.row_count} rows detected. Results include bootstrapped confidence intervals.
              </p>
            </div>
          )}

          {/* Warnings list */}
          {result.warnings?.length > 0 && (
            <ul className="px-5 pb-4 space-y-1.5">
              {result.warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-2">
                  <AlertTriangle size={13} className="text-warning mt-0.5 flex-shrink-0" />
                  <span className="text-xs text-warning/90">{w}</span>
                </li>
              ))}
            </ul>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
