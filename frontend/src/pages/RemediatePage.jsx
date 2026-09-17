import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";
import toast from "react-hot-toast";

import PageWrapper from "../components/Layout/PageWrapper";
import useAnalysisStore from "../store/analysisStore";
import client from "../api/client";

import TechniqueCard from "../components/Mitigation/TechniqueCard";
import BeforeAfterChart from "../components/Mitigation/BeforeAfterChart";
import TradeoffChart from "../components/Mitigation/TradeoffChart";
import BiasCopilot from "../components/Copilot/BiasCopilot";

export default function RemediatePage() {
  const navigate = useNavigate();
  const store = useAnalysisStore();
  const { sessionId, targetCol, sensitiveAttrs, mitigation, setMitigation, setStep } = store;

  const [loading, setLoading] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [activeAttr, setActiveAttr] = useState(sensitiveAttrs[0] || "");
  const [simulatedAttrs, setSimulatedAttrs] = useState({});

  useEffect(() => {
    if (!sessionId || !targetCol || sensitiveAttrs.length === 0) {
      navigate("/analyze");
      return;
    }

    // Run mitigation on mount if we don't have results yet
    if (!mitigation) {
      runMitigation(activeAttr);
    }
  }, [sessionId, targetCol, sensitiveAttrs, navigate, mitigation]);

  const runMitigation = async (attr, forceSimulate = false) => {
    let cancelled = false;
    setLoading(true);
    try {
      const shouldSimulate = forceSimulate || Boolean(simulatedAttrs[attr]);
      const { data } = await client.post("/api/mitigate", {
        session_id: sessionId,
        target_col: targetCol,
        sensitive_attr: attr,
        model_id: store.modelId || undefined,
        simulate_threshold: shouldSimulate,
      });
      if (!cancelled) {
        setMitigation(data);
        setStep(3);
      }
    } catch {
      // toast handled by interceptor
    } finally {
      if (!cancelled) setLoading(false);
    }
    return () => { cancelled = true; };
  };

  const handleRunSimulation = async () => {
    setSimulating(true);
    setSimulatedAttrs(prev => ({ ...prev, [activeAttr]: true }));
    try {
      await runMitigation(activeAttr, true);
      toast.success("Simulation generated for Threshold Adjustment");
    } catch {
      /* interceptor */
    } finally {
      setSimulating(false);
    }
  };

  const handleAttrChange = (attr) => {
    setActiveAttr(attr);
    runMitigation(attr);
  };

  if (!sessionId) return null;

  return (
    <PageWrapper>
      <div className="mb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-textPrimary mb-1">Bias Mitigation</h1>
          <p className="text-textSecondary text-sm">
            Compare automated mitigation strategies and their trade-offs.
          </p>
        </div>
        
        <button 
          onClick={() => {
            setStep(4);
            navigate("/report-view");
          }}
          className="btn-primary"
          disabled={loading || !mitigation}
        >
          Generate Report
        </button>
      </div>

      {sensitiveAttrs.length > 1 && (
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-textSecondary uppercase tracking-wider mr-2">Target Attribute:</span>
          {sensitiveAttrs.map(attr => (
            <button
              key={attr}
              onClick={() => handleAttrChange(attr)}
              disabled={loading}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 border disabled:opacity-50
                ${activeAttr === attr 
                  ? "bg-accent/20 text-accent border-accent/40" 
                  : "bg-surface text-textSecondary border-white/10 hover:border-white/30"
                }`}
            >
              {attr}
            </button>
          ))}
        </div>
      )}

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-20"
          >
            <div className="w-64">
              <div className="flex justify-between text-sm text-textSecondary mb-2">
                <span>Running Mitigation</span>
                <Loader2 size={16} className="animate-spin text-accent" />
              </div>
              <div className="h-2 w-full bg-surface rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-accent"
                  initial={{ width: "0%" }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 2.5, ease: "easeOut", repeat: Infinity }}
                />
              </div>
              <p className="text-xs text-textSecondary text-center mt-3">
                Applying Reweighing and evaluating Threshold Adjustment...
              </p>
            </div>
          </motion.div>
        ) : mitigation ? (
          <motion.div
            key="content"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            {/* ── Technique Cards ────────────────────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TechniqueCard 
                name="reweigh" 
                data={mitigation.reweigh} 
                isWinner={mitigation.winner === "reweigh"} 
                winnerReason={mitigation.winner === "reweigh" ? mitigation.winner_reason : null}
              />
              <TechniqueCard 
                name="threshold" 
                data={mitigation.threshold} 
                isWinner={mitigation.winner === "threshold"} 
                winnerReason={mitigation.winner === "threshold" ? mitigation.winner_reason : null}
                onRunSimulation={handleRunSimulation}
                isSimulating={simulating}
              />
            </div>

            {/* ── Charts ─────────────────────────────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card p-6 border border-white/[0.06]">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-textPrimary">Fairness Improvement</h3>
                  <p className="text-sm text-textSecondary">Comparing fairness metrics before and after mitigation across parity differences and disparate impact ratio.</p>
                </div>
                <BeforeAfterChart mitigation={mitigation} />
              </div>

              <div className="glass-card p-6 border border-white/[0.06]">
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-textPrimary">Performance Trade-off</h3>
                  <p className="text-sm text-textSecondary">Analyzing the cost of fairness in terms of retained accuracy.</p>
                </div>
                <TradeoffChart mitigation={mitigation} />
              </div>
            </div>

          </motion.div>
        ) : null}
      </AnimatePresence>

      <BiasCopilot />
    </PageWrapper>
  );
}
