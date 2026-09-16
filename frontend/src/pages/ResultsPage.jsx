import { AlertTriangle, Info } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import PageWrapper from "../components/Layout/PageWrapper";
import useAnalysisStore from "../store/analysisStore";
import client from "../api/client";

import AuditScoreGauge from "../components/Dashboard/AuditScoreGauge";
import ValidationBanner from "../components/Dashboard/ValidationBanner";
import MetricCards from "../components/Dashboard/MetricCards";
import GroupBarChart from "../components/Dashboard/GroupBarChart";
import FairnessRadar from "../components/Dashboard/FairnessRadar";
import ExplainerPanel from "../components/Dashboard/ExplainerPanel";
import BiasCopilot from "../components/Copilot/BiasCopilot";

export default function ResultsPage() {
  const navigate = useNavigate();
  const store = useAnalysisStore();
  const { 
    sessionId, metrics, validation, auditScore, grade, sensitiveAttrs, overallSeverity,
    setExplanation, setGeminiExplanation, explanation
  } = store;

  const [activeTab, setActiveTab] = useState("Metrics");
  const [activeAttr, setActiveAttr] = useState(sensitiveAttrs[0] || "");
  const [fetchingExplanations, setFetchingExplanations] = useState(false);

  // Redirect if no session
  useEffect(() => {
    if (!sessionId || !metrics) {
      navigate("/analyze");
    }
  }, [sessionId, metrics, navigate]);

  // Fetch explanations on mount for all sensitive attributes
  useEffect(() => {
    if (!sessionId || !sensitiveAttrs.length) return;

    let cancelled = false;

    const fetchAllExplanations = async () => {
      setFetchingExplanations(true);
      
      try {
        // Run explainer parallel requests
        const explPromises = sensitiveAttrs.map(attr => 
          client.post("/api/explain", { session_id: sessionId, target_col: store.targetCol, sensitive_attr: attr })
        );
        const explResults = await Promise.allSettled(explPromises);
        
        const explanationsObj = {};
        explResults.forEach((res, i) => {
          if (res.status === "fulfilled") {
            const data = res.value.data;
            explanationsObj[sensitiveAttrs[i]] = data;
            if (data.gemini_explanation) {
              setGeminiExplanation(sensitiveAttrs[i], data.gemini_explanation);
            }
          }
        });
        
        if (!cancelled) {
          // We set the whole explanations object
          setExplanation(explanationsObj);
        }
      } catch (err) {
        console.error("Failed to fetch explanations", err);
      } finally {
        if (!cancelled) setFetchingExplanations(false);
      }
    };

    fetchAllExplanations();

    return () => { cancelled = true; };
  }, [sessionId, sensitiveAttrs, store.targetCol, setExplanation, setGeminiExplanation]);

  if (!sessionId || !metrics) return null;

  const currentMetrics = metrics[activeAttr] || {};
  const groupStats = currentMetrics.group_stats || {};

  // Compute worst/best group and gap for MetricCards "Why This Matters"
  const sortedGroups = Object.entries(groupStats).sort(
    (a, b) => (a[1].positive_rate ?? 0) - (b[1].positive_rate ?? 0)
  );
  const worstGroup = sortedGroups[0]?.[0] ?? null;
  const bestGroup  = sortedGroups[sortedGroups.length - 1]?.[0] ?? null;
  const gapPct = sortedGroups.length >= 2
    ? ((sortedGroups[sortedGroups.length - 1][1].positive_rate ?? 0) -
       (sortedGroups[0][1].positive_rate ?? 0)) * 100
    : null;

  // Plain reason from explainer for the active attribute
  const attrExplanation = explanation?.[activeAttr] || {};
  const plainReason = attrExplanation.plain_reason || attrExplanation.cause_label || null;

  return (
    <PageWrapper>
      <div className="mb-8 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-textPrimary mb-1">Analysis Results</h1>
          <p className="text-textSecondary text-sm">
            Review fairness metrics and AI-generated explanations for the detected biases.
          </p>
        </div>
        
        <button 
          onClick={() => navigate("/remediate")}
          className="btn-primary"
        >
          Proceed to Mitigation
        </button>
      </div>

      {/* ── Top Section ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-card p-6 flex items-center justify-center border border-white/[0.06]">
          <AuditScoreGauge score={auditScore} grade={grade} />
        </div>
        <div className="md:col-span-2">
          <ValidationBanner
            validation={validation}
            metricsPerAttr={metrics}
            auditScore={auditScore}
            grade={grade}
            overallSeverity={overallSeverity}
          />
        </div>
      </div>

      {/* ── Attribute Selector ─────────────────────────────────────────────── */}
      {sensitiveAttrs.length > 1 && (
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-textSecondary uppercase tracking-wider mr-2">Attribute:</span>
          {sensitiveAttrs.map(attr => (
            <button
              key={attr}
              onClick={() => setActiveAttr(attr)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 border
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

      {/* ── Middle Section: Tabs ────────────────────────────────────────────── */}
      <div className="glass-card overflow-hidden border border-white/[0.06] flex flex-col min-h-[500px]">
        {/* Tab Header */}
        <div className="flex border-b border-white/[0.06] bg-surface/50 px-2">
          {["Metrics", "Groups", "Explanation"].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-4 text-sm font-bold relative transition-colors duration-200
                ${activeTab === tab ? "text-accent" : "text-textSecondary hover:text-textPrimary"}
              `}
            >
              {tab}
              {activeTab === tab && (
                <motion.div 
                  layoutId="activeTab"
                  className="absolute bottom-0 left-0 w-full h-0.5 bg-accent"
                />
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="p-6 flex-1">
          <AnimatePresence mode="wait">
            {activeTab === "Metrics" && (
              <motion.div
                key="Metrics"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-8"
              >
                {currentMetrics.error && (
                  <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs flex items-center gap-3">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>{currentMetrics.error}</span>
                  </div>
                )}
                <MetricCards
                  metrics={currentMetrics}
                  worstGroup={worstGroup}
                  bestGroup={bestGroup}
                  gapPct={gapPct}
                  plainReason={plainReason}
                  grade={grade}
                />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center pt-4 border-t border-white/[0.06]">
                  <div>
                    <h4 className="text-sm font-semibold text-textSecondary uppercase tracking-wider mb-2">Fairness Profile</h4>
                    <p className="text-sm text-textSecondary leading-relaxed mb-4">
                      This radar chart plots the four key fairness metrics. 
                      The <span className="text-accent font-medium">teal area</span> represents your dataset's current state. 
                      The <span className="text-success border-b border-dashed border-success">green dashed line</span> represents the ideal fair threshold for each metric.
                    </p>
                  </div>
                  <FairnessRadar metrics={currentMetrics} />
                </div>
              </motion.div>
            )}

            {activeTab === "Groups" && (
              <motion.div
                key="Groups"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <div className="mb-6">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                    <h3 className="text-lg font-semibold text-textPrimary">Group Positive Rates</h3>
                    {currentMetrics.binning_applied && (
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-accent/15 text-accent border border-accent/20 font-mono">
                        Data-Driven Grouping (Median)
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-textSecondary">
                    The percentage of favorable outcomes received by each demographic group within <span className="font-medium text-textPrimary">{activeAttr}</span>.
                  </p>
                  {currentMetrics.binning_note && (
                    <p className="text-xs text-textSecondary/80 mt-1 italic">
                      <Info className="w-3.5 h-3.5 inline-block mr-1 text-accent shrink-0" />{currentMetrics.binning_note}
                    </p>
                  )}
                </div>

                {currentMetrics.error && (
                  <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs flex items-center gap-3">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>{currentMetrics.error}</span>
                  </div>
                )}

                <GroupBarChart groupStats={groupStats} />
              </motion.div>
            )}

            {activeTab === "Explanation" && (
              <motion.div
                key="Explanation"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-textPrimary mb-1">Root Cause Analysis</h3>
                  <p className="text-sm text-textSecondary">
                    Statistical analysis and AI-driven narrative explaining why bias exists for <span className="font-medium text-textPrimary">{activeAttr}</span>.
                  </p>
                </div>
                <ExplainerPanel attrName={activeAttr} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Copilot ────────────────────────────────────────────────────────── */}
      <BiasCopilot />

    </PageWrapper>
  );
}
