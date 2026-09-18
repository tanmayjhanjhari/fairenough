import { create } from "zustand";

const useAnalysisStore = create((set, get) => ({
  // ── Session ─────────────────────────────────────────────────────────────────
  sessionId: null,
  modelId: null,
  hasRealModel: false,
  modelType: null,
  modelInfo: null,

  // ── Dataset metadata ─────────────────────────────────────────────────────────
  columns: [],
  dtypes: {},
  numericCols: [],
  categoricalCols: [],
  preview: [],
  rowCount: 0,
  filename: "",
  preprocessingReport: null,
  suggestedSensitive: [],
  blockedSensitive: [],

  // ── Configuration ─────────────────────────────────────────────────────────────
  targetCol: "",
  sensitiveAttrs: [],

  // ── Scenario ──────────────────────────────────────────────────────────────────
  scenario: null,
  scenarioConfidence: null,
  scenarioReason: "",

  // ── Validation ────────────────────────────────────────────────────────────────
  validation: null,

  // ── Analysis results ──────────────────────────────────────────────────────────
  metrics: null,
  auditScore: null,
  grade: null,
  overallSeverity: null,

  // ── Explanations ──────────────────────────────────────────────────────────────
  explanation: null,
  geminiExplanations: {},

  // ── Mitigation ────────────────────────────────────────────────────────────────
  mitigation: null,

  // ── Copilot chat ──────────────────────────────────────────────────────────────
  geminiHistory: [],

  // ── UI state ──────────────────────────────────────────────────────────────────
  step: 0,          // 0=Upload 1=Configure 2=Analyze 3=Remediate 4=Report
  isLoading: false,
  error: null,

  // ── Actions ───────────────────────────────────────────────────────────────────

  setSession: (sessionId, meta = {}) => {
    const prevModelId = get().modelId;
    const effModelId = meta.model_id || prevModelId || null;
    const effHasRealModel = Boolean(meta.has_real_model || meta.model_id || (prevModelId && get().hasRealModel));
    return set({
      sessionId,
      modelId: effModelId,
      hasRealModel: effHasRealModel,
      columns:        meta.columns        ?? [],
      dtypes:         meta.dtypes         ?? {},
      numericCols:    meta.numeric_cols   ?? [],
      categoricalCols: meta.categorical_cols ?? [],
      preview:        meta.preview        ?? [],
      rowCount:       meta.row_count      ?? 0,
      filename:       meta.filename       ?? "",
      preprocessingReport: meta.preprocessing_report ?? null,
      scenario:            meta.scenario ?? meta.preprocessing_report?.detected_scenario ?? null,
      suggestedSensitive:  meta.suggested_sensitive  ?? [],
      blockedSensitive:    meta.blocked_from_sensitive ?? [],
    });
  },

  setModel: (modelId, meta = null) =>
    set({
      modelId,
      hasRealModel: Boolean(modelId),
      modelInfo: meta ?? (modelId ? (get().modelInfo || { model_id: modelId }) : null),
      modelType: meta?.model_type ?? get().modelType ?? null,
    }),

  setColumns: (columns) => set({ columns }),

  setTarget: (targetCol) => set({ targetCol }),

  setSensitive: (sensitiveAttrs) => set({ sensitiveAttrs }),

  setScenario: (data) =>
    set({
      scenario:           data.scenario           ?? null,
      scenarioConfidence: data.confidence_pct     ?? null,
      scenarioReason:     data.reason             ?? "",
    }),

  setValidation: (validation) => set({ validation }),

  setMetrics: (data) =>
    set({
      metrics:         data.metrics_per_attr  ?? null,
      auditScore:      data.audit_score       ?? null,
      grade:           data.grade             ?? null,
      overallSeverity: data.overall_severity  ?? null,
      scenario:        (get().scenario && get().scenario !== "other") ? get().scenario : (data.scenario ?? get().scenario),
      validation:      data.validation        ?? get().validation,
      modelId:         data.model_id          ?? get().modelId,
      hasRealModel:    data.has_real_model    ?? data.model_used ?? get().hasRealModel,
      modelType:       data.model_type        ?? get().modelType,
    }),

  setExplanation: (explanation) => set({ explanation }),

  setGeminiExplanation: (attr, text) =>
    set((state) => ({
      geminiExplanations: { ...state.geminiExplanations, [attr]: text },
    })),

  setMitigation: (mitigation) =>
    set({
      mitigation,
      modelId: mitigation?.model_id ?? get().modelId,
      hasRealModel: (mitigation?.has_real_model ?? get().hasRealModel),
    }),

  addGeminiMessage: (role, content) =>
    set((state) => ({
      geminiHistory: [...state.geminiHistory, { role, content }],
    })),

  setStep: (step) => set({ step }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error }),

  reset: () =>
    set({
      sessionId: null,
      modelId: null,
      hasRealModel: false,
      modelType: null,
      modelInfo: null,
      columns: [],
      dtypes: {},
      numericCols: [],
      categoricalCols: [],
      preview: [],
      rowCount: 0,
      filename: "",
      targetCol: "",
      sensitiveAttrs: [],
      scenario: null,
      scenarioConfidence: null,
      scenarioReason: "",
      validation: null,
      metrics: null,
      auditScore: null,
      grade: null,
      overallSeverity: null,
      explanation: null,
      geminiExplanations: {},
      mitigation: null,
      geminiHistory: [],
      step: 0,
      isLoading: false,
      error: null,
    }),
}));

export default useAnalysisStore;
