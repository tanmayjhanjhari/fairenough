import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle, CloudUpload, File, Loader2, X } from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import client from "../../api/client";
import useAnalysisStore from "../../store/analysisStore";

// ── CSV Dropzone ───────────────────────────────────────────────────────────────
function CsvDropzone() {
  const { setSession, setStep } = useAnalysisStore();
  const sessionId = useAnalysisStore((s) => s.sessionId);
  const filename  = useAnalysisStore((s) => s.filename);
  const rowCount  = useAnalysisStore((s) => s.rowCount);
  const columns   = useAnalysisStore((s) => s.columns);
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const currentModelId = useAnalysisStore.getState().modelId;
      if (currentModelId) {
        form.append("model_id", currentModelId);
      }
      const { data } = await client.post("/api/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSession(data.session_id, data);
      setStep(1);
      toast.success(`Uploaded "${data.filename}" — ${data.row_count.toLocaleString()} rows`, {
        style: { background: "#1E293B", color: "#F1F5F9", border: "1px solid rgba(34,197,94,0.4)" },
        iconTheme: { primary: "#22C55E", secondary: "#F1F5F9" },
      });
    } catch {
      /* interceptor handles toast */
    } finally {
      setLoading(false);
    }
  }, [setSession, setStep]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
      'application/octet-stream': ['.parquet', '.data', '.gz'],
      'application/zip': ['.zip'],
      'text/plain': ['.txt', '.data']
    },
    multiple: false,
    disabled: loading || !!sessionId,
  });

  // ── Already uploaded → show info card ────────────────────────────────────
  if (sessionId) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4 bg-success/10 border border-success/30 rounded-xl px-5 py-4"
      >
        <CheckCircle size={22} className="text-success flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-textPrimary truncate">{filename}</p>
          <p className="text-xs text-textSecondary mt-0.5">
            {rowCount.toLocaleString()} rows · {columns.length} columns
          </p>
        </div>
        <div className="flex gap-2">
          <span className="px-2.5 py-0.5 rounded-full bg-success/20 text-success text-xs font-medium">
            {rowCount.toLocaleString()} rows
          </span>
          <span className="px-2.5 py-0.5 rounded-full bg-accent/20 text-accent text-xs font-medium">
            {columns.length} cols
          </span>
        </div>
      </motion.div>
    );
  }

  // ── Dropzone ──────────────────────────────────────────────────────────────
  return (
    <motion.div
      {...getRootProps()}
      animate={isDragActive
        ? { scale: 1.02, borderColor: "#14B8A6" }
        : { scale: 1,    borderColor: "rgba(148,163,184,0.2)" }
      }
      transition={{ duration: 0.2 }}
      className="relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center
                 bg-surface/40 hover:bg-surface/60 transition-colors duration-200 overflow-hidden"
    >
      <input {...getInputProps()} />

      {/* Animated gradient glow — always on */}
      <motion.div
        className="absolute inset-0 rounded-2xl pointer-events-none"
        animate={{ opacity: isDragActive ? 1 : 0.4 }}
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(20,184,166,0.15) 0%, transparent 70%)",
        }}
      />

      {loading ? (
        <div className="flex flex-col items-center gap-3">
          <Loader2 size={40} className="text-accent animate-spin" />
          <p className="text-textSecondary text-sm">Uploading and analysing columns…</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <motion.div
            animate={isDragActive ? { y: -6, scale: 1.15 } : { y: 0, scale: 1 }}
            transition={{ duration: 0.25 }}
          >
            <CloudUpload size={44} className={isDragActive ? "text-accent" : "text-textSecondary"} />
          </motion.div>
          <div>
            <p className="mt-4 text-sm font-medium text-textPrimary">
              {isDragActive ? "Drop to upload..." : "Drop your dataset here — CSV, Excel, JSON, ZIP and more supported"}
            </p>
            <p className="mt-1 text-xs text-textSecondary">
              or click to browse from your computer
            </p>

            {/* Formats badges */}
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-accent/20 text-accent border border-accent/30">
                CSV
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-surface border border-white/10 text-textSecondary">
                Excel
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-surface border border-white/10 text-textSecondary">
                JSON
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-surface border border-white/10 text-textSecondary">
                TSV
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-surface border border-white/10 text-textSecondary">
                ZIP
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-surface border border-white/10 text-textSecondary">
                Parquet
              </span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

// ── Model Dropzone ─────────────────────────────────────────────────────────────
function ModelDropzone() {
  const sessionId = useAnalysisStore((s) => s.sessionId);
  const modelId = useAnalysisStore((s) => s.modelId);
  const modelInfo = useAnalysisStore((s) => s.modelInfo);
  const { setModel } = useAnalysisStore();
  const [loading, setLoading]   = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      if (sessionId) form.append("session_id", sessionId);
      const url = sessionId ? `/api/upload-model?session_id=${encodeURIComponent(sessionId)}` : "/api/upload-model";
      const { data } = await client.post(url, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setModel(data.model_id, data);
      toast.success(`Model loaded: ${data.model_type}`);
    } catch {
      /* interceptor handles toast */
    } finally {
      setLoading(false);
    }
  }, [sessionId, setModel]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/octet-stream": [".pkl", ".joblib"],
      "application/x-pkl": [".pkl"],
    },
    multiple: false,
    disabled: loading || !!modelInfo || !!modelId,
  });

  return (
    <div className="mt-4">
      <p className="section-label">ML Model (optional)</p>

      {(modelInfo || modelId) ? (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 bg-accent2/10 border border-accent2/30 rounded-xl px-4 py-3"
        >
          <File size={18} className="text-accent2 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <span className="text-sm font-medium text-textPrimary truncate block">
              {modelInfo?.filename || modelInfo?.model_type || "ML Model Attached"}
            </span>
            <span className="text-xs text-textSecondary">
              {modelInfo?.model_type || "Trained Estimator"}
            </span>
          </div>
          <span className="px-2 py-0.5 rounded-full bg-accent2/20 text-accent2 text-xs font-medium">
            {modelInfo?.n_features ?? "?"} features
          </span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setModel(null, null);
            }}
            className="p-1 hover:bg-white/10 rounded-lg text-textSecondary hover:text-textPrimary transition-colors"
            title="Remove model"
          >
            <X size={15} />
          </button>
        </motion.div>
      ) : (
        <motion.div
          {...getRootProps()}
          animate={isDragActive ? { borderColor: "#14B8A6" } : { borderColor: "rgba(148,163,184,0.15)" }}
          className="cursor-pointer rounded-xl border-2 border-dashed px-6 py-5 text-center
                     bg-surface/30 hover:bg-surface/50 transition-colors"
        >
          <input {...getInputProps()} />
          {loading ? (
            <Loader2 size={20} className="text-accent animate-spin mx-auto" />
          ) : (
            <p className="text-sm text-textSecondary">
              Drop <span className="text-textPrimary font-medium">.pkl</span> or{" "}
              <span className="text-textPrimary font-medium">.joblib</span> model file here (optional)
            </p>
          )}
        </motion.div>
      )}
    </div>
  );
}

// ── Combined export ───────────────────────────────────────────────────────────
export default function UploadZone() {
  return (
    <div className="glass-card p-6">
      <p className="section-label">Dataset</p>
      <CsvDropzone />
      <ModelDropzone />
    </div>
  );
}
