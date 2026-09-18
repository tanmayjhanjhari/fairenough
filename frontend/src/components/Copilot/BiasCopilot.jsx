import { AnimatePresence, motion } from "framer-motion";
import { Loader2, Send, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import toast from "react-hot-toast";
import client from "../../api/client";
import useAnalysisStore from "../../store/analysisStore";

export default function BiasCopilot() {
  const store = useAnalysisStore();
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef(null);

  const { sessionId, geminiHistory, addGeminiMessage } = store;

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [geminiHistory, isTyping, isOpen]);

  // Initial greeting when opened for the first time
  useEffect(() => {
    if (isOpen && geminiHistory.length === 0 && sessionId) {
      handleSend("I've analyzed your dataset. Here's a summary of what I found:", true);
    }
  }, [isOpen, geminiHistory.length, sessionId]);

  const handleSend = async (textOverride, isSystemInit = false) => {
    const text = textOverride || input;
    if (!text.trim() || !sessionId) return;

    if (!isSystemInit) {
      addGeminiMessage("user", text);
      setInput("");
    }
    
    setIsTyping(true);

    try {
      // Create a payload history array from store history
      const historyPayload = geminiHistory.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const contextPayload = {
        audit_score: store.auditScore,
        grade: store.grade,
        overall_severity: store.overallSeverity,
        scenario: typeof store.scenario === "string" ? store.scenario : store.scenario?.scenario || "unknown",
        target_col: store.targetCol,
        sensitive_attrs: store.sensitiveAttrs,
        metrics_per_attr: store.metrics || {},
        mitigation_winner: store.mitigation?.winner,
      };

      const { data } = await client.post("/api/gemini-chat", {
        session_id: sessionId,
        message: text,
        history: historyPayload,
        context: contextPayload,
      }, { skipToast: true });

      addGeminiMessage("model", data.reply);
    } catch {
      // interceptor handles toast, but we should add a fallback message
      addGeminiMessage("model", "I'm having trouble connecting right now. Please try again later.");
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-accent text-primary flex items-center justify-center shadow-2xl z-40 group hover:shadow-[0_0_20px_rgba(20,184,166,0.5)] transition-all duration-300"
      >
        <Sparkles size={24} className="group-hover:animate-pulse" />
        {/* Tooltip on hover */}
        <span className="absolute right-full mr-4 bg-surface text-textPrimary text-sm font-medium px-3 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-white/10 shadow-lg">
          Ask AI
        </span>
      </motion.button>

      {/* Chat Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed bottom-24 right-6 w-[380px] h-[500px] max-h-[80vh] bg-primary rounded-2xl shadow-2xl z-50 flex flex-col border border-accent/20 overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-surface border-b border-white/10">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
                  <Sparkles size={16} className="text-accent" />
                </div>
                <div>
                  <h3 className="font-bold text-textPrimary text-sm">Bias Copilot</h3>
                  <p className="text-[10px] text-textSecondary">Powered by Google Gemini</p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-textSecondary hover:text-textPrimary p-1 rounded-md hover:bg-white/5 transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Messages Area */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
              {geminiHistory.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-accent text-primary rounded-br-sm"
                        : "bg-surface text-textPrimary border border-white/5 rounded-bl-sm"
                    }`}
                  >
                    {/* Render paragraphs cleanly */}
                    {msg.content.split("\n\n").map((p, i) => (
                      <p key={i} className={i > 0 ? "mt-2" : ""}>{p}</p>
                    ))}
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-surface text-textPrimary border border-white/5 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-textSecondary animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-textSecondary animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-textSecondary animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-3 bg-surface border-t border-white/10">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSend();
                }}
                className="relative flex items-center"
              >
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={isTyping}
                  placeholder="Ask about fairness, metrics, or mitigations..."
                  className="w-full bg-primary border border-white/10 rounded-xl pl-4 pr-12 py-3 text-sm text-textPrimary placeholder-textSecondary focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/50 disabled:opacity-50 transition-all"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isTyping}
                  className="absolute right-2 p-2 rounded-lg bg-accent text-primary disabled:bg-transparent disabled:text-textSecondary hover:bg-teal-400 transition-colors"
                >
                  {isTyping ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                </button>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
