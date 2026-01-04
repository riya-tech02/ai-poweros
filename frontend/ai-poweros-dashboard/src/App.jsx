import { useEffect, useState } from "react";

const API = "https://ai-poweros.onrender.com";

export default function App() {
  const [status, setStatus] = useState("checking...");
  const [activeApp, setActiveApp] = useState("home");

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setStatus(d.status || "unknown"))
      .catch(() => setStatus("offline"));
  }, []);

  return (
    <div style={styles.os}>
      {/* Top Bar */}
      <div style={styles.topbar}>
        <div>🧠 AI-PowerOS</div>
        <div style={styles.topRight}>
          <span>Status: <b>{status}</b></span>
          <span>{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Body */}
      <div style={styles.body}>
        {/* Dock */}
        <div style={styles.dock}>
          {[
            ["home", "🏠"],
            ["routine", "🔮"],
            ["schedule", "🗓"],
            ["memory", "🧠"],
            ["logs", "📊"],
            ["settings", "⚙️"],
          ].map(([key, icon]) => (
            <button
              key={key}
              onClick={() => setActiveApp(key)}
              style={{
                ...styles.dockBtn,
                background: activeApp === key ? "#0ea5e9" : "transparent",
              }}
            >
              {icon}
            </button>
          ))}
        </div>

        {/* Workspace */}
        <div style={styles.workspace}>
          {activeApp === "home" && (
            <Panel title="System Overview">
              <p>Backend: {API}</p>
              <p>Health: {status}</p>
              <p>Version: 1.0.0</p>
            </Panel>
          )}

          {activeApp === "routine" && (
            <Panel title="Routine Predictor">
              <p>Endpoint:</p>
              <code>/api/v1/predict/routine</code>
            </Panel>
          )}

          {activeApp === "schedule" && (
            <Panel title="Intelligent Scheduler">
              <p>Endpoint:</p>
              <code>/api/v1/advanced/schedule/intelligent</code>
            </Panel>
          )}

          {activeApp === "memory" && (
            <Panel title="Memory Graph">
              <p>Graph + Vector DB status</p>
            </Panel>
          )}

          {activeApp === "logs" && (
            <Panel title="System Logs">
              <p>Live logs coming soon…</p>
            </Panel>
          )}

          {activeApp === "settings" && (
            <Panel title="Settings">
              <p>Environment: Production</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div style={styles.panel}>
      <h2>{title}</h2>
      <div>{children}</div>
    </div>
  );
}

const styles = {
  os: {
    background: "#020617",
    color: "white",
    minHeight: "100vh",
    fontFamily: "system-ui",
  },
  topbar: {
    height: 48,
    background: "#020617",
    borderBottom: "1px solid #0f172a",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 16px",
  },
  topRight: {
    display: "flex",
    gap: 16,
    opacity: 0.9,
  },
  body: {
    display: "flex",
    height: "calc(100vh - 48px)",
  },
  dock: {
    width: 64,
    background: "#020617",
    borderRight: "1px solid #0f172a",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    paddingTop: 12,
    gap: 10,
  },
  dockBtn: {
    width: 44,
    height: 44,
    borderRadius: 10,
    border: "none",
    color: "white",
    fontSize: 20,
    cursor: "pointer",
  },
  workspace: {
    flex: 1,
    padding: 24,
    overflow: "auto",
  },
  panel: {
    background: "#020617",
    border: "1px solid #0f172a",
    borderRadius: 14,
    padding: 24,
    maxWidth: 900,
  },
};
