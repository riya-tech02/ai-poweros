import { useEffect, useState } from "react";

const API = "https://ai-poweros.onrender.com";

export default function App() {
  const [status, setStatus] = useState("checking...");
  const [routine, setRoutine] = useState(null);

  const [active, setActive] = useState("home");

useEffect(() => {
  const checkHealth = () => {
    fetch(`${API}/health`)
      .then(res => res.json())
      .then(data => {
        console.log("Health response:", data);
        setStatus(data.status || "unknown");
      })
      .catch(() => setStatus("offline"));
  };

  checkHealth();                 // run immediately
  const id = setInterval(checkHealth, 5000); // refresh every 5s

  return () => clearInterval(id);
  fetch(`${API}/api/v1/predict/routine`)
  .then(r => r.json())
  .then(d => setRoutine(d))
  .catch(() => setRoutine({ error: "failed" }));

}, []);


  return (
    <div className="os">
      {/* Top Bar */}
      <div className="topbar">
        <div className="brand">🧠 AI-PowerOS</div>
        <div className="right">
          <span className={`status ${status}`}>● {status}</span>
          <span className="time">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Body */}
      <div className="body">
        {/* Dock */}
        <div className="dock">
          {[
            ["home", "🏠"],
            ["routine", "🔮"],
            ["schedule", "🗓"],
            ["memory", "🧠"],
            ["logs", "📊"],
            ["settings", "⚙️"],
          ].map(([k, icon]) => (
            <button
              key={k}
              className={active === k ? "dock-btn active" : "dock-btn"}
              onClick={() => setActive(k)}
            >
              {icon}
            </button>
          ))}
        </div>

        {/* Workspace */}
        <div className="workspace">
          <div className="window">
            <h2>{active.toUpperCase()}</h2>

            {active === "home" && (
              <>
                <p><b>Backend</b></p>
                <code>{API}</code>
                <p style={{ marginTop: 12 }}>
                  System Status: <b>{status}</b>
                </p>
              </>
            )}

{active === "home" && (
  <>
    <p><b>Backend</b></p>
    <code>{API}</code>

    <p style={{ marginTop: 12 }}>
      System Status: <b>{status}</b>
    </p>

    <p style={{ marginTop: 12 }}>
      <b>Routine Prediction</b>
    </p>

    <pre style={{
      background: "#020617",
      border: "1px solid #0f172a",
      padding: 12,
      borderRadius: 8,
      marginTop: 6,
      fontSize: 12
    }}>
      {routine ? JSON.stringify(routine, null, 2) : "Loading..."}
    </pre>
  </>
)}
          </div>
        </div>
      </div>
    </div>
  );
}
