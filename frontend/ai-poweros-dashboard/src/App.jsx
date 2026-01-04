import { useEffect, useState } from "react";

const API = "https://ai-poweros.onrender.com";

export default function App() {
  const [user, setUser] = useState(null);
  const [pos, setPos] = useState({ x: 100, y: 100 });

  const [status, setStatus] = useState("checking...");
  const [routine, setRoutine] = useState(null);
  const [schedule, setSchedule] = useState(null);
  const [memory, setMemory] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const [active, setActive] = useState("home");

  /* ---------- HEALTH POLLING ---------- */
  useEffect(() => {
    const checkHealth = () => {
      fetch(`${API}/health`)
        .then(r => r.json())
        .then(d => setStatus(d.status || "unknown"))
        .catch(() => setStatus("offline"));
    };

    checkHealth();
    const id = setInterval(checkHealth, 5000);
    return () => clearInterval(id);
  }, []);

  /* ---------- ROUTINE (ONCE) ---------- */
  useEffect(() => {
    fetch(`${API}/api/v1/predict/routine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user?.id || "guest",
        context: {},
        top_k: 5
      })
    })
      .then(r => r.json())
      .then(setRoutine)
      .catch(() => setRoutine({ error: "failed" }));
  }, [user]);

  /* ---------- SCHEDULE ---------- */
  useEffect(() => {
    if (active !== "schedule") return;

    fetch(`${API}/api/v1/advanced/schedule/intelligent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user?.id || "guest",
        horizon: "day"
      })
    })
      .then(r => r.json())
      .then(setSchedule)
      .catch(() => setSchedule({ error: "failed" }));
  }, [active, user]);

  /* ---------- MEMORY ---------- */
  useEffect(() => {
    if (active !== "memory") return;

    fetch(`${API}/api/v1/advanced/habits/record`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user?.id || "guest",
        action: "summary"
      })
    })
      .then(r => r.json())
      .then(setMemory)
      .catch(() => setMemory({ error: "failed" }));
  }, [active, user]);

  /* ---------- METRICS ---------- */
  useEffect(() => {
    if (active !== "logs") return;

    fetch(`${API}/health`)
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => setMetrics({ error: "failed" }));
  }, [active]);

  return (
    <div className="os">
      <div className="topbar">
        <div className="brand">🧠 AI-PowerOS</div>
        <div className="right">
          <span className={`status ${status}`}>● {status}</span>
          <span className="time">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      <div className="body">
        <div className="dock">
          {["home","routine","schedule","memory","logs","settings"].map((k,i)=>(
            <button key={k} onClick={()=>setActive(k)}>
              {["🏠","🔮","🗓","🧠","📊","⚙️"][i]}
            </button>
          ))}
        </div>

        <div className="workspace">
          <div
            className="window"
            style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }}
            onMouseDown={(e)=>{
              const sx=e.clientX-pos.x, sy=e.clientY-pos.y;
              const move=(ev)=>setPos({x:ev.clientX-sx,y:ev.clientY-sy});
              document.addEventListener("mousemove", move);
              document.addEventListener("mouseup", ()=>document.removeEventListener("mousemove", move), {once:true});
            }}
          >
            <h2>{active.toUpperCase()}</h2>

            {!user && (
              <button onClick={() => setUser({ id: "demo_user", role: "admin" })}>
                Login as Demo User
              </button>
            )}

            {active === "home" && (
              <>
                <p><b>Backend</b></p>
                <code>{API}</code>
                <pre>{routine ? JSON.stringify(routine,null,2) : "Loading routine..."}</pre>
              </>
            )}

            {active === "schedule" && <pre>{JSON.stringify(schedule,null,2)}</pre>}
            {active === "memory" && <pre>{JSON.stringify(memory,null,2)}</pre>}
            {active === "logs" && <pre>{JSON.stringify(metrics,null,2)}</pre>}

            <button
              onClick={()=>{
                fetch(`${API}/api/v1/agents/run`,{
                  method:"POST",
                  headers:{ "Content-Type":"application/json" },
                  body:JSON.stringify({
                    agent:"routine_optimizer",
                    user_id:user?.id || "guest"
                  })
                })
              }}
            >
              ▶ Run AI Agent
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
