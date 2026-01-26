import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = import.meta.env.PROD 
  ? 'https://ai-poweros.onrender.com' 
  : '';

function App() {
  const [activeWindow, setActiveWindow] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { role: 'ai', content: 'Hi! I\'m your AI-PowerOS assistant. How can I help you today?' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    checkHealth();
    return () => clearInterval(timer);
  }, []);

  const checkHealth = async () => {
    try {
      const { data } = await axios.get(`${API_URL}/health`);
      setSystemStatus(data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const testPrediction = async () => {
    try {
      const { data } = await axios.post(`${API_URL}/api/v1/advanced/routine/advanced`, {
        user_id: 'demo',
        recent_activities: ['coding', 'coffee'],
        context: { time_of_day: 'afternoon' },
        top_k: 3
      });
      alert(`Prediction Success!\nLatency: ${data.latency_ms}ms\nTop: ${data.predictions[0].activity} (${(data.predictions[0].confidence * 100).toFixed(1)}%)`);
    } catch (error) {
      alert('Error: ' + error.message);
    }
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    
    setChatMessages([...chatMessages, { role: 'user', content: chatInput }]);
    setChatInput('');
    
    setTimeout(() => {
      setChatMessages(prev => [...prev, { 
        role: 'ai', 
        content: 'I can help you with predictions, scheduling, and habit tracking. Try the Dashboard for a quick test!' 
      }]);
    }, 1000);
  };

  return (
    <div className="os-container">
      {/* Menu Bar */}
      <div className="menu-bar">
        <div className="menu-left">
          <div className="menu-logo">
            <i className="fas fa-robot"></i>
            <span>AI-PowerOS</span>
          </div>
          <div className="menu-items">
            <div className="menu-item" onClick={() => setActiveWindow('dashboard')}>Dashboard</div>
            <div className="menu-item" onClick={() => setActiveWindow('assistant')}>AI Assistant</div>
            <div className="menu-item" onClick={() => setActiveWindow('settings')}>Settings</div>
          </div>
        </div>
        <div className="menu-right">
          <div className="system-icons">
            <i className="fas fa-wifi"></i>
            <i className="fas fa-battery-full"></i>
          </div>
          <div className="clock">
            {time.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
          </div>
        </div>
      </div>

      {/* Desktop */}
      <div className="desktop">
        {/* Dashboard Window */}
        {activeWindow === 'dashboard' && (
          <div className="window dashboard-window">
            <div className="window-titlebar">
              <div className="window-title">
                <i className="fas fa-chart-line"></i> Dashboard
              </div>
              <div className="window-controls">
                <div className="window-btn close" onClick={() => setActiveWindow(null)}></div>
              </div>
            </div>
            <div className="window-content">
              <div className="dashboard">
                <h2>System Overview</h2>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Prediction Accuracy</div>
                    <div className="stat-value">87.3%</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Response Time</div>
                    <div className="stat-value">2.1ms</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Tasks Completed</div>
                    <div className="stat-value">1,247</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">System Status</div>
                    <div className="stat-value" style={{fontSize: '1.5em'}}>
                      {systemStatus?.status || 'Loading...'}
                    </div>
                  </div>
                </div>
                <h3>Quick Actions</h3>
                <div className="actions">
                  <button className="btn" onClick={testPrediction}>
                    <i className="fas fa-magic"></i> Test Prediction
                  </button>
                  <button className="btn" onClick={() => window.open('https://ai-poweros.onrender.com/docs')}>
                    <i className="fas fa-book"></i> API Docs
                  </button>
                  <button className="btn" onClick={checkHealth}>
                    <i className="fas fa-heartbeat"></i> System Health
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Assistant Window */}
        {activeWindow === 'assistant' && (
          <div className="window assistant-window">
            <div className="window-titlebar">
              <div className="window-title">
                <i className="fas fa-brain"></i> AI Assistant
              </div>
              <div className="window-controls">
                <div className="window-btn close" onClick={() => setActiveWindow(null)}></div>
              </div>
            </div>
            <div className="window-content">
              <div className="chat-container">
                <div className="chat-messages">
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`message ${msg.role}`}>
                      <div className="message-bubble">{msg.content}</div>
                    </div>
                  ))}
                </div>
                <div className="chat-input-container">
                  <input
                    type="text"
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask me anything..."
                  />
                  <button className="chat-send" onClick={sendMessage}>Send</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Settings Window */}
        {activeWindow === 'settings' && (
          <div className="window settings-window">
            <div className="window-titlebar">
              <div className="window-title">
                <i className="fas fa-cog"></i> Settings
              </div>
              <div className="window-controls">
                <div className="window-btn close" onClick={() => setActiveWindow(null)}></div>
              </div>
            </div>
            <div className="window-content">
              <div className="settings">
                <h3>AI Features</h3>
                <div className="setting-item">
                  <span>Automatic Predictions</span>
                  <div className="toggle active"></div>
                </div>
                <div className="setting-item">
                  <span>Smart Scheduling</span>
                  <div className="toggle active"></div>
                </div>
                <h3>Privacy</h3>
                <div className="setting-item">
                  <span>On-Device Processing</span>
                  <div className="toggle active"></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Dock */}
        <div className="dock">
          <div className="dock-item" onClick={() => setActiveWindow('dashboard')} title="Dashboard">
            <i className="fas fa-chart-line"></i>
          </div>
          <div className="dock-item" onClick={() => setActiveWindow('assistant')} title="AI Assistant">
            <i className="fas fa-brain"></i>
          </div>
          <div className="dock-item" onClick={() => setActiveWindow('settings')} title="Settings">
            <i className="fas fa-cog"></i>
          </div>
          <div className="dock-item" onClick={() => window.open('https://ai-poweros.onrender.com/docs')} title="API Docs">
            <i className="fas fa-book"></i>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
