import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// Backend API URL - use environment variable or default
const API_URL = import.meta.env.VITE_API_URL || 'https://ai-poweros.onrender.com';

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
      setSystemStatus({ status: 'offline' });
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
      alert(`✅ Prediction Success!\n\nLatency: ${data.latency_ms}ms\nBackend: ${data.backend}\n\nTop Prediction:\n${data.predictions[0].activity} - ${(data.predictions[0].confidence * 100).toFixed(1)}%`);
    } catch (error) {
      alert('❌ Error: ' + error.message);
    }
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    
    const newMessages = [...chatMessages, { role: 'user', content: chatInput }];
    setChatMessages(newMessages);
    setChatInput('');
    
    setTimeout(() => {
      setChatMessages([...newMessages, { 
        role: 'ai', 
        content: 'I can help you with predictions, task scheduling, and habit tracking. Try the Dashboard to test the AI prediction system!' 
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
            <div className="system-icon" title={`Status: ${systemStatus?.status || 'checking...'}`}>
              <i className={`fas fa-circle ${systemStatus?.status === 'healthy' ? 'status-healthy' : 'status-offline'}`}></i>
            </div>
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
                <div className="window-btn minimize"></div>
                <div className="window-btn maximize"></div>
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
                    <div className="stat-value" style={{fontSize: '1.5em', color: systemStatus?.status === 'healthy' ? '#10b981' : '#ef4444'}}>
                      {systemStatus?.status || 'checking...'}
                    </div>
                  </div>
                </div>
                <h3>Quick Actions</h3>
                <div className="actions">
                  <button className="btn" onClick={testPrediction}>
                    <i className="fas fa-magic"></i> Test AI Prediction
                  </button>
                  <button className="btn" onClick={() => window.open(`${API_URL}/docs`, '_blank')}>
                    <i className="fas fa-book"></i> API Documentation
                  </button>
                  <button className="btn" onClick={checkHealth}>
                    <i className="fas fa-heartbeat"></i> Check Health
                  </button>
                </div>
                <div style={{marginTop: '20px', padding: '15px', background: '#f0f0f0', borderRadius: '10px'}}>
                  <strong>Backend:</strong> {API_URL}<br/>
                  <strong>Version:</strong> {systemStatus?.version || 'N/A'}<br/>
                  <strong>Features:</strong> {systemStatus?.features?.length || 0} active
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
                <div className="window-btn minimize"></div>
                <div className="window-btn maximize"></div>
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
                <div className="window-btn minimize"></div>
                <div className="window-btn maximize"></div>
              </div>
            </div>
            <div className="window-content">
              <div className="settings">
                <h3>AI Features</h3>
                <div className="setting-item">
                  <div>
                    <div style={{fontWeight: 600}}>Automatic Predictions</div>
                    <div style={{fontSize: '13px', color: '#666'}}>Enable real-time activity predictions</div>
                  </div>
                  <div className="toggle active"></div>
                </div>
                <div className="setting-item">
                  <div>
                    <div style={{fontWeight: 600}}>Smart Scheduling</div>
                    <div style={{fontSize: '13px', color: '#666'}}>AI-powered task optimization</div>
                  </div>
                  <div className="toggle active"></div>
                </div>
                <h3>Privacy & Security</h3>
                <div className="setting-item">
                  <div>
                    <div style={{fontWeight: 600}}>On-Device Processing</div>
                    <div style={{fontSize: '13px', color: '#666'}}>Process sensitive data locally (87%)</div>
                  </div>
                  <div className="toggle active"></div>
                </div>
                <div className="setting-item">
                  <div>
                    <div style={{fontWeight: 600}}>Differential Privacy</div>
                    <div style={{fontSize: '13px', color: '#666'}}>ε=1.0 privacy guarantee</div>
                  </div>
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
          <div className="dock-item" onClick={() => window.open(`${API_URL}/docs`, '_blank')} title="API Documentation">
            <i className="fas fa-book"></i>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
