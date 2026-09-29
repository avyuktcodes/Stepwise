import React, { useState, useEffect } from 'react';
import mockData from './data/mock_patients.json';
import { Activity, Brain, Droplets, LayoutGrid } from 'lucide-react';

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [activePetView, setActivePetView] = useState('amyloid');

  useEffect(() => {
    // In a real app, this would fetch from the backend
    setPatients(mockData);
    if (mockData.length > 0) {
      setSelectedPatient(mockData[0]);
    }
  }, []);

  if (!selectedPatient) return <div className="app-container">Loading...</div>;

  const getStatusClass = (status) => {
    if (status.includes("High")) return "status-high";
    if (status.includes("MCI")) return "status-mci";
    return "status-healthy";
  };

  const renderRiskMeter = (riskValue) => {
    if (riskValue === undefined) return null;
    const pct = (riskValue * 100).toFixed(0) + '%';
    let color = 'var(--accent-blue)';
    if (riskValue > 0.75) color = 'var(--accent-red)';
    else if (riskValue > 0.40) color = 'var(--accent-orange)';
    else if (riskValue > 0) color = 'var(--accent-green)';
    
    return (
      <div className="risk-meter">
        <div className="risk-fill" style={{ width: pct, backgroundColor: color }}></div>
      </div>
    );
  };

  return (
    <div className="app-container">
      
      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>StepWise Escalation</h1>
        </div>
        <div className="patient-list">
          {patients.map(p => (
            <div 
              key={p.id} 
              className={`patient-card ${selectedPatient.id === p.id ? 'active' : ''}`}
              onClick={() => setSelectedPatient(p)}
            >
              <div className="patient-name">{p.name}</div>
              <div className="patient-meta">
                <span>{p.gender}, {p.age}</span>
                <span>Stage {p.current_stage}</span>
              </div>
              <span className={`status-badge ${getStatusClass(p.status)}`}>
                {p.status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main-content">
        <div className="dashboard-header">
          <h2>{selectedPatient.name}</h2>
          <div className="patient-demographics">
            ID: {selectedPatient.id} • {selectedPatient.gender}, {selectedPatient.age} yrs • Last Visit: {selectedPatient.last_visit}
          </div>
        </div>

        <div className="pipeline-grid">
          
          {/* STAGE 1 */}
          <div className="stage-card" style={{ opacity: selectedPatient.current_stage >= 1 ? 1 : 0.3 }}>
            <div className="stage-header">
              <div className="stage-title"><Activity size={18} style={{display:'inline', marginRight:'8px', verticalAlign:'text-bottom'}}/> Stage 1: Clinical</div>
              <div className="stage-status">{selectedPatient.data.stage_1 ? "Completed" : "Pending"}</div>
            </div>
            {selectedPatient.data.stage_1 && (
              <>
                <div className="data-grid">
                  <div className="data-item">
                    <span className="data-label">MMSE Score</span>
                    <span className="data-value">{selectedPatient.data.stage_1.mmse} / 30</span>
                  </div>
                  <div className="data-item">
                    <span className="data-label">Comorbidities</span>
                    <span className="data-value">{selectedPatient.data.stage_1.comorbidities}</span>
                  </div>
                </div>
                {renderRiskMeter(selectedPatient.predictions.stage_1_risk)}
              </>
            )}
          </div>

          {/* STAGE 2 */}
          <div className="stage-card" style={{ opacity: selectedPatient.current_stage >= 2 ? 1 : 0.3 }}>
            <div className="stage-header">
              <div className="stage-title"><Droplets size={18} style={{display:'inline', marginRight:'8px', verticalAlign:'text-bottom'}}/> Stage 2: Biofluids (Z-Scores)</div>
              <div className="stage-status">{selectedPatient.data.stage_2 ? "Completed" : "Pending"}</div>
            </div>
            {selectedPatient.data.stage_2 && (
              <>
                <div className="data-grid">
                  <div className="data-item">
                    <span className="data-label">p-Tau217 (Toxicity)</span>
                    <span className="data-value">{selectedPatient.data.stage_2.pTau217_Z}</span>
                  </div>
                  <div className="data-item">
                    <span className="data-label">NfL (Neuronal Death)</span>
                    <span className="data-value">{selectedPatient.data.stage_2.NfL_Z}</span>
                  </div>
                </div>
                {renderRiskMeter(selectedPatient.predictions.stage_2_risk)}
              </>
            )}
          </div>

          {/* STAGE 3 */}
          <div className="stage-card" style={{ opacity: selectedPatient.current_stage >= 3 ? 1 : 0.3 }}>
            <div className="stage-header">
              <div className="stage-title"><Brain size={18} style={{display:'inline', marginRight:'8px', verticalAlign:'text-bottom'}}/> Stage 3: Structural MRI</div>
              <div className="stage-status">{selectedPatient.data.stage_3 ? "Completed" : "Pending"}</div>
            </div>
            {selectedPatient.data.stage_3 && (
              <>
                <div className="data-grid">
                  <div className="data-item">
                    <span className="data-label">Hippocampus Vol</span>
                    <span className="data-value">{selectedPatient.data.stage_3.hippocampus_volume_cm3} cm³</span>
                  </div>
                  <div className="data-item">
                    <span className="data-label">WM Lesions</span>
                    <span className="data-value">{selectedPatient.data.stage_3.white_matter_lesions}</span>
                  </div>
                </div>
                {renderRiskMeter(selectedPatient.predictions.stage_3_risk)}
              </>
            )}
          </div>

        </div>

        {/* STAGE 4 SANDBOX VIEWER */}
        <div className="sandbox-viewer" style={{ opacity: selectedPatient.current_stage >= 4 ? 1 : 0.3 }}>
          <div className="sandbox-header">
            <div style={{fontWeight: 500}}><LayoutGrid size={18} style={{display:'inline', marginRight:'8px', verticalAlign:'text-bottom'}}/> Stage 4: PET Co-Registration Sandbox</div>
            {selectedPatient.data.stage_4 && (
              <div className="toggles">
                <button className={`toggle-btn ${activePetView === 'amyloid' ? 'active' : ''}`} onClick={() => setActivePetView('amyloid')}>Amyloid PET</button>
                <button className={`toggle-btn ${activePetView === 'tau' ? 'active' : ''}`} onClick={() => setActivePetView('tau')}>Tau PET</button>
                <button className={`toggle-btn ${activePetView === 'fdg' ? 'active' : ''}`} onClick={() => setActivePetView('fdg')}>FDG PET</button>
              </div>
            )}
          </div>
          
          <div className="sandbox-panes">
            {selectedPatient.data.stage_4 ? (
              <>
                <div className="pane">
                  <span className="pane-label">Axial</span>
                  <div className="pane-placeholder" style={{borderColor: activePetView === 'amyloid' ? '#ef4444' : (activePetView === 'tau' ? '#f59e0b' : '#3b82f6')}}>
                    Slice 48
                  </div>
                </div>
                <div className="pane">
                  <span className="pane-label">Sagittal</span>
                  <div className="pane-placeholder" style={{borderColor: activePetView === 'amyloid' ? '#ef4444' : (activePetView === 'tau' ? '#f59e0b' : '#3b82f6')}}>
                     Slice 65
                  </div>
                </div>
                <div className="pane">
                  <span className="pane-label">Coronal</span>
                  <div className="pane-placeholder" style={{borderColor: activePetView === 'amyloid' ? '#ef4444' : (activePetView === 'tau' ? '#f59e0b' : '#3b82f6')}}>
                     Slice 52
                  </div>
                </div>
              </>
            ) : (
              <div style={{margin: 'auto', color: 'var(--text-secondary)'}}>Patient has not reached Stage 4.</div>
            )}
          </div>
          
          {selectedPatient.data.stage_4 && (
            <div style={{padding: '16px 24px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem'}}>
              <span>Centiloids: <strong>{selectedPatient.data.stage_4.amyloid_centiloid}</strong></span>
              <span>Braak Stage: <strong>{selectedPatient.data.stage_4.braak_stage}</strong></span>
              <span>Final Risk: <strong>{(selectedPatient.predictions.stage_4_risk * 100).toFixed(0)}%</strong></span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
