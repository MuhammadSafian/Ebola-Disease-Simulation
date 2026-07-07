/**
 * ui.js — Main UI Controller
 * Handles interactions, API communication, and coordinates Sim/Charts.
 */

const App = (() => {
    let isRunning = false;
    let simInterval = null;
    let currentCountry = 'Guinea';
    
    // DOM Elements
    const els = {};

    function init() {
        bindElements();
        bindEvents();
        
        SimRenderer.init();
        ChartManager.init();
        
        fetchEbolaData(currentCountry);
        resetSimulation();
    }

    function bindElements() {
        ['kpiHealthy', 'kpiExposed', 'kpiInfected', 'kpiRecovered', 'kpiDead', 'kpiDay',
         'sliderPop', 'popVal', 'sliderInf', 'infVal', 'sliderInitInf', 'initInfVal', 'sliderSpeed', 'speedVal',
         'btnStart', 'btnPause', 'btnReset', 'riskGrid', 'statusDot', 'statusText',
         'badgeABM', 'badgeANN', 'slpStatus', 'slpAccuracy', 'slpLoss', 'slpEpochs', 
         'slpWeights', 'ebolaCountryLabel'].forEach(id => {
            els[id] = document.getElementById(id);
        });
        
        els.countryBtns = document.querySelectorAll('.country-btn');
    }

    function bindEvents() {
        els.btnStart.addEventListener('click', startSimulation);
        els.btnPause.addEventListener('click', pauseSimulation);
        els.btnReset.addEventListener('click', resetSimulation);
        
        els.sliderPop.addEventListener('input', e => els.popVal.textContent = e.target.value);
        els.sliderInf.addEventListener('input', e => els.infVal.textContent = e.target.value + '%');
        els.sliderInitInf.addEventListener('input', e => els.initInfVal.textContent = e.target.value);
        els.sliderSpeed.addEventListener('input', e => {
            const val = parseInt(e.target.value);
            els.speedVal.textContent = val === 1 ? 'Slow' : val === 2 ? 'Normal' : 'Fast';
            if (isRunning) { pauseSimulation(); startSimulation(); }
        });

        els.countryBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                els.countryBtns.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                currentCountry = e.target.dataset.country;
                els.ebolaCountryLabel.textContent = currentCountry;
                fetchEbolaData(currentCountry);
                if (!isRunning) resetSimulation();
            });
        });
    }

    // --- API Calls ---

    async function fetchEbolaData(country) {
        try {
            const res = await fetch(`/api/ebola-data?country=${encodeURIComponent(country)}`);
            const data = await res.json();
            ChartManager.updateEbolaChart(data);
        } catch (e) { console.error("Error fetching Ebola data:", e); }
    }

    async function initModel() {
        const payload = {
            population: parseInt(els.sliderPop.value),
            country: currentCountry,
            infection_rate: parseInt(els.sliderInf.value) / 100.0,
            initial_infected: parseInt(els.sliderInitInf.value)
        };

        try {
            const res = await fetch('/api/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (data.status === 'ok') {
                updateUI(data.state);
                ChartManager.resetSpreadChart();
                ChartManager.updateSpreadChart(data.state.history);
                
                if (data.slp_info) {
                    updateSlpInfo(data.slp_info);
                    ChartManager.updateSlpChart(data.slp_info);
                }
            }
        } catch (e) { console.error("Init Error:", e); }
    }

    let simTimeout = null;

    async function stepLoop() {
        if (!isRunning) return;

        try {
            // Always request 1 step for smooth continuous animation
            const res = await fetch('/api/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 1 })
            });
            const data = await res.json();
            
            if (data.status === 'ok') {
                updateUI(data.state);
                ChartManager.updateSpreadChart(data.state.history);
            } else {
                console.error("Step Error:", data.error || "Unknown error");
                pauseSimulation();
                return;
            }
        } catch (e) { 
            console.error("Step Error:", e);
            pauseSimulation();
            return;
        }

        if (isRunning) {
            // Adjust delay based on speed slider
            const speed = parseInt(els.sliderSpeed.value);
            const intervalMs = speed === 3 ? 10 : (speed === 2 ? 50 : 200);
            simTimeout = setTimeout(stepLoop, intervalMs);
        }
    }

    // --- UI Updates ---

    function updateUI(state) {
        if (!state) return;
        
        // Update KPIs
        els.kpiHealthy.textContent = state.counts.S;
        els.kpiExposed.textContent = state.counts.E;
        els.kpiInfected.textContent = state.counts.I;
        els.kpiRecovered.textContent = state.counts.R;
        els.kpiDead.textContent = state.counts.D;
        els.kpiDay.textContent = state.step;

        // Render Canvas
        SimRenderer.setAgents(state.agents);

        // Render Risk Cards
        renderRiskCards(state.top_agents);
    }

    function renderRiskCards(agents) {
        if (!agents || !agents.length) return;
        
        let html = '';
        agents.forEach(a => {
            // Lower threshold so risk > 50% shows as red (risk-high)
            const riskLevel = a.risk > 50 ? 'risk-high' : a.risk > 20 ? 'risk-med' : 'risk-low';
            html += `
                <div class="risk-card ${riskLevel}">
                    <div class="risk-dot"></div>
                    <div class="risk-agent-name">Agent ${a.id}</div>
                    <div class="risk-pct">${a.risk.toFixed(1)}%</div>
                    <div class="risk-bar">
                        <div class="risk-bar-fill" style="width: ${a.risk}%"></div>
                    </div>
                </div>
            `;
        });
        els.riskGrid.innerHTML = html;
    }

    function updateSlpInfo(info) {
        if (!info) return;
        els.slpStatus.textContent = info.is_trained ? 'Trained on WHO Data' : 'Error';
        els.slpStatus.style.color = info.is_trained ? '#43A047' : '#E53935';
        els.slpAccuracy.textContent = (info.accuracy * 100).toFixed(1) + '%';
        els.slpLoss.textContent = info.final_loss.toFixed(4);
        els.slpEpochs.textContent = info.epochs_trained;
        
        const w = info.weights;
        const fn = info.feature_names;
        els.slpWeights.innerHTML = `${fn[0]}: ${w[0].toFixed(2)} | ${fn[1]}: ${w[1].toFixed(2)}<br>${fn[2]}: ${w[2].toFixed(2)} | ${fn[3]}: ${w[3].toFixed(2)}<br>Bias: ${info.bias.toFixed(2)}`;
    }

    // --- Simulation Controls ---

    function startSimulation() {
        if (isRunning) return;
        isRunning = true;
        
        els.btnStart.disabled = true;
        els.btnStart.textContent = '▶ Running...';
        els.btnPause.disabled = false;
        els.btnPause.textContent = '⏸ Pause';
        
        els.sliderPop.disabled = true; // Lock population while running
        els.sliderInitInf.disabled = true;
        
        els.statusDot.className = 'status-dot running';
        els.statusText.textContent = 'Simulation running...';

        stepLoop();
    }

    function pauseSimulation() {
        if (!isRunning) return;
        isRunning = false;
        
        els.btnStart.disabled = false;
        els.btnStart.textContent = '▶ Resume';
        els.btnPause.disabled = true;
        els.btnPause.textContent = '⏸ Paused';
        
        els.statusDot.className = 'status-dot paused';
        els.statusText.textContent = 'Simulation paused';
        
        if (simTimeout) clearTimeout(simTimeout);
    }

    async function resetSimulation() {
        pauseSimulation();
        
        // Show resetting feedback
        const oldResetText = els.btnReset.textContent;
        els.btnReset.textContent = '↻ Resetting...';
        els.btnStart.textContent = '▶ Start';
        els.btnPause.textContent = '⏸ Pause';
        
        els.sliderPop.disabled = false;
        els.sliderInitInf.disabled = false;
        els.statusDot.className = 'status-dot';
        els.statusText.textContent = 'Ready — click Start to begin';
        
        ChartManager.resetSpreadChart();
        await initModel();
        
        // Restore reset button
        els.btnReset.textContent = '↻ Reset';
    }

    return { init };
})();

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', App.init);
