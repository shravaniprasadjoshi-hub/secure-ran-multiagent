const API = 'http://localhost:8000';

// Agent data (synced from /state)
let agents = [
  {id:0,name:'Cell 0',trust:1.00,status:'healthy',load:72,ho:72},
  {id:1,name:'Cell 1',trust:0.95,status:'handover',load:45,ho:58},
  {id:2,name:'Cell 2',trust:0.88,status:'healthy',load:60,ho:61},
  {id:3,name:'Cell 3',trust:0.92,status:'healthy',load:55,ho:80},
  {id:4,name:'Cell 4',trust:0.85,status:'healthy',load:48,ho:66},
  {id:5,name:'Cell 5',trust:0.82,status:'healthy',load:58,ho:55},
  {id:6,name:'Cell 6',trust:0.90,status:'healthy',load:62,ho:70},
];

let agentMetrics = {};
let dataLoaded = false;
let detectedAgents = new Set(); // track which agents we've already shown detection banner for

// Tab switching
function showTab(name, btn) {
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'data' && !dataLoaded) loadDataExploration();
  if (name === 'agents' || name === 'trainval') loadAgentMetrics();
  if (name === 'marl') loadMarlTraining();
}

// Colours
function cssVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

function tColor(t) {
  if (t >= 0.7) return cssVar('--sage');
  if (t >= 0.4) return cssVar('--gold');
  return cssVar('--rust');
}

const fills = {
  healthy:'url(#gg)',
  handover:'url(#go)',
  degraded:'url(#gd)',
  byzantine:'url(#gr)'
};

const filters = {
  healthy:'url(#fg)',
  handover:'url(#fo)',
  degraded:'url(#fo)',
  byzantine:'url(#fr)'
};

const sLabels = {
  healthy:'Healthy',
  handover:'Handover active',
  degraded:'Degraded',
  byzantine:'Byzantine - quarantined'
};

function getStatusColor(status) {
  return {
    healthy: cssVar('--sage'),
    handover: cssVar('--gold'),
    degraded: cssVar('--orange'),
    byzantine: cssVar('--rust')
  }[status] || cssVar('--text-dim');
}

// Trust rendering
function renderTrust(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = agents.map(a => `
    <div class="agent-row">
      <span class="agent-id">${a.name}</span>
      <div class="tbar-bg"><div class="tbar-fill" style="width:${a.trust*100}%;background:${tColor(a.trust)}"></div></div>
      <span class="tval" style="color:${tColor(a.trust)}">${a.trust.toFixed(2)}</span>
      <span class="badge ${a.status==='byzantine'?'badge-danger':a.trust>=0.7?'badge-ok':'badge-warn'}">${a.status==='byzantine'?'Byzantine':a.trust>=0.7?'Trusted':'Degraded'}</span>
    </div>`).join('');
}

function renderAllTrust() {
  renderTrust('ov-trust');
  renderTrust('sec-trust');
}

// Hex twin
function applyHexStyle(id) {
  const a = agents[id];
  const poly = document.getElementById(`p${id}`);
  const ttext = document.getElementById(`t${id}`);
  const mainText = document.querySelector(`#h${id} text:first-of-type`);
  if (!poly) return;
  poly.setAttribute('fill', fills[a.status] || fills.healthy);
  poly.setAttribute('filter', filters[a.status] || filters.healthy);
  ttext.textContent = a.trust.toFixed(2);
  if (a.status === 'byzantine') {
    ttext.classList.add('pulse');
    if(mainText) mainText.classList.add('pulse');
  } else {
    ttext.classList.remove('pulse');
    if(mainText) mainText.classList.remove('pulse');
  }
}

function pickHex(g) {
  document.querySelectorAll('.hx polygon').forEach(p => p.setAttribute('stroke','rgba(255,255,255,0.3)'));
  g.querySelector('polygon').setAttribute('stroke','rgba(255,255,255,0.8)');
  const id = parseInt(g.dataset.cell);
  const a = agents[id];
  document.getElementById('cell-info').innerHTML = `
    <div class="cell-info-title">${g.dataset.name}</div>
    <div class="cell-info-row">
      <span>Status: <b style="color:${getStatusColor(a.status)}">${sLabels[a.status]}</b></span>
      <span>Trust: <b>${a.trust.toFixed(2)}</b></span>
      <span>Load: <b>${a.load}%</b></span>
      <span>HO rate: <b>${a.ho}%</b></span>
    </div>`;
}

// Fault injection
async function injectFault() {
  const atk = document.getElementById('atk-select').value;
  const cellSel = parseInt(document.getElementById('cell-select').value);
  const clean = agents.filter(a => a.status === 'healthy');
  if (!clean.length) { alert('No healthy agents left!'); return; }
  const target = cellSel >= 0 ? agents[cellSel] : clean[Math.floor(Math.random() * clean.length)];
  if (target.status !== 'healthy') { alert(`Cell ${target.id} is already compromised!`); return; }

  try {
    await fetch(`${API}/inject?agent_id=${target.id}&attack_type=${atk}`, {method:'POST'});
  } catch(e) {}

  // cell stays green - this is ground-truth injection only, real detection
  // happens during "Run live sim" via AnomalyDetector/PQC in sim_runner.py
  target.status = 'healthy';
  target.trust = 1.0;
  applyHexStyle(target.id);
  renderAllTrust();
  updateOverviewMetrics();

  addAlert('Byzantine', `Cell ${target.id} injected with ${atk} attack - monitoring...`);
  showBanner(`Cell ${target.id} injected with ${atk} attack - watching for suspicious behavior...`);

  document.getElementById('inject-status').innerHTML =
    `<span style="color:var(--gold-lt)">⏳ Cell ${target.id} injected with ${atk} attack - detection in progress, run simulation to observe...</span>`;
}

async function clearFaults() {
  try { await fetch(`${API}/clear`, {method:'POST'}); } catch(e) {}
  agents.forEach(a => { if (a.status !== 'handover') { a.status='healthy'; a.trust=1.0; } applyHexStyle(a.id); });
  detectedAgents.clear();
  renderAllTrust();
  updateOverviewMetrics();
  hideBanner();
  addAlert('Recovered','All faults cleared - agents restored');
  document.getElementById('inject-status').textContent = 'No faults injected - system clean';
  document.getElementById('cell-info').innerHTML =
    '<div class="cell-info-title">All faults cleared</div><div class="cell-info-row"><span style="color:var(--sage-lt)">All 7 agents restored ✓</span></div>';
}

function simHandover() {
  agents.forEach(a => { if (a.status === 'healthy') { a.status='handover'; applyHexStyle(a.id); } });
  addAlert('System','Handover event - UEs switching between cells');
  setTimeout(() => {
    agents.forEach(a => { if (a.status === 'handover') { a.status='healthy'; applyHexStyle(a.id); } });
    addAlert('System','Handover complete');
  }, 2200);
}

async function startSim() {
  try {
    const r = await fetch(`${API}/start-sim`, {method:'POST'});
    const d = await r.json();
    document.getElementById('sim-status-pill').textContent = d.ok ? 'Simulation running...' : d.msg;
    document.getElementById('sim-status-pill').style.color = d.ok ? 'var(--sage-lt)' : 'var(--rust-lt)';
  } catch(e) {
    document.getElementById('sim-status-pill').textContent = 'API not reachable';
  }
}

// Overview metrics
function updateOverviewMetrics() {
  const byz = agents.filter(a => a.status === 'byzantine').length;
  const rate = Math.max(30, 72 - byz * 8);
  document.getElementById('ov-consensus').textContent = rate + '%';
  document.getElementById('ov-consensus-sub').textContent = byz > 0
    ? `${byz} agent${byz>1?'s':''} quarantined`
    : '↑ from 6.6% in the shared baseline';
}

// Alerts
function addAlert(type, msg) {
  const cls = {Byzantine:'badge-danger',Policy:'badge-warn',Consensus:'badge-sand',System:'badge-ok',Recovered:'badge-ok'}[type]||'badge-sand';
  const el = document.getElementById('ov-alerts');
  el.insertAdjacentHTML('afterbegin', `
    <div class="alert-item">
      <span class="badge ${cls}">${type}</span>
      <div><div class="alert-text">${msg}</div><div class="alert-time">Just now</div></div>
    </div>`);
  while (el.children.length > 5) el.removeChild(el.lastChild);
}

function showBanner(msg) {
  document.getElementById('alert-text').textContent = msg;
  document.getElementById('alert-banner').classList.add('visible');
}
function hideBanner() { document.getElementById('alert-banner').classList.remove('visible'); }

// Consensus log
function renderConsensusLog(log) {
  const el = document.getElementById('sec-consensus-log');
  if (!log || !log.length) return;
  el.innerHTML = log.map(e => `
    <div class="cons-row">
      <span class="cons-step">Step ${e.step}</span>
      <span class="badge ${e.ok?'badge-ok':'badge-warn'}">${e.ok?'✓':'✗'} ${e.agreement}%</span>
      <span class="cons-detail">action=${e.final_action ?? '-'} · excluded: ${e.excluded?.length ? e.excluded.join(',') : 'none'}${e.pqc_excluded?.length ? ' · PQC-rejected: ' + e.pqc_excluded.join(',') : ''}</span>
    </div>`).join('');
}

// PQC (Exp 4) panel — pqc block already lives in sim_state, no new endpoint needed
function renderPqc(pqc) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('pqc-kem-alg', pqc.kem_alg || '—');
  set('pqc-sig-alg', pqc.sig_alg || '—');
  set('pqc-kem-handshakes', pqc.kem_handshakes ?? 0);
  set('pqc-rejections', pqc.rejections ?? 0);
  set('pqc-sig-issued', pqc.signatures_issued ?? 0);
  set('pqc-verifications', pqc.verifications ?? 0);
  set('pqc-avg-sign', `${(pqc.avg_sign_ms ?? 0).toFixed(2)} ms`);
  set('pqc-avg-verify', `${(pqc.avg_verify_ms ?? 0).toFixed(2)} ms`);
}

// Poll /state every 2s
async function pollState() {
  try {
    const r = await fetch(`${API}/state`);
    const data = await r.json();
    document.getElementById('sys-status').textContent = 'System online';

    if (data.agents) {
      data.agents.forEach((ag, i) => {
        agents[i].trust = ag.trust;
        agents[i].status = ag.status;
        applyHexStyle(i);

        // show detection banner only once real detection has dropped trust meaningfully
        if (ag.trust < 0.5 && ag.status === 'byzantine' && !detectedAgents.has(i)) {
          detectedAgents.add(i);
          showBanner(`⚠ Cell ${i} detected as Byzantine - trust dropped to ${ag.trust.toFixed(2)} - quarantined from consensus`);
          addAlert('Byzantine', `Cell ${i} caught - trust ${ag.trust.toFixed(2)}, excluded from voting`);
          document.getElementById('inject-status').innerHTML =
            `<span style="color:var(--rust-lt)">⚠ Cell ${i} detected · trust → ${ag.trust.toFixed(2)} · excluded from consensus · network stable</span>`;
        }
      });
      renderAllTrust();
      updateOverviewMetrics();
    }

    if (data.consensus_rate !== undefined) {
      document.getElementById('ov-consensus').textContent = Math.round(data.consensus_rate * 100) + '%';
    }

    if (data.alerts && data.alerts.length) {
      const el = document.getElementById('ov-alerts');
      el.innerHTML = data.alerts.slice(0,5).map(a => `
        <div class="alert-item">
          <span class="badge ${a.type==='Byzantine'?'badge-danger':a.type==='Recovered'?'badge-ok':'badge-sand'}">${a.type}</span>
          <div><div class="alert-text">${a.msg}</div><div class="alert-time">${a.time}</div></div>
        </div>`).join('');
    }

    if (data.consensus_log) renderConsensusLog(data.consensus_log);

    if (data.pqc) renderPqc(data.pqc);

    if (data.running) {
      document.getElementById('sim-status-pill').textContent = `Running - step ${data.step}`;
    } else if (data.step > 0) {
      document.getElementById('sim-status-pill').textContent = `Complete - ${data.step} steps`;
    }

  } catch(e) {
    document.getElementById('sys-status').textContent = 'API offline';
  }
}

// Data Exploration
async function loadDataExploration() {
  try {
    const r = await fetch(`${API}/telemetry-stats`);
    const d = await r.json();
    dataLoaded = true;

    const pt = getPlotlyTheme();

    const darkLayout = {
      paper_bgcolor:'rgba(0,0,0,0)',
      plot_bgcolor:'rgba(0,0,0,0)',
      font:{color:pt.text},
      margin:{t:40,b:40,l:50,r:20},
      xaxis:{color:pt.text, gridcolor:pt.grid},
      yaxis:{color:pt.text, gridcolor:pt.grid}
    };

    if (d.rsrp) {
      const edges = d.rsrp.edges;
      const x = edges.slice(0,-1).map((v,i)=>((v+edges[i+1])/2).toFixed(1));
      Plotly.newPlot('rsrp-chart', [{type:'bar',x,y:d.rsrp.counts,marker:{color:pt.teal},name:'RSRP'}],
        {...darkLayout, title:{text:'RSRP Distribution (dBm)',font:{color:pt.title}}, xaxis:{title:'dBm',color:pt.text}, yaxis:{color:pt.text}});
    }

    if (d.sinr) {
      const edges = d.sinr.edges;
      const x = edges.slice(0,-1).map((v,i)=>((v+edges[i+1])/2).toFixed(1));
      Plotly.newPlot('sinr-chart', [{type:'bar',x,y:d.sinr.counts,marker:{color:pt.blue},name:'SINR'}],
        {...darkLayout, title:{text:'SINR Distribution (dB)',font:{color:pt.title}}, xaxis:{title:'dB',color:pt.text}, yaxis:{color:pt.text}});
    }

    if (d.correlation) {
      const cols = Object.keys(d.correlation);
      const z = cols.map(r => cols.map(c => d.correlation[r][c] ?? 0));
      Plotly.newPlot('corr-chart',
        [{type:'heatmap',z,x:cols,y:cols,colorscale:'RdBu',zmid:0,text:z.map(r=>r.map(v=>v.toFixed(2))),texttemplate:'%{text}'}],
        {...darkLayout, title:{text:'Feature Correlation Heatmap',font:{color:pt.title}}});
    }

    if (d.cdf) {
      Plotly.newPlot('cdf-chart',
        [{x:d.cdf.x, y:d.cdf.y, mode:'lines', line:{color:pt.teal,width:2}, name:'latency_ms'}],
        {...darkLayout, title:{text:'CDF - Latency (ms)',font:{color:pt.title}}, xaxis:{title:'ms',color:pt.text}, yaxis:{title:'CDF',color:pt.text}});
    }

    if (d.sinr_by_scenario) {
      const scenarios = Object.keys(d.sinr_by_scenario);
      const values = scenarios.map(s => d.sinr_by_scenario[s]);
      Plotly.newPlot('sinr-scenario-chart',
        [{type:'bar', x:scenarios, y:values, marker:{color:pt.sage}, name:'Median SINR'}],
        {...darkLayout, title:{text:'SINR by Scenario (median)',font:{color:pt.title}}, xaxis:{color:pt.text}, yaxis:{title:'dB',color:pt.text}});
    }

    if (d.scenario_distribution) {
      const labels = Object.keys(d.scenario_distribution);
      const values = labels.map(l => d.scenario_distribution[l]);
      Plotly.newPlot('ov-scenario-chart',
        [{type:'pie', labels, values, hole:0.4, marker:{colors:[pt.teal,pt.blue,pt.rust,pt.orange,pt.gold,pt.text]}}],
        {...darkLayout, margin:{t:10,b:10,l:10,r:10}, showlegend:true, legend:{font:{color:pt.text},orientation:'v'}});
    }

  } catch(e) {
    console.error('Data exploration load error:', e);
  }
}

// Agent Metrics
async function loadAgentMetrics() {
  if (Object.keys(agentMetrics).length) { renderAgentSummary(); return; }
  try {
    const r = await fetch(`${API}/agent-metrics`);
    const d = await r.json();
    agentMetrics = d.agents || {};

    const sel = document.getElementById('agent-select');
    Object.keys(agentMetrics).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name.replace('_agent','').replace('_',' ');
      sel.appendChild(opt);
    });
    sel.value = Object.keys(agentMetrics)[0];
    renderAgentDetail();
    renderAgentSummary();
    renderTrainVal();
  } catch(e) { console.error(e); }
}

function getKey(ag) { return ag.task === 'classification' ? 'f1' : 'r2'; }
function scoreColor(v) { return v >= 0.85 ? 'score-good' : v >= 0.5 ? 'score-mid' : 'score-bad'; }

function renderAgentDetail() {
  const name = document.getElementById('agent-select').value;
  if (!name || !agentMetrics[name]) return;
  const ag = agentMetrics[name];
  const key = getKey(ag);
  document.getElementById('ag-train').textContent = (ag.train[key]||0).toFixed(3);
  document.getElementById('ag-val').textContent   = (ag.val[key]||0).toFixed(3);
  document.getElementById('ag-test').textContent  = (ag.test[key]||0).toFixed(3);

  const pt = getPlotlyTheme();
  const darkLayout = { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:pt.text}, margin:{t:40,b:40,l:50,r:20}, xaxis:{color:pt.text,gridcolor:pt.grid}, yaxis:{color:pt.text,gridcolor:pt.grid} };
  Plotly.newPlot('agent-bar-chart',
    [{type:'bar', x:['Train','Validation','Test'], y:[ag.train[key]||0, ag.val[key]||0, ag.test[key]||0],
      marker:{color:[pt.teal,pt.sage,pt.rust]}, text:[(ag.train[key]||0).toFixed(3),(ag.val[key]||0).toFixed(3),(ag.test[key]||0).toFixed(3)], textposition:'outside'}],
    {...darkLayout, title:{text:`${name} - ${key.toUpperCase()} across splits`,font:{color:pt.title}},
      yaxis:{range:[0,1.1],color:pt.text}, xaxis:{color:pt.text}});
}

function renderAgentSummary() {
  const table = document.getElementById('agent-summary-table');
  table.innerHTML = `<tr><th>Agent</th><th>Model</th><th>Task</th><th>Train</th><th>Val</th><th>Test</th></tr>` +
    Object.entries(agentMetrics).map(([name, ag]) => {
      const key = getKey(ag);
      const tr = ag.train[key]||0, vl = ag.val[key]||0, te = ag.test[key]||0;
      return `<tr>
        <td style="color:var(--cream);font-weight:600">${name.replace('_agent','')}</td>
        <td style="color:var(--text-dim)">${ag.model_type}</td>
        <td style="color:var(--text-dim)">${ag.task}</td>
        <td class="${scoreColor(tr)}">${tr.toFixed(3)}</td>
        <td class="${scoreColor(vl)}">${vl.toFixed(3)}</td>
        <td class="${scoreColor(te)}">${te.toFixed(3)} ${key.toUpperCase()}</td>
      </tr>`;
    }).join('');
}

function renderTrainVal() {
  const names = Object.keys(agentMetrics).map(n=>n.replace('_agent',''));
  const pt = getPlotlyTheme();
  const darkLayout = { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:pt.title}, margin:{t:50,b:80,l:50,r:20}, xaxis:{color:pt.text,gridcolor:pt.grid}, yaxis:{color:pt.text,gridcolor:pt.grid} };
  const traces = ['train','val','test'].map((split,i) => ({
    type:'bar', name:split.charAt(0).toUpperCase()+split.slice(1),
    x:names,
    y:Object.values(agentMetrics).map(ag=>ag[split][getKey(ag)]||0),
    marker:{color:[pt.teal,pt.sage,pt.rust][i]},
  }));
  Plotly.newPlot('trainval-chart', traces,
    {...darkLayout, barmode:'group', title:{text:'All Agents - Train / Val / Test',font:{color:pt.title}},
      yaxis:{range:[-0.1,1.1],color:pt.text}, xaxis:{color:pt.text}});

  const table = document.getElementById('model-details-table');
  table.innerHTML = `<tr><th>Agent</th><th>Model</th><th>Task</th><th>Test score</th></tr>` +
    Object.entries(agentMetrics).map(([name,ag]) => {
      const key = getKey(ag);
      const te = ag.test[key]||0;
      return `<tr>
        <td style="color:var(--cream);font-weight:600">${name.replace('_agent','')}</td>
        <td style="color:var(--text-dim)">${ag.model_type}</td>
        <td style="color:var(--text-dim)">${ag.task}</td>
        <td class="${scoreColor(te)}">${te.toFixed(3)} ${key.toUpperCase()}</td>
      </tr>`;
    }).join('');
}

// MARL Training
async function loadMarlTraining() {
  try {
    const r = await fetch(`${API}/training-log`);
    const d = await r.json();
    if (!d.episodes || !d.episodes.length) return;

    const pt = getPlotlyTheme();
    const darkLayout = { paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:pt.text}, margin:{t:50,b:50,l:60,r:20}, xaxis:{color:pt.text,gridcolor:pt.grid}, yaxis:{color:pt.text,gridcolor:pt.grid} };

    const rewards = d.rewards;
    const window = 50;
    const rollingMean = rewards.map((_,i) => {
      const start = Math.max(0,i-window+1);
      const slice = rewards.slice(start,i+1);
      return slice.reduce((a,b)=>a+b,0)/slice.length;
    });
    const ema = [];
    rewards.forEach((v,i) => ema.push(i === 0 ? v : 0.05*v + 0.95*ema[i-1]));

    Plotly.newPlot('reward-chart',
      [{x:d.episodes,y:rewards,mode:'lines',line:{color:pt.teal,width:1},opacity:0.12,name:'Raw reward'},
       {x:d.episodes,y:rollingMean,mode:'lines',line:{color:pt.teal,width:2},name:'Rolling mean (w=50)'},
       {x:d.episodes,y:ema,mode:'lines',line:{color:pt.tealLight,width:1.5,dash:'dash'},name:'EMA'}],
      {...darkLayout,title:{text:'Training Reward — Raw / Rolling Mean / EMA',font:{color:pt.title}},
        xaxis:{title:'Episode',color:pt.text,gridcolor:pt.grid},yaxis:{title:'Total Reward',color:pt.text,gridcolor:pt.grid},
        legend:{font:{color:pt.text}}});

    const lastRewards = rewards.slice(-100);

    Plotly.newPlot('reward-hist-chart',
      [{x:lastRewards,type:'histogram',nbinsx:30,opacity:0.65,marker:{color:pt.teal},name:'Reward'}],
      {...darkLayout,title:{text:'Reward Distribution (last 100 episodes)',font:{color:pt.title}},
        xaxis:{title:'Total Reward',color:pt.text,gridcolor:pt.grid},yaxis:{title:'Count',color:pt.text,gridcolor:pt.grid},
        showlegend:false});

    Plotly.newPlot('reward-box-chart',
      [{y:lastRewards,type:'box',name:'Training reward',marker:{color:pt.teal},line:{color:pt.teal},boxmean:true,boxpoints:false}],
      {...darkLayout,title:{text:'Reward Spread (last 100 episodes)',font:{color:pt.title}},
        xaxis:{color:pt.text,gridcolor:pt.grid},yaxis:{title:'Total Reward',color:pt.text,gridcolor:pt.grid},
        showlegend:false});

  } catch(e) { console.error('MARL training load error:',e); }
}

// Chatbot
const chatResponses = {
  'byzantine': 'A Byzantine agent is a compromised node sending malicious actions to manipulate consensus. Our system detects it using three methods: statistical outlier detection, voting disagreement tracking, and behavioral drift analysis across a sliding window.',
  'consensus': 'Our consensus engine uses Byzantine-robust voting - it excludes flagged agents before tallying votes. Minimum agreement threshold is 60%. In the shared baseline, consensus accept rate is only 6.6%. Our MARL system reaches ~72%.',
  'trust': 'Trust scores range from 0 to 1. They drop when an agent is flagged by the anomaly detector or violates policy checks, and slowly recover when the agent behaves consistently with the group.',
  'handover': 'A handover is when a UE switches from one cell to another. Our MARL agents learn when to trigger handovers based on RSRP and SINR. The consensus engine ensures no single compromised agent can force a bad handover decision.',
  'oran': 'O-RAN is the open architecture this system runs in. Our agents live at the Near-RT RIC layer - making millisecond-to-second decisions pushed down to the O-DU and O-CU.',
  'exp 3': 'Experiment 3 added Byzantine injection and security modules to MARL training. The security layer correctly slows convergence by throttling learning signal during quarantine. Reward went from -2,520 to -259 over 1000 episodes. Given 2000-3000 episodes it will likely close the gap.',
  'rsrp': 'RSRP (Reference Signal Received Power) measures the signal strength from the serving cell. Values range from around -125 to -60 dBm, with the distribution centered around -90 dBm.',
  'sinr': 'SINR (Signal to Interference and Noise Ratio) measures signal quality. Values above 10 dB are generally good. SINR drops sharply during jamming attacks - the clearest attack signature in the data.',
  'mappo': 'MAPPO - Multi-Agent Proximal Policy Optimization - uses centralized training with decentralized execution. During training, a shared critic sees all 7 agents observations globally. During execution, each agent acts independently using only its local observations.',
  'rrc': 'RRC - Radio Resource Control - manages the connection between UEs and the RAN. It controls handover decisions, access control, resource allocation, and mobility management. In 6G, our MAPPO agents replace traditional rule-based RRC controllers.',
  'reward': 'Our reward function: successful handover = +10, healthy defer = +1, radio link failure = -10, ping-pong = -5. These incentives teach agents to trigger handovers when genuinely needed and avoid unnecessary switching.',
  'pqc': 'PQC - Post-Quantum Cryptography - secures inter-agent communication with Kyber768 key exchange and Dilithium3 signatures via liboqs, so a compromised agent can be caught spoofing a message even if its behavior alone looks fine. This is Experiment 4.',
  'default': 'Based on the project knowledge base - this relates to our secure multi-agent AI framework for 6G RAN control. Try asking about Byzantine attacks, consensus mechanisms, trust scoring, MAPPO, RSRP, SINR, PQC, or O-RAN architecture.',
};

async function sendChat() {
  const inp = document.getElementById('chat-input');
  const msg = inp.value.trim();
  if (!msg) return;

  const msgs = document.getElementById('chat-msgs');
  msgs.insertAdjacentHTML('beforeend',
    `<div class="chat-msg chat-user">${msg}</div>`);
  inp.value = '';

  const typingId = 'typing-' + Date.now();
  msgs.insertAdjacentHTML('beforeend',
    `<div class="chat-msg chat-bot" id="${typingId}" style="color:var(--text-dim);font-style:italic">Searching knowledge base...</div>`);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const r = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: msg})
    });
    const data = await r.json();
    document.getElementById(typingId)?.remove();

    let sourceNote = '';
    if (data.rag_available && data.sources && data.sources.length) {
      const src = data.sources[0];
      sourceNote = `<div style="font-size:11px;color:var(--text-dim);margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06)">
        📄 ${src.title} · <span style="color:var(--teal)">${src.source}</span>
      </div>`;
    }

    msgs.insertAdjacentHTML('beforeend',
      `<div class="chat-msg chat-bot">${data.answer}${sourceNote}</div>`);

  } catch(e) {
    document.getElementById(typingId)?.remove();
    const lower = msg.toLowerCase();
    let reply = chatResponses['default'];
    for (const [k, v] of Object.entries(chatResponses)) {
      if (lower.includes(k)) { reply = v; break; }
    }
    msgs.insertAdjacentHTML('beforeend',
      `<div class="chat-msg chat-bot">${reply}<div style="font-size:11px;color:var(--text-dim);margin-top:6px">⚠ API offline - using cached responses</div></div>`);
  }

  msgs.scrollTop = msgs.scrollHeight;
}

function askSuggested(btn) {
  document.getElementById('chat-input').value = btn.textContent;
  sendChat();
}

// Theme
function updateThemeIcon() {
  const isLight = document.documentElement.classList.contains('light');
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (icon) icon.textContent = isLight ? '☀' : '☾';
  if (label) label.textContent = isLight ? 'Light' : 'Dark';
}

function toggleTheme() {
  document.documentElement.classList.toggle('light');
  const theme = document.documentElement.classList.contains('light') ? 'light' : 'dark';
  localStorage.setItem('ran-theme', theme);
  updateThemeIcon();
  updatePlotlyTheme();
}

function initTheme() {
  const saved = localStorage.getItem('ran-theme');
  if (saved === 'light') document.documentElement.classList.add('light');
  else document.documentElement.classList.remove('light');
  updateThemeIcon();
}

function getPlotlyTheme() {
  const light = document.documentElement.classList.contains('light');
  return {
    text:      light ? '#647777' : '#8FA5A8',
    title:     light ? '#172525' : '#E6F1F2',
    grid:      light ? 'rgba(100,119,119,0.15)' : 'rgba(143,165,168,0.12)',
    teal:      light ? '#159B8C' : '#35C6B0',
    tealLight: light ? '#08796D' : '#75DDD0',
    blue:      light ? '#287BA8' : '#55A7D9',
    gold:      light ? '#C98B16' : '#E5B85C',
    sage:      light ? '#3C9A58' : '#63C174',
    rust:      light ? '#D05252' : '#E36A6A',
    orange:    light ? '#C96835' : '#E58A5C'
  };
}

function updatePlotlyTheme() {
  const ids = ['rsrp-chart','sinr-chart','corr-chart','cdf-chart',
               'sinr-scenario-chart','ov-scenario-chart','agent-bar-chart',
               'trainval-chart','reward-chart','reward-hist-chart','reward-box-chart'];
  const theme = getPlotlyTheme();
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el || !el.data) return;
    Plotly.relayout(id, {
      'font.color': theme.text, 'title.font.color': theme.title,
      'xaxis.color': theme.text, 'yaxis.color': theme.text,
      'xaxis.gridcolor': theme.grid, 'yaxis.gridcolor': theme.grid,
      'legend.font.color': theme.text,
      'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'
    });
  });
}

// Init
initTheme();
agents.forEach((_,i) => applyHexStyle(i));
renderAllTrust();
updateOverviewMetrics();
loadDataExploration();
pollState();
setInterval(pollState, 2000);