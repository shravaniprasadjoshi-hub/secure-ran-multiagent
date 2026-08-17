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
function tColor(t) {
  if (t >= 0.7) return '#7A9E7E';
  if (t >= 0.4) return '#C9933A';
  return '#B85C38';
}
const fills   = {healthy:'url(#gg)',handover:'url(#go)',degraded:'url(#gd)',byzantine:'url(#gr)'};
const filters = {healthy:'url(#fg)',handover:'url(#fo)',degraded:'url(#fo)',byzantine:'url(#fr)'};
const sLabels = {healthy:'Healthy',handover:'Handover active',degraded:'Degraded',byzantine:'Byzantine — quarantined'};
const sColors = {healthy:'#7A9E7E',handover:'#C9933A',degraded:'#8B6914',byzantine:'#B85C38'};

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
      <span>Status: <b style="color:${sColors[a.status]}">${sLabels[a.status]}</b></span>
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

  target.status = 'byzantine';
  target.trust = 0.41;
  applyHexStyle(target.id);
  renderAllTrust();
  updateOverviewMetrics();
  addAlert('Byzantine', `Cell ${target.id} compromised — ${atk} attack`);
  addAlert('Consensus', `Cell ${target.id} excluded from voting`);
  showBanner(`⚠ Byzantine agent — Cell ${target.id} quarantined (${atk} attack)`);
  document.getElementById('inject-status').innerHTML =
    `<span style="color:var(--rust-lt)">⚠ Cell ${target.id} compromised · ${atk} attack · trust → 0.41 · excluded from consensus</span>`;
}

async function clearFaults() {
  try { await fetch(`${API}/clear`, {method:'POST'}); } catch(e) {}
  agents.forEach(a => { if (a.status !== 'handover') { a.status='healthy'; a.trust=1.0; } applyHexStyle(a.id); });
  renderAllTrust();
  updateOverviewMetrics();
  hideBanner();
  addAlert('Recovered','All faults cleared — agents restored');
  document.getElementById('inject-status').textContent = 'No faults injected — system clean';
  document.getElementById('cell-info').innerHTML =
    '<div class="cell-info-title">All faults cleared</div><div class="cell-info-row"><span style="color:var(--sage-lt)">All 7 agents restored ✓</span></div>';
}

function simHandover() {
  agents.forEach(a => { if (a.status === 'healthy') { a.status='handover'; applyHexStyle(a.id); } });
  addAlert('System','Handover event — UEs switching between cells');
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
      <span class="cons-detail">action=${e.final_action ?? '—'} · excluded: ${e.excluded?.length ? e.excluded.join(',') : 'none'}</span>
    </div>`).join('');
}

// Poll /state every 2s
async function pollState() {
  try {
    const r = await fetch(`${API}/state`);
    const data = await r.json();
    document.getElementById('sys-status').textContent = 'System online';

    // update agents from backend
    if (data.agents) {
      data.agents.forEach((ag, i) => {
        agents[i].trust = ag.trust;
        agents[i].status = ag.status;
        applyHexStyle(i);
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

    if (data.running) {
      document.getElementById('sim-status-pill').textContent = `Running — step ${data.step}`;
    } else if (data.step > 0) {
      document.getElementById('sim-status-pill').textContent = `Complete — ${data.step} steps`;
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

    const darkLayout = {
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      font:{color:'#A89880'}, margin:{t:40,b:40,l:50,r:20},
    };

    // RSRP histogram
    if (d.rsrp) {
      const edges = d.rsrp.edges;
      const x = edges.slice(0,-1).map((v,i)=>((v+edges[i+1])/2).toFixed(1));
      Plotly.newPlot('rsrp-chart', [{type:'bar',x,y:d.rsrp.counts,marker:{color:'#C9933A'},name:'RSRP'}],
        {...darkLayout, title:{text:'RSRP Distribution (dBm)',font:{color:'#EDE0C8'}}, xaxis:{title:'dBm',color:'#A89880'}, yaxis:{color:'#A89880'}});
    }

    // SINR histogram
    if (d.sinr) {
      const edges = d.sinr.edges;
      const x = edges.slice(0,-1).map((v,i)=>((v+edges[i+1])/2).toFixed(1));
      Plotly.newPlot('sinr-chart', [{type:'bar',x,y:d.sinr.counts,marker:{color:'#7A9E7E'},name:'SINR'}],
        {...darkLayout, title:{text:'SINR Distribution (dB)',font:{color:'#EDE0C8'}}, xaxis:{title:'dB',color:'#A89880'}, yaxis:{color:'#A89880'}});
    }

    // Correlation heatmap
    if (d.correlation) {
      const cols = Object.keys(d.correlation);
      const z = cols.map(r => cols.map(c => d.correlation[r][c] ?? 0));
      Plotly.newPlot('corr-chart',
        [{type:'heatmap',z,x:cols,y:cols,colorscale:'RdBu',zmid:0,text:z.map(r=>r.map(v=>v.toFixed(2))),texttemplate:'%{text}'}],
        {...darkLayout, title:{text:'Feature Correlation Heatmap',font:{color:'#EDE0C8'}}});
    }

    // CDF
    if (d.cdf) {
      Plotly.newPlot('cdf-chart',
        [{x:d.cdf.x, y:d.cdf.y, mode:'lines', line:{color:'#C9933A',width:2}, name:'latency_ms'}],
        {...darkLayout, title:{text:'CDF — Latency (ms)',font:{color:'#EDE0C8'}}, xaxis:{title:'ms',color:'#A89880'}, yaxis:{title:'CDF',color:'#A89880'}});
    }

    // SINR by scenario
    if (d.sinr_by_scenario) {
      const scenarios = Object.keys(d.sinr_by_scenario);
      const values = scenarios.map(s => d.sinr_by_scenario[s]);
      Plotly.newPlot('sinr-scenario-chart',
        [{type:'bar', x:scenarios, y:values, marker:{color:'#7A9E7E'}, name:'Median SINR'}],
        {...darkLayout, title:{text:'SINR by Scenario (median)',font:{color:'#EDE0C8'}}, xaxis:{color:'#A89880'}, yaxis:{title:'dB',color:'#A89880'}});
    }

    // Scenario donut
    if (d.scenario_distribution) {
      const labels = Object.keys(d.scenario_distribution);
      const values = labels.map(l => d.scenario_distribution[l]);
      Plotly.newPlot('ov-scenario-chart',
        [{type:'pie', labels, values, hole:0.4, marker:{colors:['#C9933A','#7A9E7E','#B85C38','#8B6914','#D4B896','#A89880']}}],
        {...darkLayout, margin:{t:10,b:10,l:10,r:10}, showlegend:true, legend:{font:{color:'#A89880'},orientation:'v'}});
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

  const darkLayout = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#A89880'},margin:{t:40,b:40,l:50,r:20}};
  Plotly.newPlot('agent-bar-chart',
    [{type:'bar', x:['Train','Validation','Test'], y:[ag.train[key]||0, ag.val[key]||0, ag.test[key]||0],
      marker:{color:['#C9933A','#7A9E7E','#B85C38']}, text:[(ag.train[key]||0).toFixed(3),(ag.val[key]||0).toFixed(3),(ag.test[key]||0).toFixed(3)], textposition:'outside'}],
    {...darkLayout, title:{text:`${name} — ${key.toUpperCase()} across splits`,font:{color:'#EDE0C8'}},
      yaxis:{range:[0,1.1],color:'#A89880'}, xaxis:{color:'#A89880'}});
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
  const darkLayout = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#A89880'},margin:{t:50,b:80,l:50,r:20}};
  const traces = ['train','val','test'].map((split,i) => ({
    type:'bar', name:split.charAt(0).toUpperCase()+split.slice(1),
    x:names,
    y:Object.values(agentMetrics).map(ag=>ag[split][getKey(ag)]||0),
    marker:{color:['#C9933A','#7A9E7E','#B85C38'][i]},
  }));
  Plotly.newPlot('trainval-chart', traces,
    {...darkLayout, barmode:'group', title:{text:'All Agents — Train / Val / Test',font:{color:'#EDE0C8'}},
      yaxis:{range:[-0.1,1.1],color:'#A89880'}, xaxis:{color:'#A89880'}});

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

    const darkLayout = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#A89880'},margin:{t:50,b:50,l:60,r:20}};

    Plotly.newPlot('reward-chart',
      [{x:d.episodes, y:d.rewards, mode:'lines', line:{color:'#C9933A',width:2}, name:'Total reward'},
       {x:d.episodes, y:d.rewards.map((_,i,a) => {
         const w=50, s=Math.max(0,i-w), slice=a.slice(s,i+1);
         return slice.reduce((a,b)=>a+b,0)/slice.length;
       }), mode:'lines', line:{color:'#F0C97A',width:2,dash:'dot'}, name:'50-ep moving avg'}],
      {...darkLayout, title:{text:'MAPPO Reward over 1000 Episodes',font:{color:'#EDE0C8'}},
        xaxis:{title:'Episode',color:'#A89880'}, yaxis:{title:'Total reward',color:'#A89880'}});

    Plotly.newPlot('loss-chart',
      [{x:d.episodes, y:d.actor_loss, mode:'lines', line:{color:'#7A9E7E',width:1.5}, name:'Actor loss'},
       {x:d.episodes, y:d.critic_loss, mode:'lines', line:{color:'#B85C38',width:1.5}, name:'Critic loss', yaxis:'y2'}],
      {...darkLayout, title:{text:'Actor & Critic Loss',font:{color:'#EDE0C8'}},
        xaxis:{title:'Episode',color:'#A89880'},
        yaxis:{title:'Actor loss',color:'#7A9E7E'},
        yaxis2:{title:'Critic loss',color:'#B85C38',overlaying:'y',side:'right'}});

  } catch(e) { console.error(e); }
}

// Chatbot
const chatResponses = {
  'byzantine':'A Byzantine agent is a compromised node sending malicious actions to manipulate consensus. Our system detects it using three methods: statistical outlier detection, voting disagreement tracking, and behavioral drift analysis across a sliding window.',
  'consensus':'Our consensus engine uses Byzantine-robust voting — it excludes flagged agents before tallying votes. Minimum agreement threshold is 60%. In the shared baseline, consensus accept rate is only 6.6%. Our MARL system reaches ~72%.',
  'trust':'Trust scores range from 0 to 1. They drop when an agent is flagged by the anomaly detector or violates policy checks, and slowly recover when the agent behaves consistently with the group.',
  'handover':'A handover is when a UE switches from one cell to another. Our MARL agents learn when to trigger handovers based on RSRP and SINR. The consensus engine ensures no single compromised agent can force a bad handover decision.',
  'oran':'O-RAN is the open architecture this system runs in. Our agents live at the Near-RT RIC layer — making millisecond-to-second decisions pushed down to the O-DU and O-CU.',
  'exp 3':'Experiment 3 added Byzantine injection and security modules to MARL training. The security layer correctly slows convergence by throttling learning signal during quarantine. Reward went from -2,520 → -259 over 1000 episodes. Given 2000-3000 episodes it will likely close the gap.',
  'rsrp':'RSRP (Reference Signal Received Power) measures the signal strength from the serving cell. In the Nokia dataset, RSRP values range from around -125 to -60 dBm, with the distribution centered around -90 dBm.',
  'sinr':'SINR (Signal to Interference and Noise Ratio) measures signal quality. Values above 10 dB are generally good. The dataset shows SINR drops sharply during jamming attacks — the clearest attack signature in the data.',
  'default':'Based on the RAN knowledge corpus — this relates to how the multi-agent framework coordinates decisions across cells while maintaining security against adversarial agents. Try asking about Byzantine attacks, consensus, trust scores, or O-RAN architecture.',
};

function sendChat() {
  const inp = document.getElementById('chat-input');
  const msg = inp.value.trim();
  if (!msg) return;
  const msgs = document.getElementById('chat-msgs');
  msgs.insertAdjacentHTML('beforeend', `<div class="chat-msg chat-user">${msg}</div>`);
  inp.value = '';
  setTimeout(() => {
    const lower = msg.toLowerCase();
    let reply = chatResponses.default;
    for (const [k, v] of Object.entries(chatResponses)) {
      if (lower.includes(k)) { reply = v; break; }
    }
    msgs.insertAdjacentHTML('beforeend', `<div class="chat-msg chat-bot">${reply}</div>`);
    msgs.scrollTop = msgs.scrollHeight;
  }, 500);
}

function askSuggested(btn) {
  document.getElementById('chat-input').value = btn.textContent;
  sendChat();
}

// Init
agents.forEach((_,i) => applyHexStyle(i));
renderAllTrust();
updateOverviewMetrics();
loadDataExploration();

// Poll state every 2s
pollState();
setInterval(pollState, 2000);