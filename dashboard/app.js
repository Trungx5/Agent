const API_BASE = "http://localhost:5000";

const statusPill = document.getElementById("statusPill");
const batteryValue = document.getElementById("batteryValue");
const healthValue = document.getElementById("healthValue");
const batteryBar = document.getElementById("batteryBar");
const healthBar = document.getElementById("healthBar");
const rewardValue = document.getElementById("rewardValue");
const tpValue = document.getElementById("tpValue");
const dropValue = document.getElementById("dropValue");
const epsValue = document.getElementById("epsValue");
const chartImage = document.getElementById("chartImage");
const logList = document.getElementById("logList");

const epsilonSlider = document.getElementById("epsilon");
const epsilonLabel = document.getElementById("epsilonLabel");
const solarSlider = document.getElementById("solar");
const solarLabel = document.getElementById("solarLabel");

const actionButtons = Array.from(document.querySelectorAll(".seg"));

function setStatus(text, status) {
  statusPill.textContent = text.toUpperCase();
  const colors = {
    training: "rgba(61, 214, 208, 0.2)",
    paused: "rgba(247, 127, 0, 0.2)",
    done: "rgba(144, 190, 109, 0.2)",
    idle: "rgba(151, 163, 182, 0.2)",
  };
  statusPill.style.background = colors[status] || colors.idle;
}

function setActionActive(action) {
  actionButtons.forEach((btn) => {
    const isActive = btn.dataset.action === action;
    btn.classList.toggle("active", isActive);
  });
}

function percent(val) {
  return Math.max(0, Math.min(100, val * 100));
}

async function fetchState() {
  const res = await fetch(`${API_BASE}/state`);
  if (!res.ok) return null;
  return res.json();
}

async function refresh() {
  const state = await fetchState();
  if (!state) return;

  const stats = state.stats || {};
  const control = state.control || {};

  setStatus(stats.status || "idle", stats.status || "idle");

  batteryValue.textContent = percent(stats.battery || 0).toFixed(1);
  healthValue.textContent = percent(stats.health || 0).toFixed(1);
  batteryBar.style.width = `${percent(stats.battery || 0).toFixed(1)}%`;
  healthBar.style.width = `${percent(stats.health || 0).toFixed(1)}%`;

  const currentReward = stats.reward_current ?? stats.reward ?? 0;
  rewardValue.textContent = Number(currentReward).toFixed(2);
  tpValue.textContent = stats.throughput ?? 0;
  dropValue.textContent = stats.drop_rate ?? 0;
  epsValue.textContent = (stats.epsilon ?? 0).toFixed(4);

  const actionOverride = control.action_override;
  if (actionOverride === null || actionOverride === undefined) {
    setActionActive("auto");
  } else if (actionOverride === 0) {
    setActionActive("sleep");
  } else if (actionOverride === 1) {
    setActionActive("low");
  } else if (actionOverride === 2) {
    setActionActive("high");
  }

  if (control.epsilon_override !== null && control.epsilon_override !== undefined) {
    epsilonSlider.value = Number(control.epsilon_override).toFixed(2);
    epsilonLabel.textContent = Number(control.epsilon_override).toFixed(2);
  }

  if (control.solar_override !== null && control.solar_override !== undefined) {
    solarSlider.value = Number(control.solar_override).toFixed(2);
    solarLabel.textContent = `${Number(control.solar_override).toFixed(2)}x`;
  }

  const log = state.log || [];
  logList.innerHTML = log
    .slice()
    .reverse()
    .map((entry) => {
      return `<div class="log-item">
        <span>${entry.time} ${entry.source.toUpperCase()}</span>
        <span>${entry.result}</span>
      </div>`;
    })
    .join("");

  chartImage.src = `${API_BASE}/chart?ts=${Date.now()}`;
}

async function postControl(payload) {
  await fetch(`${API_BASE}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

actionButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const action = btn.dataset.action;
    const map = { auto: "auto", sleep: "sleep", low: "low", high: "high" };
    postControl({ type: "set_action", value: map[action], source: "dashboard" });
    setActionActive(action);
  });
});

epsilonSlider.addEventListener("input", (event) => {
  epsilonLabel.textContent = Number(event.target.value).toFixed(2);
});

solarSlider.addEventListener("input", (event) => {
  solarLabel.textContent = `${Number(event.target.value).toFixed(2)}x`;
});

const applyEpsilon = document.getElementById("applyEpsilon");
applyEpsilon.addEventListener("click", () => {
  postControl({ type: "set_epsilon", value: Number(epsilonSlider.value), source: "dashboard" });
});

const applySolar = document.getElementById("applySolar");
applySolar.addEventListener("click", () => {
  postControl({ type: "set_solar", value: Number(solarSlider.value), source: "dashboard" });
});

const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");
const resetBtn = document.getElementById("resetBtn");

pauseBtn.addEventListener("click", () => postControl({ type: "pause", source: "dashboard" }));
resumeBtn.addEventListener("click", () => postControl({ type: "resume", source: "dashboard" }));
resetBtn.addEventListener("click", () => postControl({ type: "reset", source: "dashboard" }));

refresh();
setInterval(refresh, 2000);
