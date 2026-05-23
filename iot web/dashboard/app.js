const CURRENT_PROTOCOL = window.location.protocol;
const CURRENT_HOSTNAME = window.location.hostname || "localhost";
const API_BASE = `${CURRENT_PROTOCOL}//${CURRENT_HOSTNAME}:5000`;
const DEVICE_WEB_BASE = `${CURRENT_PROTOCOL}//${CURRENT_HOSTNAME}:8081`;
const DEVICE_WEB_BASES = [...new Set(["", DEVICE_WEB_BASE, "http://localhost:8081", "http://127.0.0.1:8081"])];
const DEVICE_WRITE_BASES = [...new Set(["", DEVICE_WEB_BASE, "http://localhost:8081", "http://127.0.0.1:8081"])];
const API_BASES = [...new Set([API_BASE, "http://localhost:5000", "http://127.0.0.1:5000"])];
const MAIN_BATTERY_MAX = 500;
const AUX_BATTERY_MAX = 200;
const POLL_INTERVAL_MS = 2000;
const MAX_POINTS = 45;
const USAGE_PER_SEC_MIN = 0;
const USAGE_PER_SEC_MAX = 30;
const ENERGY_PER_PACKET_MIN = 1;
const ENERGY_PER_PACKET_MAX = 10;
const CONSUMPTION_SYNC_DEBOUNCE_MS = 180;

const state = {
  dataSource: "DEVICE_WEB",
  connection: {
    offline: false,
    lastErrorMessage: ""
  },
  current: {
    battery: 0,
    auxBattery: 0,
    auxToMainEnabled: false,
    solar: 0,
    energyPerPacket: 0,
    usagePerSec: 0,
    dataGen: 0,
    bufferMax: 0,
    totalDataGen: 0,
    bufferOccupancy: 0,
    systemError: false,
    systemDead: false,
    episode: 0,
    source: "iot-device-web"
  },
  metrics: {
    bufferSentTotal: 0,
    lastSentAmount: 0
  },
  series: {
    battery: [],
    solar: [],
    usage: [],
    dataGen: [],
    throughput: []
  }
};

const el = {
  batteryCanvas: document.getElementById("batteryCanvas"),
  solarCanvas: document.getElementById("solarCanvas"),
  usageCanvas: document.getElementById("usageCanvas"),
  dataGenCanvas: document.getElementById("dataGenCanvas"),
  throughputCanvas: document.getElementById("throughputCanvas"),
  batteryLiveValue: document.getElementById("batteryLiveValue"),
  solarLiveValue: document.getElementById("solarLiveValue"),
  usageLiveValue: document.getElementById("usageLiveValue"),
  dataGenLiveValue: document.getElementById("dataGenLiveValue"),
  throughputLiveValue: document.getElementById("throughputLiveValue"),
  batteryNowText: document.getElementById("batteryNowText"),
  batteryMinText: document.getElementById("batteryMinText"),
  batteryMaxChartText: document.getElementById("batteryMaxChartText"),
  batteryAvgText: document.getElementById("batteryAvgText"),
  solarNowText: document.getElementById("solarNowText"),
  solarMinText: document.getElementById("solarMinText"),
  solarMaxText: document.getElementById("solarMaxText"),
  solarAvgText: document.getElementById("solarAvgText"),
  usageNowText: document.getElementById("usageNowText"),
  usageMinText: document.getElementById("usageMinText"),
  usageMaxText: document.getElementById("usageMaxText"),
  usageAvgText: document.getElementById("usageAvgText"),
  dataGenNowText: document.getElementById("dataGenNowText"),
  dataGenMinText: document.getElementById("dataGenMinText"),
  dataGenMaxText: document.getElementById("dataGenMaxText"),
  dataGenAvgText: document.getElementById("dataGenAvgText"),
  batteryText: document.getElementById("batteryText"),
  auxBatteryText: document.getElementById("auxBatteryText"),
  auxModeText: document.getElementById("auxModeText"),
  energyPacketText: document.getElementById("energyPacketText"),
  bufferMaxText: document.getElementById("bufferMaxText"),
  totalDataGenText: document.getElementById("totalDataGenText"),
  bufferOccupancyText: document.getElementById("bufferOccupancyText"),
  bufferSentText: document.getElementById("bufferSentText"),
  systemErrorText: document.getElementById("systemErrorText"),
  systemHealthText: document.getElementById("systemHealthText"),
  batteryDeltaText: document.getElementById("batteryDeltaText"),
  overflowAmountText: document.getElementById("overflowAmountText"),
  estimatedPacketsText: document.getElementById("estimatedPacketsText"),
  deadBanner: document.getElementById("deadBanner"),
  liveBadge: document.getElementById("liveBadge"),
  liveText: document.getElementById("liveText"),
  logList: document.getElementById("logList"),
  sourceText: document.getElementById("sourceText"),
  openDeviceBtn: document.getElementById("openDeviceBtn"),
  bufferSendInput: document.getElementById("bufferSendInput"),
  bufferSendValue: document.getElementById("bufferSendValue"),
  sendBufferBtn: document.getElementById("sendBufferBtn"),
  enableAuxChargeBtn: document.getElementById("enableAuxChargeBtn"),
  disableAuxChargeBtn: document.getElementById("disableAuxChargeBtn"),
  usagePerSecControl: document.getElementById("usagePerSecControl"),
  usagePerSecControlValue: document.getElementById("usagePerSecControlValue"),
  energyPerPacketControl: document.getElementById("energyPerPacketControl"),
  energyPerPacketControlValue: document.getElementById("energyPerPacketControlValue")
};

let consumptionSyncTimer = null;

function clampInt(value, min, max, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.round(num)));
}

function syncConsumptionControls() {
  const usage = clampInt(state.current.usagePerSec, USAGE_PER_SEC_MIN, USAGE_PER_SEC_MAX, USAGE_PER_SEC_MIN);
  const packetCost = clampInt(state.current.energyPerPacket, ENERGY_PER_PACKET_MIN, ENERGY_PER_PACKET_MAX, ENERGY_PER_PACKET_MIN);

  if (el.usagePerSecControl) {
    el.usagePerSecControl.value = String(usage);
  }
  if (el.usagePerSecControlValue) {
    el.usagePerSecControlValue.textContent = String(usage);
  }
  if (el.energyPerPacketControl) {
    el.energyPerPacketControl.value = String(packetCost);
  }
  if (el.energyPerPacketControlValue) {
    el.energyPerPacketControlValue.textContent = String(packetCost);
  }
}

function getConsumptionControlValues() {
  const nextUsage = clampInt(el.usagePerSecControl?.value, USAGE_PER_SEC_MIN, USAGE_PER_SEC_MAX, USAGE_PER_SEC_MIN);
  const nextPacketCost = clampInt(el.energyPerPacketControl?.value, ENERGY_PER_PACKET_MIN, ENERGY_PER_PACKET_MAX, ENERGY_PER_PACKET_MIN);
  return { nextUsage, nextPacketCost };
}

function scheduleConsumptionSync() {
  if (!el.usagePerSecControl || !el.energyPerPacketControl) {
    return;
  }

  if (consumptionSyncTimer) {
    clearTimeout(consumptionSyncTimer);
  }

  consumptionSyncTimer = setTimeout(async () => {
    consumptionSyncTimer = null;
    const { nextUsage, nextPacketCost } = getConsumptionControlValues();

    try {
      await postDeviceState({ usage_per_sec: nextUsage, energy_per_packet: nextPacketCost });
      state.current.usagePerSec = nextUsage;
      state.current.energyPerPacket = nextPacketCost;
      render();
    } catch (error) {
      appendLog(`Cap nhat consumption loi: ${error.message}`);
    }
  }, CONSUMPTION_SYNC_DEBOUNCE_MS);
}

function syncBufferSendSlider() {
  const totalDataGen = Math.max(0, Math.floor(Number(state.current.totalDataGen) || 0));
  const currentValue = Math.max(0, Math.floor(Number(el.bufferSendInput.value) || 0));
  const nextValue = Math.min(currentValue, totalDataGen);

  el.bufferSendInput.min = "0";
  el.bufferSendInput.max = String(totalDataGen);
  el.bufferSendInput.value = String(nextValue);
  el.bufferSendInput.disabled = totalDataGen <= 0;
  el.sendBufferBtn.disabled = totalDataGen <= 0;

  if (el.bufferSendValue) {
    el.bufferSendValue.textContent = `${nextValue} / ${totalDataGen}`;
  }
}

function appendLog(message) {
  const now = new Date();
  const ts = now.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const li = document.createElement("li");
  li.textContent = `[${ts}] ${message}`;
  el.logList.prepend(li);
  while (el.logList.children.length > 45) {
    el.logList.removeChild(el.logList.lastChild);
  }
}

function pushSeries(series, value) {
  series.push(value);
  if (series.length > MAX_POINTS) {
    series.shift();
  }
}

function drawLineChart(canvas, points, color, minY, maxY) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(112, 182, 230, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    const y = (h / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (!points.length) {
    return;
  }

  const stepX = points.length > 1 ? w / (points.length - 1) : w;
  const spread = (maxY - minY) || 1;

  ctx.beginPath();
  points.forEach((value, idx) => {
    const clamped = Math.max(minY, Math.min(maxY, value));
    const x = idx * stepX;
    const y = h - ((clamped - minY) / spread) * (h - 12) - 6;
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.stroke();
}

function getSeriesStats(series) {
  if (!series.length) {
    return { now: 0, min: 0, max: 0, avg: 0 };
  }
  const now = series[series.length - 1];
  const min = Math.min(...series);
  const max = Math.max(...series);
  const avg = series.reduce((sum, value) => sum + value, 0) / series.length;
  return { now, min, max, avg };
}

function renderSeriesStats() {
  const battery = getSeriesStats(state.series.battery);
  const solar = getSeriesStats(state.series.solar);
  const usage = getSeriesStats(state.series.usage);
  const dataGen = getSeriesStats(state.series.dataGen);

  if (el.batteryLiveValue) {
    el.batteryLiveValue.textContent = battery.now.toFixed(0);
  }
  if (el.solarLiveValue) {
    el.solarLiveValue.textContent = solar.now.toFixed(0);
  }
  if (el.usageLiveValue) {
    el.usageLiveValue.textContent = usage.now.toFixed(0);
  }
  if (el.dataGenLiveValue) {
    el.dataGenLiveValue.textContent = dataGen.now.toFixed(0);
  }
  if (el.throughputLiveValue) {
    el.throughputLiveValue.textContent = state.metrics.lastSentAmount.toFixed(0);
  }

  if (el.batteryNowText) {
    el.batteryNowText.textContent = battery.now.toFixed(1);
    el.batteryMinText.textContent = battery.min.toFixed(1);
    el.batteryMaxChartText.textContent = battery.max.toFixed(1);
    el.batteryAvgText.textContent = battery.avg.toFixed(1);
  }

  if (el.solarNowText) {
    el.solarNowText.textContent = solar.now.toFixed(1);
    el.solarMinText.textContent = solar.min.toFixed(1);
    el.solarMaxText.textContent = solar.max.toFixed(1);
    el.solarAvgText.textContent = solar.avg.toFixed(1);
  }

  if (el.usageNowText) {
    el.usageNowText.textContent = usage.now.toFixed(1);
    el.usageMinText.textContent = usage.min.toFixed(1);
    el.usageMaxText.textContent = usage.max.toFixed(1);
    el.usageAvgText.textContent = usage.avg.toFixed(1);
  }

  if (el.dataGenNowText) {
    el.dataGenNowText.textContent = dataGen.now.toFixed(1);
    el.dataGenMinText.textContent = dataGen.min.toFixed(1);
    el.dataGenMaxText.textContent = dataGen.max.toFixed(1);
    el.dataGenAvgText.textContent = dataGen.avg.toFixed(1);
  }
}

function render() {
  const current = state.current;
  const systemDead = current.battery <= 0.5;
  const hasSystemError = current.systemError || systemDead;

  el.batteryText.textContent = `${current.battery.toFixed(0)}/${MAIN_BATTERY_MAX}`;
  el.auxBatteryText.textContent = `${current.auxBattery.toFixed(0)}/${AUX_BATTERY_MAX}`;
  el.auxModeText.textContent = current.auxToMainEnabled ? "ON" : "OFF";
  el.energyPacketText.textContent = current.energyPerPacket.toFixed(0);
  el.bufferMaxText.textContent = current.bufferMax.toFixed(0);
  el.totalDataGenText.textContent = current.totalDataGen.toFixed(0);
  el.bufferOccupancyText.textContent = current.bufferOccupancy.toFixed(0);
  el.bufferSentText.textContent = state.metrics.bufferSentTotal.toFixed(0);
  el.systemErrorText.textContent = hasSystemError ? "YES" : "NO";
  el.systemErrorText.classList.toggle("dead", hasSystemError);

  el.systemHealthText.textContent = systemDead ? "DEAD" : "ALIVE";
  el.systemHealthText.classList.toggle("dead", systemDead);
  el.systemHealthText.classList.toggle("alive", !systemDead);

  if (el.deadBanner) {
    el.deadBanner.classList.toggle("hidden", !systemDead);
  }

  const batteryDelta = current.solar - current.usagePerSec;
  const overflowAmount = Math.max(0, current.totalDataGen - current.bufferMax);
  const packetCost = Math.max(1, Math.floor(current.energyPerPacket || 1));
  const estimatedPackets = Math.floor(Math.max(0, current.battery) / packetCost);
  el.batteryDeltaText.textContent = batteryDelta.toFixed(1);
  el.overflowAmountText.textContent = overflowAmount.toFixed(0);
  el.estimatedPacketsText.textContent = estimatedPackets.toFixed(0);

  document.body.classList.toggle("system-dead", systemDead);
  el.sourceText.textContent = state.dataSource === "DEVICE_WEB" ? "SOURCE: /device-state proxy" : "SOURCE: API :5000";

  drawLineChart(el.batteryCanvas, state.series.battery, "#7bf8e4", 0, MAIN_BATTERY_MAX);
  drawLineChart(el.solarCanvas, state.series.solar, "#67cfff", 0, 50);
  drawLineChart(el.usageCanvas, state.series.usage, "#ffc96e", 0, 30);
  drawLineChart(el.dataGenCanvas, state.series.dataGen, "#8cff9e", 0, 10);
  const throughputMax = Math.max(20, ...state.series.throughput, state.metrics.lastSentAmount + 2);
  drawLineChart(el.throughputCanvas, state.series.throughput, "#1f9e67", 0, throughputMax);
  syncBufferSendSlider();
  syncConsumptionControls();
  renderSeriesStats();
}

function setLiveStatus(isOnline) {
  if (isOnline) {
    el.liveBadge.classList.remove("offline");
    el.liveText.textContent = "LIVE";
    return;
  }
  el.liveBadge.classList.add("offline");
  el.liveText.textContent = "OFFLINE";
}

function normalizePayload(payload = {}) {
  const batteryRaw = payload.battery ?? state.current.battery;
  const solarRaw = payload.solar ?? state.current.solar;
  const batteryValue = Number(batteryRaw);
  const systemDead = batteryValue <= 0.5;
  const payloadSystemError = Boolean(payload.system_error ?? state.current.systemError);
  const totalDataGenValue = Number(payload.total_data_gen ?? state.current.totalDataGen);
  const bufferMaxValue = Number(payload.buffer_max ?? state.current.bufferMax);
  const queueRaw = Number(payload.queue);
  const bufferOccupancyValue = Number.isFinite(queueRaw)
    ? Math.max(0, Math.min(bufferMaxValue, queueRaw))
    : Math.max(0, Math.min(bufferMaxValue, totalDataGenValue));
  const auxBatteryValue = Number(payload.aux_battery ?? state.current.auxBattery);
  const auxToMainEnabledValue = Boolean(payload.aux_to_main_enabled ?? state.current.auxToMainEnabled);

  return {
    battery: batteryValue,
    auxBattery: auxBatteryValue,
    auxToMainEnabled: auxToMainEnabledValue,
    solar: Number(solarRaw),
    energyPerPacket: clampInt(payload.energy_per_packet ?? state.current.energyPerPacket, ENERGY_PER_PACKET_MIN, ENERGY_PER_PACKET_MAX, ENERGY_PER_PACKET_MIN),
    usagePerSec: Number(payload.usage_per_sec ?? state.current.usagePerSec),
    dataGen: Number(payload.data_gen ?? state.current.dataGen),
    bufferMax: bufferMaxValue,
    totalDataGen: totalDataGenValue,
    bufferOccupancy: bufferOccupancyValue,
    systemError: payloadSystemError || systemDead,
    systemDead,
    episode: Number(payload.episode ?? state.current.episode),
    source: String(payload.source ?? state.current.source ?? "iot-device-web")
  };
}

async function fetchStateFromCandidates(candidates, endpoint) {
  let lastError = new Error("unreachable");
  for (const base of candidates) {
    try {
      const target = base ? `${base}${endpoint}` : endpoint;
      const response = await fetch(target, {
        method: "GET",
        headers: { "Accept": "application/json" }
      });
      if (response.ok) {
        return { payload: await response.json(), base: base || "same-origin" };
      }
      lastError = new Error(`${target} -> ${response.status}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function getStateFromApi() {
  try {
    const stateResponse = await fetchStateFromCandidates(API_BASES, "/state");
    return normalizePayload(stateResponse.payload);
  } catch (_error) {
    const statsResponse = await fetchStateFromCandidates(API_BASES, "/stats");
    return normalizePayload(statsResponse.payload);
  }
}

async function postDeviceState(payload) {
  let lastError = new Error("Khong the cap nhat device-state");
  for (const base of DEVICE_WRITE_BASES) {
    try {
      const response = await fetch(`${base}/device-state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (response.ok) {
        return response.json().catch(() => ({}));
      }
      const body = await response.text();
      lastError = new Error(body || `POST /device-state failed (${response.status})`);
    } catch (_error) {
      lastError = _error;
    }
  }
  throw lastError;
}

async function tick() {
  try {
    const wasDead = state.current.battery <= 0;
    let next;
    try {
      const deviceResponse = await fetchStateFromCandidates(DEVICE_WEB_BASES, "/device-state");
      next = normalizePayload(deviceResponse.payload);
      if (state.dataSource !== "DEVICE_WEB") {
        appendLog(`Đã chuyển lại nguồn dữ liệu từ IoT Device Web (${deviceResponse.base})`);
      }
      state.dataSource = "DEVICE_WEB";
    } catch (deviceError) {
      next = await getStateFromApi();
      if (state.dataSource !== "API") {
        appendLog("IoT Device Web lỗi, fallback sang Flask API (:5000)");
      }
      state.dataSource = "API";
    }

    state.current = {
      ...state.current,
      ...next
    };

    pushSeries(state.series.battery, state.current.battery);
    pushSeries(state.series.solar, state.current.solar);
    pushSeries(state.series.usage, state.current.usagePerSec);
    pushSeries(state.series.dataGen, state.current.dataGen);
    const isDead = state.current.battery <= 0;
    if (!wasDead && isDead) {
      appendLog("CRITICAL: Battery ve 0, he thong chet.");
    }
    if (wasDead && !isDead) {
      appendLog("INFO: He thong da khoi phuc do Battery > 0.");
    }

    if (state.connection.offline) {
      appendLog("KET NOI DA KHOI PHUC.");
    }
    state.connection.offline = false;
    state.connection.lastErrorMessage = "";

    setLiveStatus(true);
    render();
  } catch (error) {
    setLiveStatus(false);
    const errorMessage = `Mat ket noi ca Device Web va API (${error.message})`;
    if (!state.connection.offline || state.connection.lastErrorMessage !== errorMessage) {
      appendLog(errorMessage);
    }
    state.connection.offline = true;
    state.connection.lastErrorMessage = errorMessage;
  }
}

function bindEvents() {
  el.sendBufferBtn.addEventListener("click", async () => {
    const amount = Number(el.bufferSendInput.value || 0);

    if (!Number.isInteger(amount) || amount <= 0) {
      appendLog("So luong gui buffer phai la so nguyen duong.");
      return;
    }

    const beforeQueue = Math.max(0, Number(state.current.bufferOccupancy) || 0);
    const beforeTotalDataGen = Math.max(0, Number(state.current.totalDataGen) || 0);
    const beforeBattery = Math.max(0, Number(state.current.battery) || 0);
    const packetCost = Math.max(1, Math.floor(Number(state.current.energyPerPacket) || 1));
    const maxSendByBattery = Math.floor(beforeBattery / packetCost);
    const actualSent = Math.min(beforeQueue, amount, maxSendByBattery);
    const batteryUsed = actualSent * packetCost;
    const nextTotalDataGen = Math.max(0, beforeTotalDataGen - actualSent);
    const nextQueue = Math.max(0, Math.min(state.current.bufferMax, nextTotalDataGen));
    const nextBattery = Math.max(0, beforeBattery - batteryUsed);

    if (actualSent <= 0) {
      appendLog(`Khong the gui goi tin: buffer hoac pin khong du (cost/goi=${packetCost}).`);
      return;
    }

    try {
      await postDeviceState({ total_data_gen: nextTotalDataGen, queue: nextQueue, battery: nextBattery });
      state.current.totalDataGen = nextTotalDataGen;
      state.current.bufferOccupancy = nextQueue;
      state.current.battery = nextBattery;
      state.metrics.bufferSentTotal += actualSent;
      state.metrics.lastSentAmount = actualSent;
      pushSeries(state.series.throughput, actualSent);
      render();
      appendLog(`Gui Data Buffer: yeu cau ${amount}, gui ${actualSent}, cost/goi=${packetCost}, pin mat=${batteryUsed}. TotalDataGen: ${beforeTotalDataGen} -> ${nextTotalDataGen}, Queue: ${beforeQueue} -> ${nextQueue}, MainBattery: ${beforeBattery} -> ${nextBattery}.`);
      syncBufferSendSlider();
    } catch (error) {
      appendLog(`Gui Data Buffer loi: ${error.message}`);
    }
  });

  el.bufferSendInput.addEventListener("input", () => {
    const amount = Math.max(0, Math.floor(Number(el.bufferSendInput.value) || 0));
    const maxAmount = Math.max(0, Math.floor(Number(el.bufferSendInput.max) || 0));
    if (el.bufferSendValue) {
      el.bufferSendValue.textContent = `${amount} / ${maxAmount}`;
    }
  });

  el.enableAuxChargeBtn.addEventListener("click", async () => {
    try {
      await postDeviceState({ aux_to_main_enabled: true });
      state.current.auxToMainEnabled = true;
      render();
      appendLog("Agent mode: bat nap pin phu vao pin chinh (ON).");
    } catch (error) {
      appendLog(`Bat aux-charge loi: ${error.message}`);
    }
  });

  el.disableAuxChargeBtn.addEventListener("click", async () => {
    try {
      await postDeviceState({ aux_to_main_enabled: false });
      state.current.auxToMainEnabled = false;
      render();
      appendLog("Agent mode: tat nap pin phu vao pin chinh (OFF).");
    } catch (error) {
      appendLog(`Tat aux-charge loi: ${error.message}`);
    }
  });

  if (el.usagePerSecControl) {
    el.usagePerSecControl.addEventListener("input", () => {
      const nextValue = clampInt(el.usagePerSecControl.value, USAGE_PER_SEC_MIN, USAGE_PER_SEC_MAX, USAGE_PER_SEC_MIN);
      el.usagePerSecControlValue.textContent = String(nextValue);
      scheduleConsumptionSync();
    });
  }

  if (el.energyPerPacketControl) {
    el.energyPerPacketControl.addEventListener("input", () => {
      const nextValue = clampInt(el.energyPerPacketControl.value, ENERGY_PER_PACKET_MIN, ENERGY_PER_PACKET_MAX, ENERGY_PER_PACKET_MIN);
      el.energyPerPacketControlValue.textContent = String(nextValue);
      scheduleConsumptionSync();
    });
  }

  el.openDeviceBtn.addEventListener("click", () => {
    window.open(`${DEVICE_WEB_BASE}`, "_blank", "noopener,noreferrer");
  });
}

function bootstrap() {
  bindEvents();
  tick();
  setInterval(tick, POLL_INTERVAL_MS);
}

bootstrap();
