const API = `${window.location.protocol}//${window.location.hostname || "localhost"}:8081/device-state`;
const RANGE_MIN = 0;
const BATTERY_MAX = 500;
const AUX_BATTERY_MAX = 200;
const SOLAR_MAX = 50;
const PARAM_MAX = 10;
const USAGE_PER_SEC_MAX = 30;
const TICK_MS = 1000;
const MANUAL_EDIT_GRACE_MS = 1400;
const AUX_TO_MAIN_TRANSFER_MAX = 25;

const el = {
  battery: document.getElementById("battery"),
  solar: document.getElementById("solar"),
  energyPerPacket: document.getElementById("energyPerPacket"),
  usagePerSec: document.getElementById("usagePerSec"),
  dataGen: document.getElementById("dataGen"),
  bufferMax: document.getElementById("bufferMax"),
  auxToMainEnabled: document.getElementById("auxToMainEnabled"),
  batteryValue: document.getElementById("batteryValue"),
  solarValue: document.getElementById("solarValue"),
  energyPacketValue: document.getElementById("energyPacketValue"),
  usagePerSecValue: document.getElementById("usagePerSecValue"),
  dataGenValue: document.getElementById("dataGenValue"),
  totalDataGen: document.getElementById("totalDataGen"),
  bufferMaxView: document.getElementById("bufferMaxView"),
  bufferOccupancy: document.getElementById("bufferOccupancy"),
  auxBatteryValue: document.getElementById("auxBatteryValue"),
  auxModeText: document.getElementById("auxModeText"),
  systemHealthText: document.getElementById("systemHealthText"),
  statusCard: document.getElementById("statusCard"),
  errorText: document.getElementById("errorText"),
  deadText: document.getElementById("deadText"),
  resetTestBtn: document.getElementById("resetTestBtn")
};

const sim = {
  battery: clamp0To1000(el.battery ? el.battery.value : 500),
  solarInput: clamp0To50(el.solar ? el.solar.value : 20),
  energyPerPacket: clamp1To10(el.energyPerPacket ? el.energyPerPacket.value : 3),
  usagePerSec: clamp0To30(el.usagePerSec ? el.usagePerSec.value : 5),
  dataGen: clamp1To10(el.dataGen ? el.dataGen.value : 2),
  bufferMax: clampPositive(el.bufferMax ? el.bufferMax.value : 120),
  auxBattery: clamp0To200(0),
  auxToMainEnabled: false,
  totalDataGen: 0,
  bufferOccupancy: 0,
  systemError: false,
  systemDead: false,
  tick: 0
};

let lastManualEditAt = 0;
let publishTimer = null;

function readSliderValue(node, fallback) {
  if (!node) {
    return fallback;
  }
  return node.value;
}

function setText(node, value) {
  if (node) {
    node.textContent = String(value);
  }
}

function setValue(node, value) {
  if (node) {
    node.value = String(value);
  }
}

function clamp0To1000(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return RANGE_MIN;
  }
  return Math.max(RANGE_MIN, Math.min(BATTERY_MAX, Math.round(n)));
}

function clamp0To50(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 0;
  }
  return Math.max(0, Math.min(SOLAR_MAX, Math.round(n)));
}

function clamp0To200(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 0;
  }
  return Math.max(0, Math.min(AUX_BATTERY_MAX, Math.round(n)));
}

function clamp1To10(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 1;
  }
  return Math.max(1, Math.min(PARAM_MAX, Math.round(n)));
}

function clamp0To30(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 0;
  }
  return Math.max(0, Math.min(USAGE_PER_SEC_MAX, Math.round(n)));
}

function clampPositive(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 1;
  }
  return Math.max(1, Math.round(n));
}

function getPayload() {
  return {
    battery: sim.battery,
    aux_battery: sim.auxBattery,
    solar: sim.solarInput,
    queue: sim.bufferOccupancy,
    buffer_max: sim.bufferMax,
    energy_per_packet: sim.energyPerPacket,
    data_gen: sim.dataGen,
    total_data_gen: sim.totalDataGen,
    system_error: sim.systemError,
    system_dead: sim.systemDead,
    throughput: 0,
    reward: 0,
    episode: sim.tick,
    usage_per_sec: sim.usagePerSec,
    aux_to_main_enabled: sim.auxToMainEnabled,
    mode: "auto",
    paused: false,
    source: "iot-device-web"
  };
}

function syncUI() {
  setValue(el.battery, sim.battery);
  setText(el.batteryValue, sim.battery);
  setText(el.solarValue, sim.solarInput);
  setText(el.energyPacketValue, sim.energyPerPacket);
  setText(el.usagePerSecValue, sim.usagePerSec);
  setText(el.dataGenValue, sim.dataGen);
  setText(el.totalDataGen, sim.totalDataGen);
  setText(el.bufferMaxView, sim.bufferMax);
  setText(el.bufferOccupancy, sim.bufferOccupancy);
  setText(el.auxBatteryValue, sim.auxBattery);
  setText(el.auxModeText, sim.auxToMainEnabled ? "ON" : "OFF");
  setText(el.systemHealthText, sim.systemDead ? "DEAD" : "ALIVE");
  if (el.auxToMainEnabled) {
    el.auxToMainEnabled.checked = sim.auxToMainEnabled;
  }

  if (el.statusCard) {
    el.statusCard.classList.toggle("error", sim.systemError);
    el.statusCard.classList.toggle("dead", sim.systemDead);
  }
  if (el.errorText) {
    el.errorText.classList.toggle("hidden", !sim.systemError);
  }
  if (el.deadText) {
    el.deadText.classList.toggle("hidden", !sim.systemDead);
  }
  if (el.systemHealthText) {
    el.systemHealthText.classList.toggle("dead", sim.systemDead);
  }
}

async function publishState() {
  const payload = getPayload();

  const response = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "publish failed");
  }
}

async function loadCurrentState() {
  const response = await fetch(API, { headers: { "Accept": "application/json" } });
  if (!response.ok) {
    return;
  }

  const data = await response.json();
  sim.battery = clamp0To1000(data.battery ?? sim.battery);
  sim.solarInput = clamp0To50(data.solar ?? sim.solarInput);
  sim.energyPerPacket = clamp1To10(data.energy_per_packet ?? sim.energyPerPacket);
  sim.usagePerSec = clamp0To30(data.usage_per_sec ?? sim.usagePerSec);
  sim.dataGen = clamp1To10(data.data_gen ?? sim.dataGen);
  sim.bufferMax = clampPositive(data.buffer_max ?? sim.bufferMax);
  sim.auxBattery = clamp0To200(data.aux_battery ?? sim.auxBattery);
  sim.auxToMainEnabled = Boolean(data.aux_to_main_enabled ?? sim.auxToMainEnabled);
  sim.totalDataGen = Math.max(0, Math.round(Number(data.total_data_gen ?? sim.totalDataGen) || 0));
  sim.bufferOccupancy = Math.max(0, Math.min(sim.bufferMax, sim.totalDataGen));
  sim.systemError = sim.totalDataGen > sim.bufferMax;
  sim.systemDead = sim.battery <= 0;

  setValue(el.battery, sim.battery);
  setValue(el.solar, sim.solarInput);
  setValue(el.energyPerPacket, sim.energyPerPacket);
  setValue(el.usagePerSec, sim.usagePerSec);
  setValue(el.dataGen, sim.dataGen);
  setValue(el.bufferMax, sim.bufferMax);
  syncUI();
}

function schedulePublish(delayMs = 120) {
  if (publishTimer) {
    clearTimeout(publishTimer);
  }
  publishTimer = setTimeout(async () => {
    publishTimer = null;
    try {
      await publishState();
    } catch (_error) {
      // Ignore transient publish errors while user is editing controls.
    }
  }, delayMs);
}

function runSimulationStep() {
  sim.tick += 1;
  sim.solarInput = clamp0To50(readSliderValue(el.solar, sim.solarInput));
  sim.energyPerPacket = clamp1To10(readSliderValue(el.energyPerPacket, sim.energyPerPacket));
  sim.usagePerSec = clamp0To30(readSliderValue(el.usagePerSec, sim.usagePerSec));
  sim.dataGen = clamp1To10(readSliderValue(el.dataGen, sim.dataGen));
  sim.bufferMax = clampPositive(readSliderValue(el.bufferMax, sim.bufferMax));

  sim.auxBattery = Math.min(AUX_BATTERY_MAX, sim.auxBattery + sim.solarInput);
  if (sim.auxToMainEnabled && sim.battery < BATTERY_MAX && sim.auxBattery > 0) {
    const transfer = Math.min(AUX_TO_MAIN_TRANSFER_MAX, sim.auxBattery, BATTERY_MAX - sim.battery);
    sim.auxBattery -= transfer;
    sim.battery += transfer;
  }

  sim.battery = Math.max(0, sim.battery - sim.usagePerSec);

  sim.totalDataGen += sim.dataGen;
  sim.bufferOccupancy = Math.min(sim.bufferMax, sim.totalDataGen);
  sim.systemError = sim.totalDataGen > sim.bufferMax;
  sim.systemDead = sim.battery <= 0;

  syncUI();
}

function bind() {
  [el.battery, el.solar, el.energyPerPacket, el.usagePerSec, el.dataGen, el.bufferMax].filter(Boolean).forEach((node) => {
    node.addEventListener("input", () => {
      lastManualEditAt = Date.now();
      sim.battery = clamp0To1000(readSliderValue(el.battery, sim.battery));
      sim.solarInput = clamp0To50(readSliderValue(el.solar, sim.solarInput));
      sim.energyPerPacket = clamp1To10(readSliderValue(el.energyPerPacket, sim.energyPerPacket));
      sim.usagePerSec = clamp0To30(readSliderValue(el.usagePerSec, sim.usagePerSec));
      sim.dataGen = clamp1To10(readSliderValue(el.dataGen, sim.dataGen));
      sim.bufferMax = clampPositive(readSliderValue(el.bufferMax, sim.bufferMax));
      sim.bufferOccupancy = Math.min(sim.bufferMax, sim.totalDataGen);
      sim.systemError = sim.totalDataGen > sim.bufferMax;
      sim.systemDead = sim.battery <= 0;
      syncUI();
      schedulePublish();
    });
  });

  if (el.auxToMainEnabled) {
    el.auxToMainEnabled.addEventListener("change", () => {
      lastManualEditAt = Date.now();
      sim.auxToMainEnabled = Boolean(el.auxToMainEnabled.checked);
      syncUI();
      schedulePublish();
    });
  }

  if (el.resetTestBtn) {
    el.resetTestBtn.addEventListener("click", async () => {
      sim.totalDataGen = 0;
      sim.bufferOccupancy = 0;
      sim.auxBattery = 0;
      sim.systemError = false;
      sim.systemDead = sim.battery <= 0;
      sim.tick = 0;
      syncUI();
      try {
        await publishState();
      } catch (_error) {
        // Ignore temporary publish errors while resetting local test state.
      }
    });
  }
}

async function bootstrap() {
  bind();
  await loadCurrentState();
  syncUI();
  setInterval(async () => {
    const canPullRemote = Date.now() - lastManualEditAt > MANUAL_EDIT_GRACE_MS;
    if (canPullRemote) {
      try {
        await loadCurrentState();
      } catch (_error) {
        // Keep local simulation running if transient read fails.
      }
    }
    runSimulationStep();
    try {
      await publishState();
    } catch (_error) {
      // Keep simulation running even if publish fails transiently.
    }
  }, TICK_MS);
}

bootstrap();
