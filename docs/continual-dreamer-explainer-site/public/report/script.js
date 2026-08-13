const root = document.documentElement;
const storedTheme = localStorage.getItem("clwm-report-theme");

if (storedTheme) {
  root.dataset.theme = storedTheme;
}

document.querySelector("#theme-toggle").addEventListener("click", () => {
  const theme = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = theme;
  localStorage.setItem("clwm-report-theme", theme);
});

const results = window.ARROW_BASELINE_RESULTS;
const svgNamespace = "http://www.w3.org/2000/svg";

function svgElement(tag, attributes = {}, text = "") {
  const element = document.createElementNS(svgNamespace, tag);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  element.textContent = text;
  return element;
}

function formatScore(value) {
  return value.toLocaleString("en-US", { maximumFractionDigits: 1 });
}

function formatRetention(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function findEvaluation(task, epoch) {
  return task.evaluations.find((point) => point.epoch === epoch);
}

function renderTaskChart(task) {
  const card = document.createElement("article");
  card.className = "task-chart";
  const finalRaw = formatScore(task.finalRawDerived);
  card.innerHTML = `
    <header>
      <strong>${task.name}</strong>
      <span>scale ${task.rewardScale} · final raw≈${finalRaw}</span>
    </header>
  `;

  const width = 440;
  const height = 170;
  const margin = { top: 10, right: 12, bottom: 23, left: 39 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const values = task.evaluations.map((point) => point.mean);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values);
  const padding = Math.max((maxValue - minValue) * 0.08, 1);
  const yMin = minValue - (minValue < 0 ? padding : 0);
  const yMax = maxValue + padding;
  const x = (epoch) => margin.left + (epoch / 540) * plotWidth;
  const y = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;

  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${task.name} scaled evaluation return over epochs`,
  });
  svg.appendChild(svgElement("rect", {
    class: "phase-fill",
    x: x(task.trainStart),
    y: margin.top,
    width: x(task.trainEnd) - x(task.trainStart),
    height: plotHeight,
  }));

  [0, 0.5, 1].forEach((ratio) => {
    const value = yMin + ratio * (yMax - yMin);
    const yPosition = y(value);
    svg.appendChild(svgElement("line", {
      class: value === 0 ? "zero-line" : "grid-line",
      x1: margin.left,
      x2: width - margin.right,
      y1: yPosition,
      y2: yPosition,
    }));
    svg.appendChild(svgElement("text", {
      x: margin.left - 5,
      y: yPosition + 3,
      "text-anchor": "end",
    }, formatScore(value)));
  });

  [90, 180, 270, 360, 450].forEach((epoch) => {
    svg.appendChild(svgElement("line", {
      class: "boundary-line",
      x1: x(epoch),
      x2: x(epoch),
      y1: margin.top,
      y2: margin.top + plotHeight,
    }));
  });
  [0, 90, 180, 270, 360, 450, 540].forEach((epoch) => {
    svg.appendChild(svgElement("text", {
      x: x(epoch),
      y: height - 6,
      "text-anchor": epoch === 0 ? "start" : epoch === 540 ? "end" : "middle",
    }, `${epoch}`));
  });

  const path = task.evaluations
    .map((point, index) => `${index === 0 ? "M" : "L"}${x(point.epoch).toFixed(2)},${y(point.mean).toFixed(2)}`)
    .join(" ");
  svg.appendChild(svgElement("path", { class: "score-line", d: path }));
  svg.appendChild(svgElement("circle", {
    class: "final-dot",
    cx: x(task.final.epoch),
    cy: y(task.final.mean),
    r: 3.4,
  }));
  card.appendChild(svg);
  return card;
}

function interpretation(task) {
  if (task.index === results.tasks.length - 1) return ["最后任务，尚无后续干扰", "retention-moderate"];
  if (task.phasePeakRetention >= 0.85) return ["较强保持", "retention-strong"];
  if (task.phasePeakRetention >= 0.65) return ["中等下降", "retention-moderate"];
  return ["明显下降", "retention-weak"];
}

function renderRetentionTable() {
  const body = document.querySelector("#retention-table-body");
  results.tasks.forEach((task) => {
    const [label, className] = interpretation(task);
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${task.name}<br><small>scale = ${task.rewardScale}</small></td>
      <td>${formatScore(task.phasePeak.mean)} @ e${task.phasePeak.epoch}</td>
      <td>${formatScore(task.final.mean)} ± ${formatScore(task.final.std)}</td>
      <td>${formatScore(task.finalRawDerived)}</td>
      <td class="${className}">${formatRetention(task.phasePeakRetention)}</td>
      <td>${label}</td>
    `;
    body.appendChild(row);
  });
}

function renderBoundaryTable() {
  const boundaries = results.tasks.map((task) => task.trainEnd);
  const head = document.querySelector("#boundary-table-head");
  head.innerHTML = `<tr><th>任务</th>${boundaries.map((epoch) => `<th>e${epoch}</th>`).join("")}</tr>`;

  const body = document.querySelector("#boundary-table-body");
  results.tasks.forEach((task) => {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = task.name;
    row.appendChild(label);
    boundaries.forEach((epoch) => {
      const cell = document.createElement("td");
      if (epoch < task.trainEnd) {
        cell.className = "is-na";
        cell.textContent = "未训练";
      } else {
        const point = findEvaluation(task, epoch);
        const ratio = point.mean / task.phasePeak.mean;
        cell.className = "retention-cell";
        cell.textContent = formatRetention(ratio);
        cell.style.setProperty("--cell-strength", Math.min(Math.max(ratio * 24, 5), 28));
        cell.style.setProperty("--cell-color", ratio >= 0.85 ? "var(--green)" : ratio >= 0.65 ? "var(--blue)" : "var(--accent)");
      }
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function renderStageBars() {
  const names = {
    world_model: "World Model",
    actor: "Actor-Critic",
    collect: "Collection",
    eval: "Evaluation",
  };
  const container = document.querySelector("#stage-bars");
  results.resources.stageBreakdown
    .filter((stage) => stage.name !== "overhead")
    .forEach((stage) => {
      const row = document.createElement("div");
      row.className = "stage-row";
      row.innerHTML = `
        <span>${names[stage.name]}</span>
        <div class="stage-track"><div class="stage-fill" style="width: ${(stage.fraction * 100).toFixed(2)}%"></div></div>
        <strong>${stage.hours.toFixed(2)} h · ${(stage.fraction * 100).toFixed(1)}%</strong>
      `;
      container.appendChild(row);
    });
}

if (results) {
  const charts = document.querySelector("#baseline-small-multiples");
  results.tasks.forEach((task) => charts.appendChild(renderTaskChart(task)));
  renderRetentionTable();
  renderBoundaryTable();
  renderStageBars();
}
