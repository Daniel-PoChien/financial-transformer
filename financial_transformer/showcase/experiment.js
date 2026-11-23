"use strict";

const get = (id) => document.getElementById(id);
const canvas = get("chart");
const ctx = canvas.getContext("2d");
const buttons = [get("up"), get("down")];
let index = 0;
let selected = null;
let score = 0;

const translations = {
  en: {
    title: "Five Minutes Later? | A Stock Prediction Game",
    description: "Guess where Apple stock goes in five minutes, then compare your answer with an AI forecast and the real price.",
    nav: "Main navigation", language: "Language", choices: "Guess the price direction",
    axis: ["10 min ago", "Now", "+5 min"],
    round: (number, total) => `Round ${number} of ${total}`,
    context: (time, price) => `${time} New York time · Current price US$${price}`,
    history: (prices) => `Recent prices in US dollars: ${prices.join(", ")}.`,
    chartResult: (actual, ai) => ` Five minutes later: actual US$${actual}, AI forecast US$${ai}.`,
    guessResult: (correct, aiCorrect) => `${correct ? "You got it!" : "The price went the other way."} ${aiCorrect ? "The AI got the direction right." : "The AI guessed the wrong direction."}`,
    prices: (up, actual, ai) => `The price (green line) ${up ? "rose" : "fell"} to US$${actual}. The AI (orange dashed line) guessed US$${ai}.`,
    total: (score, count) => `You got ${score} out of ${count} right.`,
    next: "Next round →", replay: "Play again",
  },
  "zh-Hant": {
    title: "五分鐘後？｜股價預測小實驗",
    description: "用三個真實的 AAPL 歷史片段，探索人與 AI 如何猜五分鐘後的價格。",
    nav: "頁面導覽", language: "語言", choices: "選擇價格方向",
    axis: ["10 分鐘前", "現在", "+5 分鐘"],
    round: (number, total) => `第 ${number} 題 / 共 ${total} 題`,
    context: (time, price) => `${time} 紐約時間 · 當時股價 US$${price}`,
    history: (prices) => `歷史價格（美元）${prices.join("、")}。`,
    chartResult: (actual, ai) => `五分鐘後實際 ${actual}，AI 預測 ${ai}。`,
    guessResult: (correct, aiCorrect) => `${correct ? "你猜對了！" : "這次方向猜反了。"} AI ${aiCorrect ? "猜中了方向。" : "猜反了方向。"}`,
    prices: (up, actual, ai) => `實際（綠線）${up ? "漲" : "跌"}到 US$${actual}。AI（橘色虛線）猜 US$${ai}。`,
    total: (score, count) => `你答對了 ${score} / ${count} 題。`,
    next: "下一題 →", replay: "再玩一次",
  },
};
const languageButtons = [...document.querySelectorAll("[data-language]")];
let language = "en";
try {
  if (localStorage.getItem("five-minute-language") === "zh-Hant") language = "zh-Hant";
} catch {
  // The switch still works when storage is unavailable.
}
const copy = () => translations[language];

function drawChart() {
  const sample = examples[index];
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  const history = sample.history;
  const values = selected !== null ? [...history, sample.actual, sample.predicted] : history;
  const low = Math.floor((Math.min(...values) - 0.05) * 5) / 5;
  const high = Math.ceil((Math.max(...values) + 0.05) * 5) / 5;
  const left = 58, right = width - 15, top = 19, bottom = height - 25;
  const x = (step) => left + (right - left) * step / 15;
  const y = (price) => bottom - (bottom - top) * (price - low) / (high - low);

  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#514f73";
  for (let row = 0; row < 4; row++) {
    const price = low + (high - low) * row / 3;
    ctx.strokeStyle = "#e7e2f5";
    ctx.beginPath(); ctx.moveTo(left, y(price)); ctx.lineTo(right, y(price)); ctx.stroke();
    ctx.fillText("$" + price.toFixed(2), 5, y(price) + 4);
  }
  const [past, now, future] = copy().axis;
  ctx.fillText(past, left, height - 4);
  ctx.textAlign = "center";
  ctx.fillText(now, x(10), height - 4);
  ctx.textAlign = "right";
  ctx.fillText(future, right, height - 4);
  ctx.textAlign = "left";

  function line(points, color, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.setLineDash(dash);
    ctx.beginPath();
    points.forEach(([step, price], i) => {
      if (i === 0) ctx.moveTo(x(step), y(price));
      else ctx.lineTo(x(step), y(price));
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  line(history.map((price, i) => [i, price]), "#583bd2");
  if (selected !== null) {
    const last = history.at(-1);
    line([[10, last], [15, sample.predicted]], "#a05418", [4, 4]);
    line([[10, last], [15, sample.actual]], "#17705c");
  }

  const summary = copy().history(history.map((price) => price.toFixed(2)));
  canvas.setAttribute("aria-label", selected === null ? summary :
    summary + copy().chartResult(sample.actual.toFixed(2), sample.predicted.toFixed(3)));
}

function render() {
  const sample = examples[index];
  const words = copy();
  get("round").textContent = words.round(index + 1, examples.length);
  get("chart-context").textContent = words.context(sample.time, sample.history.at(-1).toFixed(2));
  const answered = selected !== null;
  buttons.forEach((button, i) => {
    button.disabled = answered;
    button.setAttribute("aria-pressed", String(answered && selected === (i === 0)));
  });
  get("reveal").hidden = !answered;
  get("feedback").replaceChildren();
  if (answered) {
    const finished = index === examples.length - 1;
    const message = document.createElement("p");
    message.textContent = words.guessResult(selected === sample.up, sample.modelUp === sample.up);
    const prices = document.createElement("p");
    prices.textContent = words.prices(sample.up, sample.actual.toFixed(2), sample.predicted.toFixed(3));
    get("feedback").append(message, prices);
    if (finished) {
      const total = document.createElement("p");
      total.textContent = words.total(score, examples.length);
      get("feedback").append(total);
    }
    get("next").textContent = finished ? words.replay : words.next;
  }
  drawChart();
}

function renderLanguage() {
  document.documentElement.lang = language;
  document.title = copy().title;
  document.querySelector('meta[name="description"]').content = copy().description;
  for (const [id, label] of [["navigation", copy().nav], ["language", copy().language], ["choices", copy().choices]]) {
    get(id).setAttribute("aria-label", label);
  }
  for (const button of languageButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }
  render();
}

function guess(up) {
  if (selected !== null) return;
  selected = up;
  if (up === examples[index].up) score++;
  render();
  get("next").focus();
}

for (const button of languageButtons) {
  button.addEventListener("click", () => {
    language = button.dataset.language;
    try {
      localStorage.setItem("five-minute-language", language);
    } catch {
      // Keep the preference for the current page.
    }
    renderLanguage();
  });
}

buttons[0].addEventListener("click", () => guess(true));
buttons[1].addEventListener("click", () => guess(false));
get("next").addEventListener("click", () => {
  index = (index + 1) % examples.length;
  if (index === 0) score = 0;
  selected = null;
  render();
  buttons[0].focus();
});
new ResizeObserver(drawChart).observe(canvas);
renderLanguage();
