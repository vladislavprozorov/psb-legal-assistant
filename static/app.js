"use strict";

const STATUS = {
  answer: ["Ответ по базе", "ok"],
  partial: ["Частичный ответ", "warn"],
  no_data: ["Нет данных в базе", "warn"],
  clarify: ["Нужно уточнение", "info"],
  refuse: ["Отказ", "bad"],
  out_of_scope: ["Вне компетенции", "muted"],
  fallback: ["Модель недоступна — найденные положения", "warn"],
};
const EXAMPLES = [
  "Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её вносить?",
  "Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?",
  "Какой максимальный срок испытания можно установить главному бухгалтеру?",
  "Подскажите гарантию по договору с «Альфой».",
  "Какой размер пособия по беременности и родам?",
];

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

let history = [];       // [{role, content}]
let answers = [];       // ответы ассистента с источниками
let busy = false;

function setTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("#tab-sources").classList.toggle("hidden", name !== "sources");
  $("#tab-doc").classList.toggle("hidden", name !== "doc");
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setTab(t.dataset.tab)));

function scrollDown() { const m = $("#messages"); m.scrollTop = m.scrollHeight; }

function addUser(text) {
  $("#empty")?.remove();
  $("#messages").append(el("div", "msg user", text));
  scrollDown();
}

// Делает «п. 3.1», «ст. 70» в тексте ответа кликабельными: открывают документ на нужном фрагменте
const CITE_RE = /(п\.\s*\d+(?:\.\d+)*|ст\.\s*\d+(?:\.\d+)?)/g;
function findCite(res, token) {
  const pool = [...(res.sources || []), ...(res.retrieved || [])];
  const num = token.replace(/^(п|ст)\.\s*/, "");
  if (token.startsWith("п")) return pool.find((s) => s.label === "п. " + num);
  return pool.find((s) => s.kind === "law" && new RegExp("ст\\. " + num.replace(".", "\\.") + "(\\D|$)").test(s.doc_title));
}
function renderAnswer(res) {
  const frag = document.createDocumentFragment();
  let last = 0;
  res.answer.replace(CITE_RE, (m, _g, offset) => {
    const hit = findCite(res, m);
    if (!hit) return m;
    frag.append(document.createTextNode(res.answer.slice(last, offset)));
    const b = el("button", "cite", m);
    b.type = "button";
    b.title = `${hit.doc_title}, ${hit.label}`;
    b.addEventListener("click", () => openDoc(hit.doc_id, hit.id));
    frag.append(b);
    last = offset + m.length;
    return m;
  });
  frag.append(document.createTextNode(res.answer.slice(last)));
  return frag;
}

function copyText(res, question) {
  const src = (res.sources || []).map((s) => `— ${s.doc_title}, ${s.label}: ${s.text.replace(/\s+/g, " ")}`);
  return `Вопрос: ${question}\n\nОтвет: ${res.answer}` + (src.length ? `\n\nИсточники:\n${src.join("\n")}` : "");
}

async function sendFeedback(res, question, vote, btns) {
  btns.forEach((b) => { b.disabled = true; b.classList.toggle("chosen", b.dataset.vote === vote); });
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vote, question, answer: res.answer, status: res.status,
        sources: (res.sources || []).map((s) => s.id) }),
    });
  } catch (e) { /* оценка необязательна */ }
}

function addAssistant(res, idx, question) {
  const box = el("div", "msg assistant");
  box.append(renderAnswer(res));
  const meta = el("div", "meta");
  const [label, cls] = STATUS[res.status] || STATUS.answer;
  meta.append(el("span", "pill " + cls, label));
  if (res.verified) meta.append(el("span", "pill ok", "✓ самопроверка"));
  const n = (res.sources || []).length;
  const link = el("button", "link", n ? `Источники: ${n}` : "Найденные фрагменты");
  link.type = "button";
  link.addEventListener("click", () => showSources(idx));
  meta.append(link);
  if (res.timings) meta.append(el("span", "pill muted", `${res.timings.total} с`));
  const tools = el("span", "tools");
  const copy = el("button", "icon-btn", "Скопировать");
  copy.type = "button";
  copy.title = "Скопировать ответ с источниками";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(copyText(res, question)); copy.textContent = "Скопировано"; }
    catch (e) { copy.textContent = "Не удалось"; }
    setTimeout(() => { copy.textContent = "Скопировать"; }, 1500);
  });
  const up = el("button", "icon-btn", "👍");
  const down = el("button", "icon-btn", "👎");
  up.dataset.vote = "up"; down.dataset.vote = "down";
  up.title = "Ответ верный"; down.title = "Ответ неверный";
  [up, down].forEach((b) => { b.type = "button"; b.addEventListener("click", () => sendFeedback(res, question, b.dataset.vote, [up, down])); });
  tools.append(copy, up, down);
  meta.append(tools);
  box.append(meta);
  box.dataset.idx = idx;
  $("#messages").append(box);
  scrollDown();
}

function srcCard(s, cited) {
  const b = el("button", "src" + (cited ? " cited" : ""));
  b.type = "button";
  const head = el("div");
  head.append(el("span", "doc", s.doc_title), el("span", "lbl", s.label));
  b.append(head, el("div", "txt", s.text));
  b.addEventListener("click", () => openDoc(s.doc_id, s.id));
  return b;
}

function showSources(idx) {
  const res = answers[idx];
  document.querySelectorAll(".msg.assistant").forEach((m) => m.classList.toggle("active", Number(m.dataset.idx) === idx));
  const body = $("#tab-sources");
  body.replaceChildren();
  if (!res) return;
  const cited = res.sources || [];
  const citedIds = new Set(cited.map((s) => s.id));
  if (cited.length) {
    body.append(el("div", "section-title", "Подтверждающие фрагменты"));
    cited.forEach((s) => body.append(srcCard(s, true)));
  } else {
    body.append(el("p", "hint", "Ответ не ссылается на конкретные фрагменты (например, отказ или уточняющий вопрос)."));
  }
  const other = (res.retrieved || []).filter((s) => !citedIds.has(s.id));
  if (other.length) {
    const det = el("details", "more");
    det.append(el("summary", null, `Другие найденные фрагменты законов (${other.length})`));
    other.forEach((s) => det.append(srcCard(s, false)));
    body.append(det);
  }
  if (res.analysis && res.analysis.length) {
    const box = el("div", "analysis");
    box.append(el("div", "section-title", "Автоматический анализ вопроса"));
    const ul = el("ul");
    res.analysis.forEach((h) => ul.append(el("li", null, h)));
    box.append(ul);
    body.prepend(box);
  }
  if (res.verify_note) body.append(el("p", "hint", "Самопроверка: " + res.verify_note));
  setTab("sources");
}

async function openDoc(docId, chunkId) {
  const body = $("#tab-doc");
  body.replaceChildren(el("p", "hint", "Загрузка документа…"));
  setTab("doc");
  try {
    const r = await fetch(`/api/documents/${encodeURIComponent(docId)}`);
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    body.replaceChildren();
    const head = el("div", "doc-head");
    head.append(el("h3", null, d.title));
    const pills = el("div", "pills");
    pills.append(el("span", "pill info", d.kind === "law" ? "Закон" : "Документ компании"));
    if (d.status) pills.append(el("span", "pill " + (d.status.includes("утратил") ? "bad" : "ok"), d.status));
    if (d.date) pills.append(el("span", "pill muted", d.date));
    head.append(pills);
    body.append(head);
    let target = null;
    let section = "";
    d.chunks.forEach((c) => {
      let text = c.text;
      if (d.kind !== "law" && text.includes("\n")) {
        // у пунктов договоров первая строка — заголовок раздела; показываем его один раз
        const [head, ...rest] = text.split("\n");
        if (/^\d+\.\s/.test(head)) {
          if (head !== section) body.append(el("p", "chunk section", head));
          section = head;
          text = rest.join("\n");
        }
      }
      const p = el("p", "chunk" + (c.id === chunkId ? " hl" : ""));
      if (d.kind === "law" && c.label) p.append(el("span", "cl", c.label));
      p.append(document.createTextNode(text));
      if (c.id === chunkId) target = p;
      body.append(p);
    });
    if (target) target.scrollIntoView({ block: "center" });
  } catch (e) {
    body.replaceChildren(el("p", "hint", "Не удалось открыть документ: " + e.message));
  }
}

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  $("#send").disabled = true;
  addUser(text);
  history.push({ role: "user", content: text });
  const typing = el("div", "msg assistant");
  const dots = el("span", "typing");
  dots.append(el("span"), el("span"), el("span"));
  typing.append(dots, document.createTextNode("  ищу в документах и проверяю ответ…"));
  $("#messages").append(typing);
  scrollDown();
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    const data = await r.json().catch(() => ({}));
    typing.remove();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    history.push({ role: "assistant", content: data.answer });
    answers.push(data);
    const idx = answers.length - 1;
    addAssistant(data, idx, text);
    showSources(idx);
  } catch (e) {
    typing.remove();
    history.pop();
    $("#messages").append(el("div", "msg assistant error", "Ошибка: " + e.message));
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#input").focus();
  }
}

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#input").value;
  $("#input").value = "";
  autosize();
  send(v);
});
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#form").requestSubmit(); }
});
function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 180) + "px"; }
$("#input").addEventListener("input", autosize);

$("#btn-new").addEventListener("click", () => {
  history = []; answers = [];
  $("#messages").replaceChildren();
  const empty = el("div", "empty");
  empty.id = "empty";
  empty.append(el("h2", null, "Новый диалог"), el("p", null, "Задайте вопрос по документам базы знаний."));
  const ex = el("div", "examples");
  renderExamples(ex);
  empty.append(ex);
  $("#messages").append(empty);
  $("#tab-sources").replaceChildren(el("p", "hint", "Здесь появятся фрагменты, на которые опирается ответ."));
});

function renderExamples(box) {
  EXAMPLES.forEach((q) => {
    const b = el("button", null, q);
    b.type = "button";
    b.addEventListener("click", () => send(q));
    box.append(b);
  });
}
renderExamples($("#examples"));

async function loadKb() {
  const list = $("#kb-list");
  list.replaceChildren(el("p", "hint", "Загрузка…"));
  const r = await fetch("/api/documents");
  const docs = await r.json();
  list.replaceChildren();
  [["company", "Документы компании"], ["law", "Законы (действующие редакции)"]].forEach(([kind, title]) => {
    list.append(el("div", "kb-group", title));
    docs.filter((d) => d.kind === kind).forEach((d) => {
      const b = el("button", "kb-item");
      b.type = "button";
      b.append(el("span", null, d.title), el("span", "pill " + (d.status.includes("утратил") ? "bad" : "muted"), d.date || d.status));
      b.addEventListener("click", () => { $("#kb-dialog").close(); openDoc(d.id, null); });
      list.append(b);
    });
  });
}
$("#btn-kb").addEventListener("click", () => { $("#kb-dialog").showModal(); loadKb(); });
$("#kb-close").addEventListener("click", () => $("#kb-dialog").close());

fetch("/api/health").then((r) => r.json()).then((h) => {
  const p = $("#health");
  p.textContent = h.llm ? `${h.documents} документов · модель подключена` : `${h.documents} документов · LLM не настроена`;
  p.className = "pill " + (h.llm ? "ok" : "warn");
}).catch(() => { $("#health").textContent = "сервер недоступен"; $("#health").className = "pill bad"; });
