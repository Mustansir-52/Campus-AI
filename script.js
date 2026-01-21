

const BACKEND_URL = "http://127.0.0.1:4000/chat"; 
const STORAGE_KEY = "campus_ai_college_msgs_v1";
const SESSION_KEY = "campus_ai_college_session_v1";

// DOM Elements
const messagesEl = document.getElementById("chat-area");
const inputEl = document.getElementById("msg");
const sendBtn = document.getElementById("send");
const voiceBtn = document.getElementById("voice");
const suggestionsEl = document.querySelector(".suggestions");
const clearBtn = document.getElementById("clear-chat");

// Create / Restore Session
let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = "sess_" + Math.random().toString(36).slice(2, 10);
  localStorage.setItem(SESSION_KEY, sessionId);
}

let messages = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");

// Utility
function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

// Build Message Row
function makeRow(m) {
  const row = document.createElement("div");
  row.className = "row " + (m.sender === "user" ? "user" : "bot");

  const bubble = document.createElement("div");
  bubble.className = "bubble " + (m.sender === "user" ? "user" : "bot");
  bubble.innerHTML = esc(m.text).replace(/\n/g, "<br>");

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = new Date(m.ts || Date.now()).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit"
  });
  bubble.appendChild(meta);

  row.appendChild(bubble);
  return row;
}

function renderAll() {
  messagesEl.innerHTML = "";
  messages.forEach((m) => messagesEl.appendChild(makeRow(m)));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Typing Indicator
let typingNode = null;
function showTyping(on = true) {
  if (on) {
    if (!typingNode) {
      typingNode = document.createElement("div");
      typingNode.className = "row bot typing";
      typingNode.innerHTML = `
        <div class="bubble bot">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      `;
      messagesEl.appendChild(typingNode);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } else {
    if (typingNode) typingNode.remove();
    typingNode = null;
  }
}

// Send + Backend POST
async function sendMessage(text) {
  if (!text.trim()) return;

  const userMsg = {
    sender: "user",
    text: text.trim(),
    ts: Date.now()
  };

  messages.push(userMsg);
  persist();
  messagesEl.appendChild(makeRow(userMsg));
  messagesEl.scrollTop = messagesEl.scrollHeight;
  inputEl.value = "";

  // Start typing indicator
  showTyping(true);

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, sessionId })
    });

    if (!res.ok) throw new Error("Server error");

    const data = await res.json();
    const reply = data.reply || data.answer || "⚠️ No reply from server.";

    const botMsg = { sender: "bot", text: reply, ts: Date.now() };
    messages.push(botMsg);
    persist();

    showTyping(false);
    messagesEl.appendChild(makeRow(botMsg));
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    showTyping(false);
    const errMsg = {
      sender: "bot",
      text: "❌ Cannot reach backend. Please check your server or CORS settings.",
      ts: Date.now()
    };

    messages.push(errMsg);
    persist();
    messagesEl.appendChild(makeRow(errMsg));
    console.error("Backend error:", err);
  }
}

// UI Handlers
sendBtn.addEventListener("click", () => {
  const v = inputEl.value;
  if (v) sendMessage(v);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});
clearBtn.addEventListener("click", () => {
  messages = [];
  persist();
  messagesEl.innerHTML = "";

  sessionId = "sess_" + Math.random().toString(36).slice(2, 10);
  localStorage.setItem(SESSION_KEY, sessionId);
});


// Suggestions
if (suggestionsEl) {
  suggestionsEl.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") {
      inputEl.value = e.target.textContent;
      inputEl.focus();
    }
  });
}

// Speech Recognition (if supported)
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (window.SpeechRecognition) {
  const rec = new SpeechRecognition();
  rec.lang = "en-IN";
  rec.interimResults = false;
  rec.maxAlternatives = 1;

  voiceBtn.addEventListener("click", () => {
    try {
      rec.start();
      voiceBtn.classList.add("listening");
    } catch {}
  });

  rec.addEventListener("result", (e) => {
    const text = e.results[0][0].transcript;
    inputEl.value = text;
    sendMessage(text);
  });

  rec.addEventListener("end", () => voiceBtn.classList.remove("listening"));
} else {
  voiceBtn.style.display = "none";
}

// Developer Helper
window.clearCollegeChat = function () {
  messages = [];
  persist();
  renderAll();
};

// First-Time Greeting
if (messages.length === 0) {
  const greet = {
    sender: "bot",
    text: "Welcome to Campus AI — Ask about schedules, events, fees, academics, or general questions!",
    ts: Date.now()
  };
  messages.push(greet);
  persist();
}

// Render on load
renderAll();
