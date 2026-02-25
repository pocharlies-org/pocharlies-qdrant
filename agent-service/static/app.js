/* ============================================================
   Pocharlies Agent — Chat UI (WebSocket client)
   ============================================================
   SECURITY: This file never uses innerHTML, outerHTML,
   insertAdjacentHTML, or similar unsafe DOM methods.
   All content is set via textContent / createElement +
   appendChild only.
   ============================================================ */

"use strict";

// -----------------------------------------------------------
// State
// -----------------------------------------------------------

const state = {
    ws: null,
    threadId: null,
    threads: [],            // [{id, name, lastMessage}]
    connected: false,
    currentAssistantEl: null, // streaming target element
};

// -----------------------------------------------------------
// DOM references
// -----------------------------------------------------------

const $messages     = document.getElementById("messages");
const $input        = document.getElementById("message-input");
const $sendBtn      = document.getElementById("send-btn");
const $newThreadBtn = document.getElementById("new-thread-btn");
const $threadList   = document.getElementById("thread-list");
const $statusDot    = document.getElementById("status-indicator");
const $statusText   = document.getElementById("status-text");

// -----------------------------------------------------------
// Utilities
// -----------------------------------------------------------

function generateId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Fallback for older browsers
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        var r = (Math.random() * 16) | 0;
        var v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

function timestamp() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
    requestAnimationFrame(function () {
        $messages.scrollTop = $messages.scrollHeight;
    });
}

// -----------------------------------------------------------
// Status
// -----------------------------------------------------------

function updateStatus(connected) {
    state.connected = connected;
    if (connected) {
        $statusDot.classList.add("connected");
        $statusDot.classList.remove("disconnected");
        $statusText.textContent = "Connected";
    } else {
        $statusDot.classList.remove("connected");
        $statusDot.classList.add("disconnected");
        $statusText.textContent = "Disconnected";
    }
}

// -----------------------------------------------------------
// WebSocket
// -----------------------------------------------------------

function connect() {
    var protocol = location.protocol === "https:" ? "wss:" : "ws:";
    var url = protocol + "//" + location.host + "/api/chat/ws";

    var ws = new WebSocket(url);
    state.ws = ws;

    ws.addEventListener("open", function () {
        updateStatus(true);
    });

    ws.addEventListener("close", function () {
        updateStatus(false);
        state.ws = null;
        // Auto-reconnect after 3 seconds
        setTimeout(connect, 3000);
    });

    ws.addEventListener("error", function (err) {
        console.error("[ws] error:", err);
    });

    ws.addEventListener("message", function (event) {
        var data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            console.error("[ws] bad JSON:", event.data);
            return;
        }
        handleServerMessage(data);
    });
}

// -----------------------------------------------------------
// Server message handler
// -----------------------------------------------------------

function handleServerMessage(data) {
    switch (data.type) {
        case "token":
            if (!state.currentAssistantEl) {
                startStreaming();
            }
            appendToken(data.content);
            break;

        case "tool_call":
            addToolCall(data.name, data.args);
            break;

        case "tool_result":
            updateToolResult(data.name, data.content);
            break;

        case "done":
            finalizeStreaming();
            if (data.thread_id) {
                state.threadId = data.thread_id;
                updateCurrentThread();
            }
            break;

        case "error":
            addMessage("error", data.message || "Unknown error");
            finalizeStreaming();
            break;

        case "interrupt":
            addInterrupt(data);
            break;

        default:
            console.warn("[ws] unknown message type:", data.type);
    }
}

// -----------------------------------------------------------
// Messages
// -----------------------------------------------------------

function addMessage(role, content) {
    var el = document.createElement("div");
    el.classList.add("message", role);

    var textEl = document.createElement("span");
    textEl.classList.add("message-text");
    textEl.textContent = content;
    el.appendChild(textEl);

    var ts = document.createElement("span");
    ts.classList.add("message-timestamp");
    ts.textContent = timestamp();
    el.appendChild(ts);

    $messages.appendChild(el);
    scrollToBottom();
    return el;
}

// -----------------------------------------------------------
// Streaming
// -----------------------------------------------------------

function startStreaming() {
    var el = document.createElement("div");
    el.classList.add("message", "assistant");

    var textEl = document.createElement("span");
    textEl.classList.add("message-text");
    el.appendChild(textEl);

    var ts = document.createElement("span");
    ts.classList.add("message-timestamp");
    ts.textContent = timestamp();
    el.appendChild(ts);

    $messages.appendChild(el);
    state.currentAssistantEl = textEl;
    scrollToBottom();
}

function appendToken(token) {
    if (state.currentAssistantEl) {
        state.currentAssistantEl.textContent += token;
        scrollToBottom();
    }
}

function finalizeStreaming() {
    if (state.currentAssistantEl) {
        // Update the final content in the thread record
        var content = state.currentAssistantEl.textContent;
        if (content && state.threadId) {
            var thread = state.threads.find(function (t) { return t.id === state.threadId; });
            if (thread) {
                thread.lastMessage = content.slice(0, 80);
                renderThreads();
            }
        }
    }
    state.currentAssistantEl = null;
}

// -----------------------------------------------------------
// Tool calls
// -----------------------------------------------------------

function addToolCall(name, args) {
    var el = document.createElement("div");
    el.classList.add("tool-call");
    el.setAttribute("data-tool-name", name);

    // Header
    var header = document.createElement("div");
    header.classList.add("tool-call-header");

    var icon = document.createElement("span");
    icon.classList.add("tool-call-icon");
    icon.textContent = "\u2699"; // gear symbol
    header.appendChild(icon);

    var nameEl = document.createElement("span");
    nameEl.classList.add("tool-call-name");
    nameEl.textContent = name;
    header.appendChild(nameEl);

    var toggle = document.createElement("span");
    toggle.classList.add("tool-call-toggle");
    toggle.textContent = "\u25B6"; // right-pointing triangle
    header.appendChild(toggle);

    el.appendChild(header);

    // Body
    var body = document.createElement("div");
    body.classList.add("tool-call-body");

    var argsLabel = document.createElement("div");
    argsLabel.classList.add("tool-call-label");
    argsLabel.textContent = "Arguments";
    body.appendChild(argsLabel);

    var argsContent = document.createElement("pre");
    argsContent.classList.add("tool-call-content");
    argsContent.textContent = typeof args === "string" ? args : JSON.stringify(args, null, 2);
    body.appendChild(argsContent);

    el.appendChild(body);

    // Toggle expand/collapse
    header.addEventListener("click", function () {
        el.classList.toggle("expanded");
    });

    $messages.appendChild(el);
    scrollToBottom();
}

function updateToolResult(name, content) {
    // Find the last tool-call element with matching name
    var toolCalls = $messages.querySelectorAll('.tool-call[data-tool-name="' + CSS.escape(name) + '"]');
    var target = toolCalls.length > 0 ? toolCalls[toolCalls.length - 1] : null;

    if (!target) {
        // No matching tool call; create a standalone result
        addToolCall(name, "");
        toolCalls = $messages.querySelectorAll('.tool-call[data-tool-name="' + CSS.escape(name) + '"]');
        target = toolCalls[toolCalls.length - 1];
    }

    var body = target.querySelector(".tool-call-body");
    if (!body) return;

    var resultDiv = document.createElement("div");
    resultDiv.classList.add("tool-call-result");

    var label = document.createElement("div");
    label.classList.add("tool-call-label");
    label.textContent = "Result";
    resultDiv.appendChild(label);

    var pre = document.createElement("pre");
    pre.classList.add("tool-call-content");
    pre.textContent = content;
    resultDiv.appendChild(pre);

    body.appendChild(resultDiv);

    // Auto-expand to show result
    target.classList.add("expanded");
    scrollToBottom();
}

// -----------------------------------------------------------
// Interrupt (approve / reject)
// -----------------------------------------------------------

function addInterrupt(data) {
    var el = document.createElement("div");
    el.classList.add("interrupt");

    var text = document.createElement("div");
    text.classList.add("interrupt-text");
    text.textContent = data.message || "The agent is requesting approval to proceed.";
    el.appendChild(text);

    var buttons = document.createElement("div");
    buttons.classList.add("interrupt-buttons");

    var approveBtn = document.createElement("button");
    approveBtn.classList.add("btn-approve");
    approveBtn.textContent = "Approve";
    approveBtn.addEventListener("click", function () {
        sendInterruptResponse("approve", data);
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
        text.textContent += " [Approved]";
    });
    buttons.appendChild(approveBtn);

    var rejectBtn = document.createElement("button");
    rejectBtn.classList.add("btn-reject");
    rejectBtn.textContent = "Reject";
    rejectBtn.addEventListener("click", function () {
        sendInterruptResponse("reject", data);
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
        text.textContent += " [Rejected]";
    });
    buttons.appendChild(rejectBtn);

    el.appendChild(buttons);
    $messages.appendChild(el);
    scrollToBottom();
}

function sendInterruptResponse(action, data) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    state.ws.send(JSON.stringify({
        type: "interrupt_response",
        action: action,
        thread_id: state.threadId,
        interrupt_id: data.interrupt_id || null,
    }));
}

// -----------------------------------------------------------
// Send user message
// -----------------------------------------------------------

function send() {
    var content = $input.value.trim();
    if (!content) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    // Ensure we have a thread
    if (!state.threadId) {
        newThread(content);
    }

    // Show user message
    addMessage("user", content);

    // Update thread preview
    var thread = state.threads.find(function (t) { return t.id === state.threadId; });
    if (thread) {
        thread.lastMessage = content.slice(0, 80);
        if (thread.name === "New thread") {
            thread.name = content.slice(0, 40);
        }
        renderThreads();
    }

    // Send over WebSocket
    state.ws.send(JSON.stringify({
        type: "message",
        content: content,
        thread_id: state.threadId,
    }));

    // Clear input
    $input.value = "";
    autoResize();
}

// -----------------------------------------------------------
// Thread management
// -----------------------------------------------------------

function newThread(firstMessage) {
    var id = generateId();
    var name = firstMessage ? firstMessage.slice(0, 40) : "New thread";
    state.threadId = id;
    state.currentAssistantEl = null;
    state.threads.unshift({ id: id, name: name, lastMessage: "" });
    $messages.textContent = ""; // clear messages
    renderThreads();
    $input.focus();
}

function switchThread(id) {
    if (state.threadId === id) return;
    state.threadId = id;
    state.currentAssistantEl = null;
    $messages.textContent = ""; // clear messages (no history persistence yet)
    renderThreads();
    $input.focus();
}

function updateCurrentThread() {
    // Ensure the current thread_id exists in our list
    var exists = state.threads.some(function (t) { return t.id === state.threadId; });
    if (!exists && state.threadId) {
        state.threads.unshift({
            id: state.threadId,
            name: "Thread",
            lastMessage: "",
        });
        renderThreads();
    }
}

function renderThreads() {
    // Clear existing items
    while ($threadList.firstChild) {
        $threadList.removeChild($threadList.firstChild);
    }

    state.threads.forEach(function (thread) {
        var btn = document.createElement("button");
        btn.classList.add("thread-item");
        if (thread.id === state.threadId) {
            btn.classList.add("active");
        }

        var nameSpan = document.createElement("span");
        nameSpan.classList.add("thread-name");
        nameSpan.textContent = thread.name;
        btn.appendChild(nameSpan);

        var previewSpan = document.createElement("span");
        previewSpan.classList.add("thread-preview");
        previewSpan.textContent = thread.lastMessage || "";
        btn.appendChild(previewSpan);

        btn.addEventListener("click", function () {
            switchThread(thread.id);
        });

        $threadList.appendChild(btn);
    });
}

// -----------------------------------------------------------
// Auto-resize textarea
// -----------------------------------------------------------

function autoResize() {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 200) + "px";
}

// -----------------------------------------------------------
// Event listeners
// -----------------------------------------------------------

// Send on Enter (without Shift)
$input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

// Auto-resize on input
$input.addEventListener("input", autoResize);

// Send button click
$sendBtn.addEventListener("click", send);

// New thread button
$newThreadBtn.addEventListener("click", function () {
    newThread();
});

// -----------------------------------------------------------
// Initialise
// -----------------------------------------------------------

(function init() {
    newThread();
    connect();
})();
