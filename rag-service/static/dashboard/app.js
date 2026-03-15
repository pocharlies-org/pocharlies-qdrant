const API_BASE = window.location.origin;
let allNotes = [];

// Tab Navigation
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'vault' && allNotes.length === 0) loadVaultTree();
    });
});

// Operations Tab
async function loadOps() {
    await Promise.all([loadStats(), loadCollections(), loadTimeline()]);
}

async function loadStats() {
    try {
        const [status, timeline] = await Promise.all([
            fetch(`${API_BASE}/knowledge/status`).then(r => r.json()),
            fetch(`${API_BASE}/dashboard/api/timeline?hours=24&limit=1`).then(r => r.json()),
        ]);
        const qdrant = status.qdrant || {};
        const lastEvent = timeline.events?.[0];
        const cards = [
            { label: 'Vectors (brain)', value: qdrant.points_count || 0 },
            { label: 'Vault Notes', value: status.notes_count || 0 },
            { label: 'Last Activity', value: lastEvent ? new Date(lastEvent.ts).toLocaleTimeString() : 'N/A' },
            { label: 'Health', value: status.health?.status || 'unknown' },
        ];
        document.getElementById('stats-cards').innerHTML = cards.map(c =>
            `<div class="stat-card"><div class="stat-value">${c.value}</div><div class="stat-label">${c.label}</div></div>`
        ).join('');
    } catch (e) {
        document.getElementById('stats-cards').innerHTML = `<div class="error">Failed to load stats: ${e.message}</div>`;
    }
}

async function loadCollections() {
    try {
        const data = await fetch(`${API_BASE}/dashboard/api/collections`).then(r => r.json());
        const maxPts = Math.max(...data.collections.map(c => c.points_count), 1);
        const rows = data.collections.map(c => {
            const pct = Math.round((c.points_count / maxPts) * 100);
            const statusClass = c.status === 'green' ? 'ok' : 'err';
            return `<tr><td>${c.name}</td><td>${c.points_count.toLocaleString()}</td><td><div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div></td><td><span class="status ${statusClass}">${c.status}</span></td></tr>`;
        }).join('');
        document.getElementById('collections-table').innerHTML =
            `<table><thead><tr><th>Collection</th><th>Points</th><th>Size</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
    } catch (e) {
        document.getElementById('collections-table').innerHTML = `<div class="error">Failed: ${e.message}</div>`;
    }
}

const EVENT_ICONS = {
    synthesis_start: '\u{1F9E0}', synthesis_note: '\u{1F4DD}', synthesis_complete: '\u2705',
    rebuild_start: '\u{1F504}', rebuild_complete: '\u2705', reindex: '\u{1F4CA}',
    product_sync: '\u{1F6CD}', competitor_crawl: '\u{1F577}',
};

async function loadTimeline() {
    try {
        const data = await fetch(`${API_BASE}/dashboard/api/timeline?hours=24&limit=100`).then(r => r.json());
        if (!data.events.length) {
            document.getElementById('timeline').innerHTML = '<p class="placeholder">No activity in the last 24 hours</p>';
            return;
        }
        const items = data.events.map(e => {
            const time = new Date(e.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const icon = EVENT_ICONS[e.event] || '\u2022';
            return `<div class="timeline-item"><span class="time">${time}</span> <span class="icon">${icon}</span> <span class="detail">${e.detail || e.event}</span></div>`;
        }).join('');
        document.getElementById('timeline').innerHTML = items;
    } catch (e) {
        document.getElementById('timeline').innerHTML = `<div class="error">Failed: ${e.message}</div>`;
    }
}

async function triggerAction(endpoint, btn) {
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent += ' ...';
    try {
        const r = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
        const data = await r.json();
        showToast(`${r.ok ? 'Success' : 'Error'}: ${JSON.stringify(data).slice(0, 200)}`);
        loadOps();
    } catch (e) {
        showToast(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = orig;
    }
}

// Vault Browser Tab
async function loadVaultTree() {
    try {
        const data = await fetch(`${API_BASE}/knowledge/notes`).then(r => r.json());
        allNotes = data.notes || [];
        renderTree(allNotes);
    } catch (e) {
        document.getElementById('vault-tree').innerHTML = `<div class="error">Failed: ${e.message}</div>`;
    }
}

function renderTree(notes) {
    const tree = {};
    notes.forEach(n => {
        const parts = n.path.split('/');
        let node = tree;
        parts.slice(0, -1).forEach(p => { node[p] = node[p] || {}; node = node[p]; });
        node[parts[parts.length - 1]] = n;
    });
    function buildHtml(obj, depth) {
        let html = '';
        const dirs = Object.keys(obj).filter(k => typeof obj[k] === 'object' && !obj[k].path).sort();
        const files = Object.keys(obj).filter(k => obj[k].path).sort();
        dirs.forEach(d => {
            const count = countFiles(obj[d]);
            html += `<div class="tree-dir" style="padding-left:${depth*16}px"><span class="dir-toggle" onclick="this.parentElement.classList.toggle('collapsed')">&#9660;</span> <span class="dir-name">${d}</span> <span class="badge">${count}</span></div><div class="dir-children">${buildHtml(obj[d], depth+1)}</div>`;
        });
        files.forEach(f => {
            const note = obj[f];
            const tp = note.type ? `<span class="type-pill ${note.type}">${note.type}</span>` : '';
            html += `<div class="tree-file" style="padding-left:${depth*16+16}px" data-path="${note.path}" onclick="loadNote('${note.path}')">${tp} ${note.title || f}</div>`;
        });
        return html;
    }
    function countFiles(obj) {
        let n = 0; Object.values(obj).forEach(v => { n += v.path ? 1 : countFiles(v); }); return n;
    }
    document.getElementById('vault-tree').innerHTML = buildHtml(tree, 0);
}

function filterNotes(query) {
    const q = query.toLowerCase();
    const filtered = q ? allNotes.filter(n => (n.title||'').toLowerCase().includes(q) || (n.path||'').toLowerCase().includes(q)) : allNotes;
    renderTree(filtered);
}

async function loadNote(path) {
    try {
        const data = await fetch(`${API_BASE}/knowledge/note/${path}`).then(r => r.json());
        const content = data.content || '';
        const fm = data.frontmatter || {};
        const badges = Object.entries(fm)
            .filter(([k]) => ['type','confidence','platform','subtype'].includes(k))
            .map(([k,v]) => `<span class="meta-pill ${k}-${v}">${k}: ${v}</span>`).join(' ');
        document.getElementById('vault-meta').innerHTML = badges;
        let md = content;
        md = md.replace(/\[\[([^\]]+)\]\]/g, (_, name) => {
            const target = allNotes.find(n => n.path.includes(name) || (n.title||'').includes(name));
            return target ? `<a href="#" onclick="loadNote('${target.path}');return false">${name}</a>` : `<span class="broken-link">${name}</span>`;
        });
        md = md.replace(/\[TOOL:show_product:([^\]]+)\]/g, (_, handle) =>
            `<a href="https://www.skirmshop.es/products/${handle}" target="_blank" class="product-badge">${handle}</a>`);
        document.getElementById('vault-note').innerHTML = marked.parse(md);
        document.querySelectorAll('#vault-note pre code').forEach(el => hljs.highlightElement(el));
        document.querySelectorAll('.tree-file').forEach(el => el.classList.remove('active'));
        document.querySelector(`.tree-file[data-path="${path}"]`)?.classList.add('active');
    } catch (e) {
        document.getElementById('vault-note').innerHTML = `<div class="error">Failed to load note: ${e.message}</div>`;
    }
}

function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 4000);
}

loadOps();
