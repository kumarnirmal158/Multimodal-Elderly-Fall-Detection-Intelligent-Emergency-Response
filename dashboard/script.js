const API_BASE_URL = window.location.protocol.startsWith("http")
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://127.0.0.1:8000";

let incidentsData = [];
let alertsData = [];
let tasksData = [];
let isLoading = false;

const connectionStatus = document.getElementById("connection-status");
const refreshButton = document.getElementById("refresh-btn");
const escalateDemoBtn = document.getElementById("escalate-demo-btn");
const resetDbBtn = document.getElementById("reset-db-btn");

const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadBtn = document.getElementById("upload-btn");

const totalIncidentsElement = document.getElementById("total-incidents");
const possibleFallsElement = document.getElementById("possible-falls");
const openAlertsElement = document.getElementById("open-alerts");
const pendingTasksElement = document.getElementById("pending-tasks");

const incidentsTableBody = document.getElementById("incidents-table-body");
const alertsContainer = document.getElementById("alerts-container");
const tasksContainer = document.getElementById("tasks-container");
const evaluationContainer = document.getElementById("evaluation-container");
const incidentFilter = document.getElementById("incident-filter");
const incidentDetailsSection = document.getElementById("incident-details-section");
const incidentDetails = document.getElementById("incident-details");

function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function apiRequest(endpoint, options = {}) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    if (!response.ok) {
        let errorMessage = `Request failed (${response.status})`;
        try {
            const errorData = await response.json();
            if (typeof errorData.detail === "string") {
                errorMessage = errorData.detail;
            }
        } catch {}
        throw new Error(errorMessage);
    }
    if (response.status === 204) return null;
    return await response.json();
}

function updateConnectionStatus(status) {
    if (!connectionStatus) return;
    if (status === "connected") {
        connectionStatus.textContent = "Backend Connected";
        connectionStatus.className = "status-indicator connected";
    } else {
        connectionStatus.textContent = "Backend Disconnected";
        connectionStatus.className = "status-indicator disconnected";
    }
}

function getStatusClass(status) {
    const val = String(status || "").toUpperCase();
    if (["HIGH", "NEW", "OPEN", "TRUE_FALL"].includes(val)) return "status-high";
    if (["REVIEW", "ACKNOWLEDGED", "PENDING", "UNCERTAIN"].includes(val)) return "status-review";
    if (["RESOLVED", "VERIFIED", "COMPLETED", "FALSE_ALERT"].includes(val)) return "status-resolved";
    return "status-low";
}

function formatScore(value) {
    if (value === null || value === undefined || value === "") return "N/A";
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(2) : escapeHTML(value);
}

function formatEventTime(incident) {
    const s = Number(incident.video_start_time);
    const e = Number(incident.video_end_time);
    if (!Number.isFinite(s) || !Number.isFinite(e)) return "N/A";
    return `${s.toFixed(2)}s - ${e.toFixed(2)}s`;
}

async function loadDashboard() {
    if (isLoading) return;
    isLoading = true;
    try {
        const [incRes, alertRes, taskRes, evalRes] = await Promise.all([
            apiRequest("/api/incidents"),
            apiRequest("/api/alerts/open"),
            apiRequest("/api/tasks"),
            apiRequest("/api/evaluation"),
        ]);

        incidentsData = incRes.incidents || [];
        alertsData = alertRes.alerts || [];
        tasksData = taskRes.tasks || [];

        updateConnectionStatus("connected");
        renderSummary();
        displayIncidents();
        displayAlerts();
        displayTasks();
        displayEvaluation(evalRes);
    } catch (err) {
        updateConnectionStatus("disconnected");
        console.error(err);
    } finally {
        isLoading = false;
    }
}

function renderSummary() {
    totalIncidentsElement.textContent = incidentsData.length;
    possibleFallsElement.textContent = incidentsData.filter(
        i => i.classification === "POSSIBLE_MULTIMODAL_FALL_CANDIDATE"
    ).length;
    openAlertsElement.textContent = alertsData.length;
    pendingTasksElement.textContent = tasksData.filter(t => t.status === "PENDING").length;
}

function displayIncidents() {
    const filter = incidentFilter ? incidentFilter.value : "ALL";
    const rows = filter === "ALL"
        ? incidentsData
        : incidentsData.filter(i => i.classification === filter);

    if (rows.length === 0) {
        incidentsTableBody.innerHTML = `<tr><td colspan="10">No incidents found.</td></tr>`;
        return;
    }

    incidentsTableBody.innerHTML = rows.map(i => `
        <tr>
            <td>${escapeHTML(i.id)}</td>
            <td>${escapeHTML(i.resident_name)} (${escapeHTML(i.room_location)})</td>
            <td>${escapeHTML(formatEventTime(i))}</td>
            <td>${formatScore(i.video_score)}</td>
            <td>${formatScore(i.audio_score)}</td>
            <td>${formatScore(i.combined_score)}</td>
            <td>${escapeHTML(i.classification)}</td>
            <td><span class="status-label ${getStatusClass(i.verification_status)}">${escapeHTML(i.verification_status)}</span></td>
            <td><span class="status-label ${getStatusClass(i.incident_status)}">${escapeHTML(i.incident_status)}</span></td>
            <td><button class="view-btn" data-incident-id="${escapeHTML(i.id)}">Inspect &amp; Verify</button></td>
        </tr>
    `).join("");
}

async function viewIncident(incidentId) {
    incidentDetailsSection.hidden = false;
    incidentDetails.innerHTML = `<p>Loading...</p>`;

    try {
        const inc = await apiRequest(`/api/incidents/${encodeURIComponent(incidentId)}`);
        const timelineHtml = (inc.temporal_timeline || []).map(t => `
            <li><strong>[${escapeHTML(t.phase)}] ${escapeHTML(t.timestamp)}s:</strong> ${escapeHTML(t.observation)}</li>
        `).join("");

        incidentDetails.innerHTML = `
            <p><strong>Resident:</strong> ${escapeHTML(inc.resident_name)} | <strong>Room:</strong> ${escapeHTML(inc.room_location)} | <strong>Source:</strong> ${escapeHTML(inc.source_video || "N/A")}</p>
            <p><strong>Scores:</strong> Video ${formatScore(inc.video_score)} | Audio ${formatScore(inc.audio_score)} | Combined ${formatScore(inc.combined_score)}</p>
            <p><strong>Post-Fall Inactivity:</strong> ${escapeHTML(inc.post_fall_inactivity_seconds ?? 0)}s (Prolonged: ${inc.prolonged_inactivity ? "Yes" : "No"})</p>
            <p><strong>Speech Distress:</strong> ${inc.distress_detected ? "YES" : "No"} | <strong>Transcript:</strong> ${escapeHTML(inc.distress_transcript || "None")}</p>
            <p><strong>VLM Activity (${escapeHTML(inc.vlm_activity_type || "N/A")}):</strong> ${escapeHTML(inc.vlm_summary || "N/A")}</p>
            <h4>Temporal Timeline (Before / During / After)</h4>
            <ul class="timeline-list">${timelineHtml || "<li>No timeline events recorded.</li>"}</ul>

            <div class="verify-box">
                <h4>Caregiver Verification (Required Before Alert Closure)</h4>
                <p>Current Verification: <span class="status-label ${getStatusClass(inc.verification_status)}">${escapeHTML(inc.verification_status)}</span></p>
                <label><input type="checkbox" id="med-assist-check"> Additional Medical Assistance Required (Auto-creates Nurse Assessment Task)</label>
                <div class="verify-actions">
                    <button class="danger-btn" onclick="submitVerification(${inc.id}, 'TRUE_FALL')">Confirm True Fall</button>
                    <button class="resolve-btn" onclick="submitVerification(${inc.id}, 'FALSE_ALERT')">Mark False Alert</button>
                    <button class="acknowledge-btn" onclick="submitVerification(${inc.id}, 'UNCERTAIN')">Uncertain / Needs Review</button>
                </div>
            </div>
        `;
        incidentDetailsSection.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        incidentDetails.innerHTML = `<p class="error-message">${escapeHTML(err.message)}</p>`;
    }
}

window.submitVerification = async function(incidentId, status) {
    const medCheck = document.getElementById("med-assist-check")?.checked || false;
    try {
        await apiRequest(`/api/incidents/${incidentId}/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                verification_status: status,
                verified_by: "Caregiver on Duty",
                medical_assistance_required: medCheck || status === "TRUE_FALL",
                action_taken: `Verified as ${status} via dashboard`,
            }),
        });
        await loadDashboard();
        await viewIncident(incidentId);
    } catch (err) {
        alert(`Verification failed: ${err.message}`);
    }
};

function displayAlerts() {
    if (alertsData.length === 0) {
        alertsContainer.innerHTML = `<p class="empty-message">No active alerts.</p>`;
        return;
    }

    alertsContainer.innerHTML = alertsData.map(a => `
        <article class="alert-card">
            <h3>Alert #${escapeHTML(a.id)} — Level ${escapeHTML(a.escalation_level)} (${escapeHTML(a.assigned_role)})</h3>
            <p><strong>Incident ID:</strong> ${escapeHTML(a.incident_id)} | <strong>Severity:</strong> <span class="status-label ${getStatusClass(a.severity)}">${escapeHTML(a.severity)}</span> | <strong>Status:</strong> ${escapeHTML(a.status)}</p>
            <p>${escapeHTML(a.message)}</p>
            <div class="alert-actions">
                ${a.status === "OPEN" ? `<button class="acknowledge-btn" data-alert-id="${a.id}" data-action="acknowledge">Acknowledge</button>` : ""}
                <button class="resolve-btn" data-alert-id="${a.id}" data-action="resolve">Resolve</button>
            </div>
        </article>
    `).join("");
}

function displayTasks() {
    if (tasksData.length === 0) {
        tasksContainer.innerHTML = `<p class="empty-message">No follow-up tasks generated yet.</p>`;
        return;
    }

    tasksContainer.innerHTML = tasksData.map(t => `
        <article class="alert-card" style="border-left-color: #2d82c7;">
            <h3>${escapeHTML(t.title)}</h3>
            <p><strong>Assigned To:</strong> ${escapeHTML(t.assigned_to)} | <strong>Priority:</strong> ${escapeHTML(t.priority)} | <strong>Status:</strong> <span class="status-label ${getStatusClass(t.status)}">${escapeHTML(t.status)}</span></p>
            <p>${escapeHTML(t.notes || "")}</p>
            ${t.status === "PENDING" ? `<div class="alert-actions"><button class="resolve-btn" onclick="completeFollowUpTask(${t.id})">Mark Assessment Completed</button></div>` : ""}
        </article>
    `).join("");
}

window.completeFollowUpTask = async function(taskId) {
    try {
        await apiRequest(`/api/tasks/${taskId}/complete`, { method: "POST" });
        await loadDashboard();
    } catch (err) {
        alert(`Error completing task: ${err.message}`);
    }
};

function displayEvaluation(evalRes) {
    const live = evalRes.live_summary || {};
    const comp = evalRes.verified_comparison || evalRes.benchmark_comparison;

    let tableHtml = `<p class="muted-note">Verify incidents (True Fall / False Alert) or run the pipeline on the 5 benchmark videos to populate the Unimodal vs. Multimodal comparison table.</p>`;
    if (comp) {
        const rows = ["video_only", "audio_only", "multimodal"].map(mode => {
            const m = comp[mode];
            return `
                <tr>
                    <td><strong>${mode.replace("_", " ").toUpperCase()}</strong></td>
                    <td>${(m.accuracy * 100).toFixed(1)}%</td>
                    <td>${(m.precision * 100).toFixed(1)}%</td>
                    <td>${(m.recall * 100).toFixed(1)}%</td>
                    <td>${(m.f1_score * 100).toFixed(1)}%</td>
                    <td>${(m.false_positive_rate * 100).toFixed(1)}%</td>
                </tr>
            `;
        }).join("");
        tableHtml = `
            <div class="table-container" style="margin-top:12px;">
                <table>
                    <thead><tr><th>Approach</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>False Positive Rate</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    evaluationContainer.innerHTML = `
        <p><strong>Verified True Falls:</strong> ${live.true_falls_confirmed ?? 0} | <strong>False Alerts:</strong> ${live.false_alerts_confirmed ?? 0} | <strong>Escalated Alerts:</strong> ${live.escalated_alerts_count ?? 0} | <strong>Mean Ack Time:</strong> ${live.avg_acknowledgement_seconds !== null && live.avg_acknowledgement_seconds !== undefined ? live.avg_acknowledgement_seconds + "s" : "N/A"}</p>
        ${tableHtml}
    `;
}

if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const videoFile = document.getElementById("video-file").files[0];
        if (!videoFile) return;

        const formData = new FormData();
        formData.append("video", videoFile);
        const audioFile = document.getElementById("audio-file").files[0];
        if (audioFile) formData.append("audio", audioFile);
        formData.append("resident_name", document.getElementById("resident-name").value);
        formData.append("room_location", document.getElementById("room-location").value);

        uploadBtn.disabled = true;
        uploadStatus.textContent = "Processing prerecorded media through MediaPipe, YAMNet, STT, VLM, and Fusion...";
        try {
            const res = await apiRequest("/api/pipeline/upload", { method: "POST", body: formData });
            uploadStatus.textContent = `Completed! Source ID: ${res.result.source_id}`;
            await loadDashboard();
        } catch (err) {
            uploadStatus.textContent = `Error: ${err.message}`;
        } finally {
            uploadBtn.disabled = false;
        }
    });
}

if (refreshButton) refreshButton.addEventListener("click", loadDashboard);

if (escalateDemoBtn) {
    escalateDemoBtn.addEventListener("click", async () => {
        await apiRequest("/api/alerts/escalate?force_step=true", { method: "POST" });
        await loadDashboard();
    });
}

if (resetDbBtn) {
    resetDbBtn.addEventListener("click", async () => {
        if (!confirm("Reset the SQLite database to a clean slate?")) return;
        await apiRequest("/api/system/reset", { method: "POST" });
        incidentDetailsSection.hidden = true;
        await loadDashboard();
    });
}

if (incidentFilter) incidentFilter.addEventListener("change", displayIncidents);

if (incidentsTableBody) {
    incidentsTableBody.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-incident-id]");
        if (btn) viewIncident(btn.dataset.incidentId);
    });
}

if (alertsContainer) {
    alertsContainer.addEventListener("click", async (e) => {
        const btn = e.target.closest("[data-action]");
        if (!btn) return;
        const id = btn.dataset.alertId;
        const action = btn.dataset.action;
        try {
            await apiRequest(`/api/alerts/${id}/${action}`, { method: "POST" });
            await loadDashboard();
        } catch (err) {
            alert(err.message);
        }
    });
}

loadDashboard();
