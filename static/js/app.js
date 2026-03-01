/**
 * Driver Drowsiness Detection — Dashboard JavaScript
 * Polls the Flask API for real-time stats and updates the UI.
 */

(function () {
    "use strict";

    // ─── DOM Elements ──────────────────────────────────────────

    const elements = {
        gaugeFill:      document.getElementById("gauge-fill"),
        gaugeValue:     document.getElementById("gauge-value"),
        gaugeLabel:     document.getElementById("gauge-label"),
        statusBanner:   document.getElementById("status-banner"),
        statusText:     document.getElementById("status-text"),
        earValue:       document.getElementById("ear-value"),
        marValue:       document.getElementById("mar-value"),
        earBar:         document.getElementById("ear-bar"),
        marBar:         document.getElementById("mar-bar"),
        blinkValue:     document.getElementById("blink-value"),
        yawnValue:      document.getElementById("yawn-value"),
        fpsDisplay:     document.getElementById("fps-display"),
        uptime:         document.getElementById("uptime"),
        eventList:      document.getElementById("event-list"),
        alertOverlay:   document.getElementById("alert-overlay"),
        videoOverlay:   document.getElementById("video-overlay"),
        videoFeed:      document.getElementById("video-feed"),
        connectionBadge: document.getElementById("connection-badge"),
        btnReset:       document.getElementById("btn-reset"),
    };

    // ─── Constants ─────────────────────────────────────────────

    const POLL_INTERVAL = 300;          // ms between stats polls
    const GAUGE_ARC_LENGTH = 251.2;     // Total arc stroke-dasharray

    const STATUS_MAP = {
        "ALERT":               { class: "alert",    icon: "✓", text: "Driver is Alert",           color: "#10b981" },
        "MILD DROWSINESS":     { class: "mild",     icon: "⚡", text: "Mild Drowsiness Detected",  color: "#f59e0b" },
        "MODERATE DROWSINESS": { class: "moderate", icon: "⚠️",  text: "Moderate Drowsiness!",      color: "#f97316" },
        "SEVERE DROWSINESS":   { class: "severe",   icon: "🚨", text: "SEVERE — Take a Break!",    color: "#ef4444" },
    };

    // ─── State ─────────────────────────────────────────────────

    let lastEventCount = 0;
    let connected = false;

    // ─── Gauge Update ──────────────────────────────────────────

    function updateGauge(score, status) {
        const offset = GAUGE_ARC_LENGTH * (1 - score / 100);
        elements.gaugeFill.style.strokeDashoffset = offset;

        const info = STATUS_MAP[status] || STATUS_MAP["ALERT"];
        elements.gaugeFill.style.stroke = info.color;
        elements.gaugeValue.textContent = Math.round(score);
        elements.gaugeLabel.textContent = status;
        elements.gaugeLabel.style.color = info.color;

        // Status banner
        elements.statusBanner.className = "status-banner " + info.class;
        elements.statusText.textContent = info.text;
        document.querySelector(".status-icon").textContent = info.icon;
    }

    // ─── Metrics Update ────────────────────────────────────────

    function updateMetrics(stats) {
        elements.earValue.textContent = stats.ear.toFixed(3);
        elements.marValue.textContent = stats.mar.toFixed(3);
        elements.blinkValue.textContent = stats.blinks;
        elements.yawnValue.textContent = stats.yawns;
        elements.fpsDisplay.textContent = stats.fps.toFixed(0) + " FPS";

        // EAR bar (inverted — lower EAR = more filled = more danger)
        const earPct = Math.max(0, Math.min(100, (1 - stats.ear / 0.4) * 100));
        elements.earBar.style.width = earPct + "%";
        if (stats.ear < 0.25) {
            elements.earBar.style.background = "linear-gradient(90deg, #f97316, #ef4444)";
        } else {
            elements.earBar.style.background = "linear-gradient(90deg, #10b981, #00d4ff)";
        }

        // MAR bar
        const marPct = Math.max(0, Math.min(100, (stats.mar / 1.0) * 100));
        elements.marBar.style.width = marPct + "%";
        if (stats.mar > 0.7) {
            elements.marBar.style.background = "linear-gradient(90deg, #f97316, #ef4444)";
        } else {
            elements.marBar.style.background = "linear-gradient(90deg, #00d4ff, #f97316)";
        }
    }

    // ─── Uptime ────────────────────────────────────────────────

    function formatUptime(seconds) {
        const h = Math.floor(seconds / 3600).toString().padStart(2, "0");
        const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
        const s = (seconds % 60).toString().padStart(2, "0");
        return h + ":" + m + ":" + s;
    }

    // ─── Events ────────────────────────────────────────────────

    function updateEvents(events) {
        if (!events || events.length === 0) return;
        if (events.length === lastEventCount) return;

        lastEventCount = events.length;
        elements.eventList.innerHTML = "";

        events.forEach(function (ev) {
            const item = document.createElement("div");
            item.className = "event-item";
            item.innerHTML =
                '<span class="event-time">' + ev.time + '</span>' +
                '<span class="event-type ' + ev.type + '">' + ev.type + '</span>' +
                '<span class="event-message">' + ev.message + '</span>';
            elements.eventList.appendChild(item);
        });
    }

    // ─── Alert Overlay ─────────────────────────────────────────

    function updateAlert(isDrowsy) {
        if (isDrowsy) {
            elements.alertOverlay.classList.add("active");
        } else {
            elements.alertOverlay.classList.remove("active");
        }
    }

    // ─── Connection Status ─────────────────────────────────────

    function setConnected(isConnected) {
        if (isConnected === connected) return;
        connected = isConnected;

        const badge = elements.connectionBadge;
        if (isConnected) {
            badge.innerHTML = '<span class="badge-dot"></span> Connected';
            badge.style.color = "#10b981";
            badge.style.borderColor = "rgba(16, 185, 129, 0.2)";
            badge.style.background = "rgba(16, 185, 129, 0.1)";
            elements.videoOverlay.classList.remove("visible");
        } else {
            badge.innerHTML = '<span class="badge-dot" style="background:#ef4444"></span> Disconnected';
            badge.style.color = "#ef4444";
            badge.style.borderColor = "rgba(239, 68, 68, 0.2)";
            badge.style.background = "rgba(239, 68, 68, 0.1)";
            elements.videoOverlay.classList.add("visible");
        }
    }

    // ─── Polling Loop ──────────────────────────────────────────

    function pollStats() {
        fetch("/api/stats")
            .then(function (res) {
                if (!res.ok) throw new Error("Network error");
                return res.json();
            })
            .then(function (stats) {
                setConnected(true);
                updateGauge(stats.drowsiness_score, stats.status);
                updateMetrics(stats);
                updateAlert(stats.drowsy_alert);
                updateEvents(stats.events);
                elements.uptime.textContent = formatUptime(stats.uptime);
            })
            .catch(function () {
                setConnected(false);
            });
    }

    // ─── Reset Button ──────────────────────────────────────────

    elements.btnReset.addEventListener("click", function () {
        fetch("/api/reset", { method: "POST" })
            .then(function () {
                lastEventCount = 0;
                elements.eventList.innerHTML =
                    '<div class="event-empty">Counters reset. Monitoring continues...</div>';
            })
            .catch(function (err) {
                console.error("Reset failed:", err);
            });
    });

    // ─── Video feed error handling ─────────────────────────────

    elements.videoFeed.addEventListener("error", function () {
        elements.videoOverlay.classList.add("visible");
        elements.videoOverlay.querySelector(".overlay-text").textContent =
            "Camera feed unavailable";
    });

    elements.videoFeed.addEventListener("load", function () {
        elements.videoOverlay.classList.remove("visible");
    });

    // ─── Start ─────────────────────────────────────────────────

    setInterval(pollStats, POLL_INTERVAL);
    pollStats();

})();
