// ══════════════════════════════════════════════════════════════════════════════
// Tennis Elo Predictor - Modular App
// ══════════════════════════════════════════════════════════════════════════════

(function(global) {
  "use strict";

  // ────────────────────────────────────────────────────────────────────────────────────────────
  // State
  // ────────────────────────────────────────────────────────────────────────────────────────────
  const state = {
    data: null,
    selectedSurface: "Hard",
    rankSort: { col: "overall", dir: -1 },
    rankData: []
  };

  // ────────────────────────────────────────────────────────────────────────────────────────────
  // Utils
  // ────────────────────────────────────────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function $$(sel) { return document.querySelectorAll(sel); }

  function relInfo(matches) {
    if (matches >= 100) return { cls: "high", tier: "HIGH RELIABILITY", desc: "Rating well established" };
    if (matches >= 30)  return { cls: "mid",  tier: "MODERATE RELIABILITY", desc: "Some uncertainty in rating" };
    return { cls: "low", tier: "LOW RELIABILITY", desc: "Small sample — treat with caution" };
  }

  function matchBadge(m) {
    const cls = m >= 100 ? "badge-high" : m >= 30 ? "badge-mid" : "badge-low";
    return `<span class="match-badge ${cls}">${m.toLocaleString()}</span>`;
  }

  function fmtSurf(val) {
    if (!val || val === 0) return "—";
    return Math.round(val * 100) + "%";
  }

  // ────────────────────────────────────────────────────────────────────────────────────────────
  // Navigation Module
  // ────────────────────────────────────────────────────────────────────────────────────────────
  const Navigation = {
    init() {
      $$(".nav-tab").forEach(tab => {
        tab.addEventListener("click", () => {
          $$(".nav-tab, .page").forEach(x => x.classList.remove("active"));
          tab.classList.add("active");
          $("page-" + tab.dataset.page).classList.add("active");
        });
      });
    }
  };

  // ────────────────��───────────────────────────────────────────────────────────────────────────
  // Predict Module
  // ────────────────────────────────────────────────────────────────────────────────────────────
  const Predict = {
    init() {
      $("subtitle-params").textContent = "ATP Tour · Elo Model · 1991–2026";

      const players = state.data.players
        .filter(p => p.rank_points > 0)
        .sort((a, b) => a.name.localeCompare(b.name))
        .map(p => p.name);

      const sA = $("playerA"), sB = $("playerB");
      players.forEach(p => { sA.add(new Option(p, p)); sB.add(new Option(p, p)); });
      sA.value = players.includes("Jannik Sinner") ? "Jannik Sinner" : players[0];
      sB.value = players.includes("Carlos Alcaraz") ? "Carlos Alcaraz" : players[1];

      $$(".surf-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          $$(".surf-btn").forEach(x => x.classList.remove("active"));
          btn.classList.add("active");
          state.selectedSurface = btn.dataset.surface;
        });
      });

      this.updateConfidence();
    },

    updateConfidence() {
      if (!state.data) return;
      const pA = $("playerA").value;
      const pB = $("playerB").value;
      if (!pA || !pB) return;
      $("relStrip").style.display = "grid";

      const playerA = state.data.players.find(p => p.name === pA);
      const playerB = state.data.players.find(p => p.name === pB);

      [["A", pA, playerA], ["B", pB, playerB]].forEach(([side, name, player]) => {
        const m = player?.matches ?? 0;
        const info = relInfo(m);
        $("relCard" + side).className = `rel-card ${info.cls}`;
        $("relName" + side).textContent = name;
        $("relTier" + side).textContent = info.tier;
        $("relMeta" + side).textContent = `${m.toLocaleString()} matches · ${info.desc}`;
      });
    },

    run() {
      const pA = $("playerA").value;
      const pB = $("playerB").value;
      const err = $("errMsg");

      if (!pA || !pB || pA === pB) {
        err.style.display = "block";
        $("result").classList.remove("visible");
        return;
      }
      err.style.display = "none";

      const surfaceKey = state.selectedSurface === "Carpet" ? "overall" : state.selectedSurface;
      const key1 = pA + "|" + pB;
      const key2 = pB + "|" + pA;
      const prediction = state.data.predictions[key1] || state.data.predictions[key2];

      if (!prediction) {
        err.textContent = "⚠ No prediction data available for this matchup.";
        err.style.display = "block";
        return;
      }

      // If we found reverse key, invert the probability
      const isReversed = !state.data.predictions[key1] && state.data.predictions[key2];
      const pW = prediction[surfaceKey] || prediction.overall;
      const finalPW = isReversed ? 1 - pW : pW;

      const pLA = Math.round(finalPW * 100), pLB = 100 - pLA;

      $("nameA").textContent = pA;
      $("nameB").textContent = pB;
      $("pctA").textContent  = pLA + "%";
      $("pctB").textContent  = pLB + "%";
      $("probA").className = "player-prob " + (finalPW >= 0.5 ? "fav" : "dog");
      $("probB").className = "player-prob " + (finalPW <  0.5 ? "fav" : "dog");
      $("barFill").style.width = pLA + "%";

      const playerA = state.data.players.find(p => p.name === pA);
      const playerB = state.data.players.find(p => p.name === pB);

      const hiO = (playerA?.rank_points || 0) >= (playerB?.rank_points || 0);
      const hiS = (playerA?.["surface_" + state.selectedSurface] || 0.5) >= (playerB?.["surface_" + state.selectedSurface] || 0.5);

      $("d_nA").textContent = pA.split(" ").slice(-1)[0];
      $("d_nB").textContent = pB.split(" ").slice(-1)[0];
      $("d_eA").textContent = playerA?.rank_points || "—";
      $("d_eB").textContent = playerB?.rank_points || "—";
      $("d_eA").className   = "val" + (hiO  ? " hi" : "");
      $("d_eB").className   = "val right" + (!hiO ? " hi" : "");

      const surfaceLabel = state.selectedSurface === "Carpet" ? "Overall" : state.selectedSurface;
      $("d_sLbl").textContent = surfaceLabel + " Win Rate";
      $("d_sA").textContent = playerA ? Math.round((playerA["surface_" + state.selectedSurface] || playerA.win_rate) * 100) + "%" : "—";
      $("d_sB").textContent = playerB ? Math.round((playerB["surface_" + state.selectedSurface] || playerB.win_rate) * 100) + "%" : "—";
      $("d_sA").className   = "val" + (hiS  ? " hi" : "");
      $("d_sB").className   = "val right" + (!hiS ? " hi" : "");

      const fav  = finalPW >= 0.5 ? pA : pB;
      const pct  = Math.max(pLA, pLB);
      const conf = pct >= 70 ? "Strong favourite" : pct >= 60 ? "Moderate favourite" : "Slight edge";

      const mA = playerA?.matches ?? 0, mB = playerB?.matches ?? 0;
      const lowSample = mA < 30 || mB < 30;
      const vEl = $("verdict");
      vEl.className = "verdict" + (lowSample ? " warn" : "");
      vEl.textContent = `→ ${conf}: ${fav} (${pct}% on ${state.selectedSurface})`
        + (lowSample
          ? `\n⚠ Small sample — ${pA.split(" ").slice(-1)[0]}: ${mA} matches, ${pB.split(" ").slice(-1)[0]}: ${mB} matches. Rating may be noisy.`
          : "");

      $("result").classList.add("visible");
    }
  };

  // ────────────────────────────────────────────────────────────────────────────────────────────
  // Rankings Module
  // ────────────────────────────────────────────────────────────────────────────────────────────
  const Rankings = {
    init() {
      state.rankData = state.data.players
        .map((p, i) => ({ rank: i + 1, ...p }))
        .sort((a, b) => (b.rank_points || 0) - (a.rank_points || 0));
      state.rankSort = { col: "rank_points", dir: -1 };
      this.render();
    },

    sort(col) {
      const colMap = {
        "rank": "rank_points",
        "Hard": "surface_Hard",
        "Clay": "surface_Clay",
        "Grass": "surface_Grass",
        "Carpet": "surface_Carpet",
        "name": "name",
        "overall": "win_rate",
        "matches": "matches"
      };
      const dataKey = colMap[col] || col;

      state.rankSort.dir = state.rankSort.col === dataKey ? state.rankSort.dir * -1 : (col === "name" ? 1 : -1);
      state.rankSort.col = dataKey;
      $$("th").forEach(t => t.classList.remove("sorted"));

      const idMap = {
        "rank": "th_rank", "name": "th_name", "overall": "th_overall",
        "Hard": "th_Hard", "Clay": "th_Clay", "Grass": "th_Grass",
        "Carpet": "th_Carpet", "matches": "th_matches"
      };
      const el = $(idMap[col] || "th_" + col);
      if (el) el.classList.add("sorted");

      this.render();
    },

    render() {
      const q = $("searchInput").value.toLowerCase();
      const min = parseInt($("minMatches").value) || 0;
      const { col, dir } = state.rankSort;

      let data = state.rankData.filter(d => d.name.toLowerCase().includes(q) && (d.matches || 0) >= min);
      data.sort((a, b) => {
        const av = a[col];
        const bv = b[col];
        if (typeof av === "string") return av.localeCompare(bv) * dir;
        if (av == null || av === "") return 1;
        if (bv == null || bv === "") return -1;
        return ((av || 0) - (bv || 0)) * dir;
      });

      const body = $("rank-body");
      body.innerHTML = "";
      data.forEach((d, i) => {
        const best = Math.max(d.surface_Hard || 0, d.surface_Clay || 0, d.surface_Grass || 0, d.surface_Carpet || 0, d.win_rate || 0);

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="rank-n">${i + 1}</td>
          <td class="pname">${d.name}</td>
          <td class="${d.rank_points >= 5000 ? "hi" : ""}">${d.rank_points || "—"}</td>
          <td class="${(d.surface_Hard || d.win_rate) === best && d.surface_Hard ? "hi" : ""}">${fmtSurf(d.surface_Hard)}</td>
          <td class="${d.surface_Clay === best && d.surface_Clay ? "hi" : ""}">${fmtSurf(d.surface_Clay)}</td>
          <td class="${d.surface_Grass === best && d.surface_Grass ? "hi" : ""}">${fmtSurf(d.surface_Grass)}</td>
          <td class="${d.surface_Carpet === best && d.surface_Carpet ? "hi" : ""}">${fmtSurf(d.surface_Carpet)}</td>
          <td>${matchBadge(d.matches || 0)}</td>`;
        body.appendChild(tr);
      });
    }
  };

  // ─────────────────────────────────────────��──────────────────────────────────────────────────
  // App Bootstrap
  // ────────────────────────────────────────────────────────────────────────────────────────────
  function init() {
    // Check for predictions data
    if (typeof predictions !== "undefined") {
      state.data = predictions;
    } else if (typeof PREDICTIONS !== "undefined") {
      state.data = PREDICTIONS;
    } else {
      $("loading").innerHTML =
        `<div style="font-family:'DM Mono',monospace;color:var(--red);text-align:center;padding:2rem;max-width:440px">
          <div style="font-size:1.3rem;margin-bottom:.8rem">⚠ predictions.js not found</div>
          <div style="font-size:.76rem;color:#aaa;line-height:1.8">
            Run <span style="color:#c8f542">python export_predictions.py</span> first.<br>
            It generates <span style="color:#c8f542">predictions.js</span> — place it next to index.html, then refresh.
          </div>
        </div>`;
      return;
    }

    $("loading").classList.add("hidden");

    // Init modules
    Navigation.init();
    Predict.init();
    Rankings.init();

    // Expose globals for inline handlers
    global.updateConfidence = () => Predict.updateConfidence();
    global.predict = () => Predict.run();
    global.sortRank = (col) => Rankings.sort(col);
    global.renderRankings = () => Rankings.render();
  }

  // Start when DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})(window);