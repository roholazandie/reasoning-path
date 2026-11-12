// record buttons that save a figure's animation as a video. one button per single figure, and one per
// row of side-by-side canvases, which are recorded together into one clip.
(function () {
  "use strict";
  const FPS = 30;
  const BITRATE = 8e6;
  const MAX_WIDTH = 1920;
  const PAD = 20;
  const GAP = 16;
  const TITLE = 34;
  // mp4 where the browser can write it (chrome, edge, safari), webm otherwise (firefox)
  const TYPES = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
  const mimeType = typeof MediaRecorder === "undefined" ? null : TYPES.find((t) => MediaRecorder.isTypeSupported(t));
  const isMp4 = !!mimeType && mimeType.startsWith("video/mp4");
  const label = isMp4 ? "Record MP4" : "Record video";
  const recordings = new Set();

  const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const clock = (ms) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  /** Return the recordable groups: each row of side-by-side canvases, then each remaining single figure. */
  function findGroups() {
    const groups = [];
    const taken = new Set();
    document.querySelectorAll(".pair, .trio").forEach((row) => {
      const figs = [...row.querySelectorAll("figure.field")].filter((f) => f.querySelector("canvas"));
      figs.forEach((f) => taken.add(f));
      if (figs.length) groups.push({ anchor: row, figures: figs });
    });
    document.querySelectorAll("figure.field").forEach((f) => {
      if (!taken.has(f) && f.querySelector("canvas")) groups.push({ anchor: f, figures: [f] });
    });
    return groups;
  }

  // a row's controls are the next .controls after it; a single figure's are inside it
  function controlsFor(group) {
    if (group.figures.length === 1 && group.anchor.tagName === "FIGURE") {
      let c = group.anchor.querySelector(".controls");
      if (!c) {
        c = document.createElement("div");
        c.className = "controls";
        group.anchor.querySelector("canvas").after(c);
      }
      return c;
    }
    let next = group.anchor.nextElementSibling;
    if (next && next.classList.contains("controls")) return next;
    next = document.createElement("div");
    next.className = "controls";
    group.anchor.after(next);
    return next;
  }

  function mount(group) {
    const panel = group.anchor.closest("[role=tabpanel]");
    const heading = panel.querySelector("h1").textContent.trim();
    const step = panel.querySelector(".step")?.textContent.trim() || "";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rec";
    btn.textContent = label;
    if (!mimeType) {
      btn.disabled = true;
      btn.title = "This browser cannot record canvas video.";
    }
    controlsFor(group).append(btn);

    let session = null;

    function start() {
      const canvases = group.figures.map((f) => f.querySelector("canvas"));
      const css = getComputedStyle(panel);
      const v = (n) => css.getPropertyValue(n).trim();
      // size every canvas to the tallest one, then shrink the whole row if it is wider than full hd
      const h = Math.max(...canvases.map((c) => c.height));
      const widths = canvases.map((c) => (c.width * h) / c.height);
      // canvases hold device pixels, so margins and type scale by the same ratio to keep their size
      const dpr = window.devicePixelRatio || 1;
      let W = (PAD * 2 + GAP * (canvases.length - 1)) * dpr + widths.reduce((a, b) => a + b, 0);
      let H = (PAD * 2 + TITLE) * dpr + h;
      const k = Math.min(1, MAX_WIDTH / W);
      const S = dpr * k;
      // h.264 needs even dimensions
      W = Math.round((W * k) / 2) * 2;
      H = Math.round((H * k) / 2) * 2;
      const out = document.createElement("canvas");
      out.width = W;
      out.height = H;
      const ctx = out.getContext("2d");
      const grid = 24 * S;

      function frame() {
        // titles are read every frame, since some change while it runs (iterated rlhf counts its rounds)
        const titles = group.figures.map((f) => f.querySelector(".field-title")?.textContent.trim() || heading);
        ctx.fillStyle = v("--bg");
        ctx.fillRect(0, 0, W, H);
        let x = PAD * S;
        canvases.forEach((c, i) => {
          const w = widths[i] * k;
          const hh = h * k;
          const y = (PAD + TITLE) * S;
          // the plotting grid is a css background, so the canvas pixels do not carry it
          ctx.fillStyle = v("--paper");
          ctx.fillRect(x, y, w, hh);
          ctx.strokeStyle = v("--grid");
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let gx = x; gx <= x + w; gx += grid) { ctx.moveTo(gx + 0.5, y); ctx.lineTo(gx + 0.5, y + hh); }
          for (let gy = y; gy <= y + hh; gy += grid) { ctx.moveTo(x, gy + 0.5); ctx.lineTo(x + w, gy + 0.5); }
          ctx.stroke();
          ctx.drawImage(c, x, y, w, hh);
          ctx.strokeStyle = v("--line");
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, hh - 1);
          ctx.fillStyle = v("--ink");
          ctx.font = `500 ${Math.round(16 * S)}px ${v("--sans") || "sans-serif"}`;
          ctx.textBaseline = "alphabetic";
          ctx.fillText(titles[i], x, (PAD + TITLE - 12) * S);
          x += w + GAP * S;
        });
        if (session) session.raf = requestAnimationFrame(frame);
      }

      const stream = out.captureStream(FPS);
      const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: BITRATE });
      const chunks = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: mimeType.split(";")[0] });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `step-${step}-${slug(heading)}.${isMp4 ? "mp4" : "webm"}`;
        document.body.append(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      };
      session = { rec, started: performance.now(), raf: 0, timer: 0 };
      frame();
      rec.start(1000);
      session.timer = setInterval(() => {
        btn.textContent = `Stop and save ${clock(performance.now() - session.started)}`;
      }, 250);
      btn.classList.add("on");
      btn.textContent = "Stop and save 0:00";
      recordings.add(stop);
    }

    function stop() {
      if (!session) return;
      cancelAnimationFrame(session.raf);
      clearInterval(session.timer);
      session.rec.stop();
      session = null;
      btn.classList.remove("on");
      btn.textContent = label;
      recordings.delete(stop);
    }

    btn.addEventListener("click", () => (session ? stop() : start()));
    // leaving the step hides its canvases, so a recording would only show a frozen frame
    window.addEventListener("panel:show", (e) => {
      if (session && e.detail !== panel.id) stop();
    });
  }

  findGroups().forEach(mount);
  window.addEventListener("pagehide", () => [...recordings].forEach((s) => s()));
})();
