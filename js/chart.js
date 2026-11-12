// small canvas line chart with a legend, a highlighted point, and a hover crosshair with tooltip.
(function (root) {
  "use strict";

  const PAD = { left: 40, right: 12, top: 10, bottom: 30 };

  function niceTicks(min, max, count) {
    const span = max - min || 1;
    const raw = span / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(+t.toFixed(10));
    return ticks;
  }

  /**
   * Mount a chart in container and return {update}. opts: series [{name, color (css var), values}],
   * x (shared x values), xLabel, yLabel, xFormat, yFormat, markers, highlight (index), tooltip(i).
   */
  function lineChart(container, initial) {
    let opts = initial;
    container.classList.add("chart");
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    const wrap = document.createElement("div");
    wrap.className = "chart-plot";
    const canvas = document.createElement("canvas");
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    tip.hidden = true;
    wrap.append(canvas, tip);
    container.append(legend, wrap);
    const ctx = canvas.getContext("2d");
    let width = 0;
    let height = 0;
    let hover = -1;
    let frame = null;

    function css(name) {
      return getComputedStyle(container).getPropertyValue(name).trim();
    }

    function renderLegend() {
      // a single series is named by the chart title, so it gets no legend
      legend.hidden = opts.series.length < 2;
      legend.innerHTML = opts.series
        // a dashed series gets a dashed swatch, so the legend carries the same second encoding as the line
        .map((s) => `<span><i class="${s.dash ? "dashed" : ""}" style="${s.dash ? `border-top:2px dashed var(${s.color});background:none` : `background:var(${s.color})`}"></i>${s.name}</span>`)
        .join("");
    }

    function bounds() {
      const xs = opts.x;
      const ys = opts.series.flatMap((s) => s.values).filter(Number.isFinite);
      // most charts keep zero on the axis; yFromData zooms in for slow changes such as a mean length
      let y0 = opts.yFromData ? Math.min(...ys) : Math.min(0, ...ys);
      let y1 = Math.max(...ys, opts.yMinTop ?? 0);
      if (!ys.length) [y0, y1] = [0, 1];
      if (opts.yMax !== undefined) y1 = opts.yMax;
      const pad = (y1 - y0) * 0.08 || 1;
      return { x0: opts.xMin ?? Math.min(...xs), x1: opts.xMax ?? Math.max(...xs), y0: y0 < 0 || opts.yFromData ? y0 - pad : y0, y1: y1 + pad };
    }

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      width = wrap.clientWidth;
      height = wrap.clientHeight;
      if (!width || !height) return;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const font = getComputedStyle(container).fontFamily;
      const muted = css("--muted");
      const grid = css("--grid-strong");
      const ink = css("--ink");
      if (!opts.x.length) {
        ctx.fillStyle = muted;
        ctx.font = `13px ${font}`;
        ctx.textAlign = "center";
        ctx.fillText(opts.empty || "No data yet", width / 2, height / 2);
        frame = null;
        return;
      }
      const b = bounds();
      // a log axis maps through log10, so rollout counts from tens to tens of thousands share one chart
      const T = opts.xLog ? Math.log10 : (x) => x;
      const px = (x) => PAD.left + ((T(x) - T(b.x0)) / (T(b.x1) - T(b.x0) || 1)) * (width - PAD.left - PAD.right);
      const py = (y) => PAD.top + (1 - (y - b.y0) / (b.y1 - b.y0)) * (height - PAD.top - PAD.bottom);
      frame = { px, py, b };

      ctx.font = `11px ${font}`;
      ctx.lineWidth = 1;
      ctx.fillStyle = muted;
      ctx.textAlign = "right";
      for (const t of niceTicks(b.y0, b.y1, 4)) {
        ctx.strokeStyle = t === 0 ? css("--line-strong") : grid;
        ctx.beginPath();
        ctx.moveTo(PAD.left, py(t));
        ctx.lineTo(width - PAD.right, py(t));
        ctx.stroke();
        ctx.fillText((opts.yFormat || String)(t), PAD.left - 6, py(t) + 3.5);
      }
      ctx.textAlign = "center";
      const xTicks = opts.xLog
        ? Array.from({ length: 8 }, (_, i) => 10 ** i).filter((t) => t >= b.x0 && t <= b.x1)
        : niceTicks(b.x0, b.x1, 5);
      for (const t of xTicks) ctx.fillText((opts.xFormat || String)(t), px(t), height - PAD.bottom + 14);
      // right-aligned so the title never sits on a middle tick label
      ctx.textAlign = "right";
      ctx.fillText(opts.xLabel || "", width - PAD.right, height - 3);
      ctx.textAlign = "center";

      for (const s of opts.series) {
        const color = css(s.color);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        // a missing value breaks the line, for runs that stop before the axis ends
        let pen = false;
        s.values.forEach((v, i) => {
          if (!Number.isFinite(v)) {
            pen = false;
            return;
          }
          if (pen) ctx.lineTo(px(opts.x[i]), py(v));
          else ctx.moveTo(px(opts.x[i]), py(v));
          pen = true;
        });
        ctx.stroke();
        ctx.setLineDash([]);
        if (opts.markers) {
          s.values.forEach((v, i) => {
            if (!Number.isFinite(v)) return;
            ctx.beginPath();
            ctx.arc(px(opts.x[i]), py(v), i === opts.highlight ? 5.5 : 3.5, 0, 2 * Math.PI);
            ctx.fill();
            // a surface ring keeps overlapping markers apart
            ctx.strokeStyle = css("--paper");
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.strokeStyle = color;
          });
        }
      }
      const mark = hover >= 0 ? hover : opts.highlight;
      if (mark !== undefined && mark >= 0 && mark < opts.x.length) {
        ctx.strokeStyle = ink;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(px(opts.x[mark]), PAD.top);
        ctx.lineTo(px(opts.x[mark]), height - PAD.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }

    function showTip(i, mx) {
      if (i < 0 || !opts.tooltip) {
        tip.hidden = true;
        return;
      }
      tip.innerHTML = opts.tooltip(i);
      tip.hidden = false;
      const left = Math.min(Math.max(mx + 12, 0), width - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(0, left)}px`;
      tip.style.top = "4px";
    }

    canvas.addEventListener("pointermove", (e) => {
      if (!frame) return;
      const r = canvas.getBoundingClientRect();
      const mx = e.clientX - r.left;
      let best = -1;
      let bestD = Infinity;
      opts.x.forEach((x, i) => {
        const d = Math.abs(frame.px(x) - mx);
        if (d < bestD) [best, bestD] = [i, d];
      });
      hover = bestD < 40 ? best : -1;
      draw();
      showTip(hover, mx);
    });
    canvas.addEventListener("pointerleave", () => {
      hover = -1;
      tip.hidden = true;
      draw();
    });
    new ResizeObserver(draw).observe(wrap);

    renderLegend();
    draw();
    return {
      update(next) {
        opts = { ...opts, ...next };
        renderLegend();
        draw();
      },
      redraw: draw,
    };
  }

  root.MiniChart = { lineChart };
})(globalThis);
