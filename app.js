(() => {
  const $ = (id) => document.getElementById(id);
  const palette = { coral: '#ed806b', mint: '#36a88b', grid: '#edf0ee', text: '#8d989d', ink: '#68757b' };
  const els = {
    bidders: $('bidders'), biddersValue: $('biddersValue'), noise: $('noise'), noiseValue: $('noiseValue'), market: $('market'),
    ruleNote: $('ruleNote'), epoch: $('epochCount'), state: $('trainingState'), progress: $('progressBar'), dLoss: $('dLoss'), gLoss: $('gLoss'), quality: $('quality'),
    train: $('trainButton'), step: $('stepButton'), stop: $('stopButton'), toast: $('toast')
  };
  let rule = 'first', generator, discriminator, combined, real = [], generated = [], lossesD = [], lossesG = [], epoch = 0;
  let running = false, trainingNow = false, toastTimer, refreshTimer;

  function gaussian() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function valuation() {
    if (els.market.value === 'bimodal') return clamp(Math.random() < .56 ? .31 + gaussian() * .105 : .72 + gaussian() * .105, .015, .99);
    if (els.market.value === 'skewed') return clamp(Math.pow(Math.random(), 1.85), .01, .99);
    return .015 + Math.random() * .97;
  }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function condition() { return { rule: rule === 'second' ? 1 : 0, n: Number(els.bidders.value), noise: Number(els.noise.value) / 100 }; }
  function sampleMarket(count = 256) {
    const c = condition();
    return Array.from({ length: count }, () => {
      const value = valuation();
      const ideal = c.rule ? .985 : (c.n - 1) / c.n;
      const ratio = clamp(ideal + gaussian() * c.noise * (c.rule ? .2 : .36), .04, 1.04);
      return { value, ratio, bid: value * ratio };
    });
  }
  function initModel() {
    if (!window.tf) return false;
    disposeModels();
    const tf = window.tf;
    const gi = tf.input({ shape: [3] });
    let gx = tf.layers.dense({ units: 32, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(gi);
    gx = tf.layers.dense({ units: 32, activation: 'relu' }).apply(gx);
    const go = tf.layers.dense({ units: 2, activation: 'sigmoid' }).apply(gx);
    generator = tf.model({ inputs: gi, outputs: go });

    const di = tf.input({ shape: [4] });
    let dx = tf.layers.dense({ units: 32, activation: 'relu' }).apply(di);
    dx = tf.layers.dense({ units: 32, activation: 'relu' }).apply(dx);
    const dout = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(dx);
    discriminator = tf.model({ inputs: di, outputs: dout });
    discriminator.compile({ optimizer: tf.train.adam(.0007, .5), loss: 'binaryCrossentropy' });
    discriminator.trainable = false;
    const noiseInput = tf.input({ shape: [1] });
    const conditionInput = tf.input({ shape: [2] });
    const ci = tf.layers.concatenate().apply([noiseInput, conditionInput]);
    const fake = generator.apply(ci);
    const joined = tf.layers.concatenate().apply([fake, conditionInput]);
    const validity = discriminator.apply(joined);
    combined = tf.model({ inputs: [noiseInput, conditionInput], outputs: validity });
    combined.compile({ optimizer: tf.train.adam(.0007, .5), loss: 'binaryCrossentropy' });
    discriminator.trainable = true;
    return true;
  }
  function disposeModels() { [generator, discriminator, combined].forEach((m) => { if (m) m.dispose(); }); generator = discriminator = combined = null; }
  function genConditions(count, noise = true) {
    const c = condition();
    return Array.from({ length: count }, () => [noise ? Math.random() * 2 - 1 : 0, c.rule, (c.n - 2) / 18]);
  }
  function packDiscriminator(samples) {
    const c = condition();
    return samples.map((s) => [s.value, s.ratio, c.rule, (c.n - 2) / 18]);
  }
  function getGenerated(count = 256) {
    if (!generator) return Array.from({ length: count }, () => ({ value: Math.random(), ratio: Math.random(), bid: Math.random() }));
    const tf = window.tf;
    const input = tf.tensor2d(genConditions(count));
    const output = generator.predict(input);
    const values = output.dataSync();
    const samples = Array.from({ length: count }, (_, i) => ({ value: values[i * 2], ratio: values[i * 2 + 1] * 1.05, bid: values[i * 2] * values[i * 2 + 1] * 1.05 }));
    input.dispose(); output.dispose();
    return samples;
  }
  function scalar(x) { if (Array.isArray(x)) return scalar(x[0]); if (x && typeof x.dataSync === 'function') { const v = x.dataSync()[0]; x.dispose(); return v; } return Number(x); }
  async function trainOne() {
    if (!generator || trainingNow) return false;
    trainingNow = true;
    const tf = window.tf, batch = 64, c = condition();
    const freshReal = sampleMarket(batch);
    const realX = tf.tensor2d(packDiscriminator(freshReal));
    const realY = tf.ones([batch, 1]);
    const gIn = tf.tensor2d(genConditions(batch));
    const fakePair = generator.predict(gIn);
    const ruleCol = tf.fill([batch, 1], c.rule), nCol = tf.fill([batch, 1], (c.n - 2) / 18);
    const fakeX = tf.concat([fakePair, ruleCol, nCol], 1);
    const fakeY = tf.zeros([batch, 1]);
    const dInput = tf.concat([realX, fakeX], 0);
    const dTargets = tf.concat([realY, fakeY], 0);
    const dl = scalar(await discriminator.trainOnBatch(dInput, dTargets));
    const noiseInput = tf.slice(gIn, [0, 0], [-1, 1]);
    const conditionInput = tf.slice(gIn, [0, 1], [-1, 2]);
    const gTargets = tf.ones([batch, 1]);
    const gl = scalar(await combined.trainOnBatch([noiseInput, conditionInput], gTargets));
    [realX, realY, gIn, fakePair, ruleCol, nCol, fakeX, fakeY, dInput, dTargets, noiseInput, conditionInput, gTargets].forEach((t) => t.dispose());
    trainingNow = false;
    epoch += 1; lossesD.push(dl); lossesG.push(gl);
    if (lossesD.length > 220) { lossesD.shift(); lossesG.shift(); }
    return true;
  }
  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }
  function axes(ctx, w, h, xLabel, yLabel, maxX = 1, maxY = 1) {
    const p = { l: 32, r: 8, t: 8, b: 27 }, cw = w - p.l - p.r, ch = h - p.t - p.b;
    ctx.clearRect(0, 0, w, h); ctx.font = '8px "DM Mono", monospace'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = p.t + ch * i / 4; ctx.strokeStyle = palette.grid; ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(w - p.r, y); ctx.stroke();
      ctx.fillStyle = palette.text; ctx.textAlign = 'right'; ctx.fillText((maxY * (1 - i / 4)).toFixed(maxY <= 1 ? 1 : 0), p.l - 6, y + 3);
      const x = p.l + cw * i / 4; ctx.strokeStyle = palette.grid; ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, p.t + ch); ctx.stroke();
      ctx.textAlign = 'center'; ctx.fillStyle = palette.text; ctx.fillText((maxX * i / 4).toFixed(maxX <= 1 ? 1 : 0), x, h - 13);
    }
    ctx.fillStyle = palette.ink; ctx.font = '8px "Noto Sans SC", sans-serif'; ctx.textAlign = 'center'; ctx.fillText(xLabel, p.l + cw / 2, h - 1);
    ctx.save(); ctx.translate(9, p.t + ch / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
    return { p, cw, ch };
  }
  function drawScatter() {
    const plot = fitCanvas($('scatterChart')); if (!plot) return;
    const { ctx, w, h } = plot, { p, cw, ch } = axes(ctx, w, h, '估值', '报价');
    const dots = (arr, color) => { ctx.fillStyle = color; arr.forEach((s) => { const x = p.l + clamp(s.value, 0, 1) * cw, y = p.t + ch - clamp(s.bid, 0, 1) * ch; ctx.globalAlpha = .57; ctx.beginPath(); ctx.arc(x, y, 2.1, 0, Math.PI * 2); ctx.fill(); }); ctx.globalAlpha = 1; };
    dots(real, palette.coral); dots(generated, palette.mint);
  }
  function histogram(arr, bins = 12) { const counts = new Array(bins).fill(0); arr.forEach((s) => { const i = clamp(Math.floor(s.ratio / 1.2 * bins), 0, bins - 1); counts[i]++; }); return counts.map((x) => x / Math.max(1, arr.length)); }
  function drawDensity() {
    const plot = fitCanvas($('densityChart')); if (!plot) return;
    const { ctx, w, h } = plot, bins = 12, realBins = histogram(real, bins), fakeBins = histogram(generated, bins), max = Math.max(.15, ...realBins, ...fakeBins) * 1.15;
    const { p, cw, ch } = axes(ctx, w, h, '报价 / 估值', '样本占比', 1.2, max);
    const bw = cw / bins;
    const bars = (values, color, offset) => values.forEach((v, i) => { ctx.fillStyle = color; ctx.globalAlpha = .76; ctx.fillRect(p.l + i * bw + offset, p.t + ch - v / max * ch, bw * .35, v / max * ch); });
    bars(realBins, palette.coral, bw * .12); bars(fakeBins, palette.mint, bw * .52); ctx.globalAlpha = 1;
  }
  function drawLoss() {
    const plot = fitCanvas($('lossChart')); if (!plot) return;
    const { ctx, w, h } = plot, max = Math.max(2, ...lossesD, ...lossesG) * 1.08;
    const { p, cw, ch } = axes(ctx, w, h, '训练迭代', '损失值', Math.max(100, epoch), max);
    const line = (values, color) => { if (values.length < 2) return; ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.lineJoin = 'round'; ctx.beginPath(); const start = Math.max(0, epoch - values.length); values.forEach((v, i) => { const x = p.l + (start + i) / Math.max(100, epoch) * cw, y = p.t + ch - v / max * ch; if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); };
    line(lossesD, palette.coral); line(lossesG, palette.mint);
  }
  function evaluate(arr) {
    const n = condition().n, groups = Math.floor(arr.length / n); let revenue = 0, efficient = 0, count = 0;
    for (let g = 0; g < groups; g++) {
      const lot = arr.slice(g * n, (g + 1) * n).slice().sort((a, b) => b.bid - a.bid);
      if (lot.length < 2) continue;
      revenue += rule === 'first' ? lot[0].bid : lot[1].bid;
      const bestValue = Math.max(...lot.map((x) => x.value)); if (lot[0].value >= bestValue - 1e-9) efficient++;
      count++;
    }
    return { revenue: count ? revenue / count : 0, efficiency: count ? efficient / count : 0 };
  }
  function updateMetrics() {
    const a = evaluate(real), b = evaluate(generated);
    $('realRevenue').textContent = a.revenue.toFixed(3); $('fakeRevenue').textContent = b.revenue.toFixed(3);
    $('realEfficiency').textContent = `${Math.round(a.efficiency * 100)}%`; $('fakeEfficiency').textContent = `${Math.round(b.efficiency * 100)}%`;
    const h1 = histogram(real), h2 = histogram(generated), gap = h1.reduce((s, x, i) => s + Math.abs(x - h2[i]), 0) / 2;
    const quality = $('quality'), insight = $('insightText');
    if (epoch < 10) { quality.textContent = '学习中'; insight.textContent = '生成器正在探索估值与报价之间的关系。继续训练，观察两组分布逐步靠近。'; }
    else if (gap < .18) { quality.textContent = '分布接近'; insight.textContent = '生成样本的折扣率分布已接近模拟市场。成交价和配置效率仍会受抽样波动影响。'; }
    else { quality.textContent = '持续拟合'; insight.textContent = '生成样本仍与市场分布存在差异。试着增加迭代，或调整竞买人数与报价噪声。'; }
  }
  function render() { drawScatter(); drawDensity(); drawLoss(); updateMetrics(); }
  function showToast(message) { els.toast.textContent = message; els.toast.classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => els.toast.classList.remove('visible'), 2400); }
  function setRuleNote() { els.ruleNote.textContent = rule === 'first' ? '第一价格拍卖下，竞买者需要在提高中标概率与压低支付价格之间权衡。' : '第二价格拍卖下，诚实报价是私人价值模型中的基准策略。'; }
  function updateSliders() {
    els.biddersValue.textContent = `${els.bidders.value} 人`;
    const n = Number(els.noise.value); els.noiseValue.textContent = n < 8 ? '低' : n < 25 ? '中等' : '高';
    [els.bidders, els.noise].forEach((input) => { const pct = (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min)) * 100; input.style.background = `linear-gradient(90deg,${palette.coral} ${pct}%,#e8e9e8 ${pct}%)`; });
  }
  function reset() {
    running = false; els.train.disabled = false; els.train.innerHTML = '<span class="play-icon">▶</span> 开始训练';
    epoch = 0; lossesD = []; lossesG = []; els.epoch.textContent = '0000'; els.dLoss.textContent = '—'; els.gLoss.textContent = '—'; els.state.textContent = '等待开始'; els.progress.style.width = '0%';
    real = sampleMarket(256); generated = getGenerated(256); $('sampleCount').textContent = '256';
    render();
  }
  async function startLoop() {
    if (running) return;
    if (!generator) { showToast('模型尚未准备好，请稍候片刻'); return; }
    running = true; els.train.disabled = true; els.state.textContent = '正在对抗训练';
    const loop = async () => {
      if (!running) return;
      try {
        const ok = await trainOne(); if (!ok) { setTimeout(loop, 80); return; }
        els.epoch.textContent = String(epoch).padStart(4, '0'); els.dLoss.textContent = lossesD.at(-1).toFixed(3); els.gLoss.textContent = lossesG.at(-1).toFixed(3); els.progress.style.width = `${Math.min(100, epoch / 2)}%`;
        if (epoch % 4 === 0) { generated = getGenerated(256); render(); }
        setTimeout(loop, 35);
      } catch (err) { running = false; els.train.disabled = false; els.state.textContent = '训练遇到问题'; console.error(err); showToast('训练暂时中断，请重新生成市场后再试'); }
    };
    els.train.innerHTML = '<span class="play-icon">◌</span> 正在训练'; loop();
  }
  async function singleStep() {
    if (!generator || trainingNow) { showToast('模型正在准备，请稍候'); return; }
    els.state.textContent = '单步训练中';
    try {
      await trainOne(); els.epoch.textContent = String(epoch).padStart(4, '0'); els.dLoss.textContent = lossesD.at(-1).toFixed(3); els.gLoss.textContent = lossesG.at(-1).toFixed(3); els.progress.style.width = `${Math.min(100, epoch / 2)}%`;
      generated = getGenerated(256); render(); els.state.textContent = '单步完成';
    } catch (err) { console.error(err); els.state.textContent = '训练遇到问题'; showToast('单步训练失败，请重置模拟市场'); }
  }
  function scheduleReset() { clearTimeout(refreshTimer); running = false; updateSliders(); const apply = () => { if (trainingNow) { refreshTimer = setTimeout(apply, 70); return; } initModel(); reset(); showToast('市场参数已更新，模型已重置'); }; refreshTimer = setTimeout(apply, 180); }

  document.querySelectorAll('.segment').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('.segment').forEach((b) => { b.classList.remove('selected'); b.setAttribute('aria-pressed', 'false'); });
    button.classList.add('selected'); button.setAttribute('aria-pressed', 'true'); rule = button.dataset.rule; setRuleNote(); scheduleReset();
  }));
  [els.bidders, els.noise, els.market].forEach((input) => input.addEventListener('input', scheduleReset));
  $('resetData').addEventListener('click', () => { scheduleReset(); });
  els.train.addEventListener('click', startLoop);
  els.step.addEventListener('click', singleStep);
  els.stop.addEventListener('click', () => { running = false; els.train.disabled = false; els.train.innerHTML = '<span class="play-icon">▶</span> 继续训练'; els.state.textContent = '训练已暂停'; });
  window.addEventListener('resize', () => requestAnimationFrame(render));

  async function boot() {
    updateSliders(); setRuleNote(); real = sampleMarket(256);
    if (!window.tf) { generated = sampleMarket(256); render(); els.state.textContent = '可视化预览'; showToast('模型库加载失败；图表仍可用于查看模拟数据'); return; }
    try { await window.tf.ready(); initModel(); generated = getGenerated(256); render(); els.state.textContent = '模型就绪'; }
    catch (err) { console.error(err); generated = sampleMarket(256); render(); els.state.textContent = '可视化预览'; showToast('模型初始化失败；请检查网络后刷新页面'); }
  }
  boot();
})();
