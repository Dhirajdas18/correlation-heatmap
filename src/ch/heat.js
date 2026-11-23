// Correlation heatmap in the browser. Mirrors src/ch/cluster.py; tests compare the two.
// `cols` maps a symbol to its daily returns (null = no price that day).

function completeRows(cols, names) {  // days on which every name has a return
  const n = cols[names[0]].length, keep = [];
  for (let t = 0; t < n; t++) if (names.every((k) => cols[k][t] !== null && cols[k][t] !== undefined)) keep.push(t);
  return keep;
}

function corrOf(series) {  // Pearson correlation matrix of equal-length arrays
  const k = series.length, n = series[0].length;
  const z = series.map((x) => {
    const m = x.reduce((a, b) => a + b, 0) / n;
    const d = x.map((v) => v - m), s = Math.sqrt(d.reduce((a, b) => a + b * b, 0));
    return d.map((v) => v / s);
  });
  const c = [];
  for (let i = 0; i < k; i++) {
    c.push([]);
    for (let j = 0; j < k; j++) { let s = 0; for (let t = 0; t < n; t++) s += z[i][t] * z[j][t]; c[i].push(s); }
  }
  return c;
}

function ranks(x) {  // ranks 1..n, ties get their average rank (as pandas and scipy do)
  const idx = x.map((v, i) => i).sort((a, b) => x[a] - x[b]), r = new Array(x.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && x[idx[j + 1]] === x[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}

// Weekly (Friday-ending, like pandas "W-FRI") compounded returns; a week is null for a name if
// any of its days is missing.
function toWeekly(cols, dates) {
  const friday = (iso) => { const d = new Date(iso + "T00:00:00Z"), add = (5 - d.getUTCDay() + 7) % 7;
    d.setUTCDate(d.getUTCDate() + add); return d.toISOString().slice(0, 10); };
  const keys = dates.map(friday), weeks = [...new Set(keys)], out = {};
  for (const [k, v] of Object.entries(cols)) {
    const acc = new Map(weeks.map((w) => [w, 1]));
    v.forEach((x, t) => { const w = keys[t]; if (acc.get(w) === null) return; acc.set(w, x === null ? null : acc.get(w) * (1 + x)); });
    out[k] = weeks.map((w) => acc.get(w) === null ? null : acc.get(w) - 1);
  }
  return { cols: out, dates: weeks };
}

// Correlation of the names' returns on complete days; with `index`, of what is left after
// taking out each name's beta × the index's return (beta from the same days);
// `rank` correlates ranks instead of values (Spearman).
function corrMatrix(cols, names, index, rank = false) {
  const rows = completeRows(cols, index ? names.concat([index]) : names);
  let series = names.map((k) => rows.map((t) => cols[k][t]));
  if (index) {
    const m = rows.map((t) => cols[index][t]), n = m.length, mm = m.reduce((a, b) => a + b, 0) / n;
    const vm = m.reduce((a, b) => a + (b - mm) ** 2, 0);
    series = series.map((x) => {
      const mx = x.reduce((a, b) => a + b, 0) / n;
      let cov = 0; for (let t = 0; t < n; t++) cov += (x[t] - mx) * (m[t] - mm);
      const beta = cov / vm;
      return x.map((v, t) => v - beta * m[t]);
    });
  }
  if (rank) series = series.map(ranks);
  return { c: corrOf(series), days: rows.length };
}

// Average-linkage clustering on d = √(2(1 − ρ)); returns the leaf order (each merge puts its
// left group before its right) and cluster labels when cut into k groups.
function clusterOrder(c, k) {
  const n = c.length, dist = (i, j) => Math.sqrt(Math.max(0, 2 * (1 - c[i][j])));
  let groups = Array.from({ length: n }, (_, i) => ({ members: [i], order: [i] }));
  const between = (a, b) => { let s = 0; for (const i of a.members) for (const j of b.members) s += dist(i, j); return s / (a.members.length * b.members.length); };
  let labelsAt = null;
  while (groups.length > 1) {
    if (groups.length === k) labelsAt = groups.map((g) => g.members.slice());
    let best = [0, 1], bd = Infinity;
    for (let a = 0; a < groups.length; a++) for (let b = a + 1; b < groups.length; b++) {
      const d = between(groups[a], groups[b]);
      if (d < bd) { bd = d; best = [a, b]; }
    }
    const [a, b] = best, merged = { members: groups[a].members.concat(groups[b].members), order: groups[a].order.concat(groups[b].order) };
    groups = groups.filter((_, i) => i !== a && i !== b).concat([merged]);
  }
  if (!labelsAt) labelsAt = [Array.from({ length: n }, (_, i) => i)];
  const labels = new Array(n);
  labelsAt.forEach((members, g) => members.forEach((i) => { labels[i] = g + 1; }));
  return { order: groups[0].order, labels };
}

// Correlation of two names over each trailing `w` days (null until `w` complete days).
function rollingPair(cols, a, b, w, index) {
  const x = cols[a], y = cols[b], n = x.length, out = new Array(n).fill(null);
  for (let t = w - 1; t < n; t++) {
    const xs = [], ys = [];
    for (let s = t - w + 1; s <= t; s++) {
      if (x[s] === null || y[s] === null || (index && cols[index][s] === null)) break;
      xs.push(x[s]); ys.push(y[s]);
    }
    if (xs.length < w) continue;
    out[t] = corrOf([xs, ys])[0][1];
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { ranks, toWeekly, completeRows, corrOf, corrMatrix, clusterOrder, rollingPair };
