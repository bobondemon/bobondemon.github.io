// 互動示範: convolution 的 kernel size / stride / padding
// 用法（文章原稿）: <div data-widget="conv-stride" data-pagefind-ignore>...替代內容...</div>
// 可選屬性: data-n (輸入大小, 預設 7)、data-k (kernel, 預設 3)、data-s (stride, 預設 1)、data-p (padding, 預設 0)
// 純 JS + SVG，不依賴任何函式庫；顏色用半透明灰與一個強調色，淺色/深色主題都能用。

const STYLE = `
.cw { --cw-acc: #2f80ed; font-size: 0.9rem; margin: 1.5rem 0; padding: 1rem; border: 1px solid rgba(127,127,127,.35); border-radius: 8px; }
.cw * { box-sizing: border-box; }
.cw-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr)); gap: .5rem 1.5rem; margin-bottom: 1rem; }
.cw-controls label { display: grid; grid-template-columns: 8.5em minmax(0, 1fr) 1.5em; align-items: center; gap: .5rem; }
.cw-controls input { min-width: 0; width: 100%; }
.cw-controls output { text-align: right; font-variant-numeric: tabular-nums; }
.cw-stage { display: flex; flex-wrap: wrap; gap: 1rem 2rem; align-items: flex-start; justify-content: center; }
.cw figure { margin: 0; text-align: center; }
.cw figcaption { margin-bottom: .35rem; opacity: .75; }
.cw svg { max-width: 100%; height: auto; touch-action: manipulation; }
.cw .cell { fill: rgba(127,127,127,.08); stroke: rgba(127,127,127,.45); stroke-width: 1; }
.cw .pad { fill: rgba(127,127,127,.03); stroke-dasharray: 3 3; }
.cw .hit { fill: rgba(47,128,237,.22); }
.cw .win { fill: none; stroke: var(--cw-acc); stroke-width: 2.5; pointer-events: none; }
.cw .sel { stroke: var(--cw-acc); stroke-width: 2.5; fill: rgba(47,128,237,.25); }
.cw .out-cell { cursor: pointer; }
.cw .out-cell:focus { outline: none; }
.cw .out-cell:focus-visible .cell { stroke: var(--cw-acc); stroke-width: 3; }
.cw text { fill: currentColor; font-size: 12px; text-anchor: middle; dominant-baseline: central; pointer-events: none; font-variant-numeric: tabular-nums; }
.cw .padtext { opacity: .35; }
.cw .formula { margin: 1rem 0 .5rem; text-align: center; font-variant-numeric: tabular-nums; }
.cw .warn { color: #d4380d; }
.cw .actions { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.cw button { font: inherit; padding: .25rem .9rem; border: 1px solid rgba(127,127,127,.5); border-radius: 6px; background: transparent; color: inherit; cursor: pointer; }
.cw button:hover { border-color: var(--cw-acc); }
.cw .hint { margin: .5rem 0 0; text-align: center; opacity: .6; font-size: .8rem; }
`

const SVG = 'http://www.w3.org/2000/svg'
const CELL = 30

function node(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVG, tag)
  for (const [k, v] of Object.entries(attrs)) {
    n.setAttribute(k, v)
  }
  if (parent) {
    parent.appendChild(n)
  }
  return n
}

function injectStyle() {
  if (document.getElementById('cw-style')) {
    return
  }
  const style = document.createElement('style')
  style.id = 'cw-style'
  style.textContent = STYLE
  document.head.appendChild(style)
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
const num = (v, d) => (Number.isFinite(Number(v)) && v !== undefined && v !== '' ? Number(v) : d)

// 固定種子的偽隨機整數 0..9，同一個輸入大小永遠長一樣
function makeInput(n) {
  let seed = n * 9301 + 49297
  return Array.from({ length: n }, () => Array.from({ length: n }, () => {
    seed = (seed * 9301 + 49297) % 233280
    return Math.floor((seed / 233280) * 10)
  }))
}

// kernel 權重依序循環 1, 0, -1，方便看出每個位置的乘加
const makeKernel = k => Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => ((i * k + j) % 3 === 0 ? 1 : (i * k + j) % 3 === 1 ? 0 : -1)))

export function mount(el) {
  injectStyle()
  const state = {
    n: clamp(num(el.dataset.n, 7), 3, 10),
    k: clamp(num(el.dataset.k, 3), 1, 5),
    s: clamp(num(el.dataset.s, 1), 1, 3),
    p: clamp(num(el.dataset.p, 0), 0, 2),
    pos: 0,
    timer: null,
  }

  el.textContent = ''
  const root = document.createElement('div')
  root.className = 'cw'
  root.setAttribute('role', 'group')
  root.setAttribute('aria-label', 'Convolution 的 kernel size、stride 與 padding 互動示範')
  root.innerHTML = `
    <div class="cw-controls"></div>
    <div class="cw-stage">
      <figure><figcaption>輸入（虛線 = padding 補的 0）</figcaption><svg class="in" role="img"></svg></figure>
      <figure><figcaption>kernel 權重</figcaption><svg class="ker" role="img"></svg></figure>
      <figure><figcaption>輸出（可點選）</figcaption><svg class="out" role="img"></svg></figure>
    </div>
    <p class="formula"></p>
    <div class="actions"><button type="button" class="prev">上一步</button><button type="button" class="play">播放</button><button type="button" class="next">下一步</button></div>
    <p class="hint">點選或用 Tab 與 Enter 選擇輸出格，會標出它在輸入中看到的範圍（receptive field）。</p>`
  el.appendChild(root)

  const controls = root.querySelector('.cw-controls')
  const defs = [['n', '輸入大小 N', 3, 10], ['k', 'kernel 大小 k', 1, 5], ['s', 'stride s', 1, 3], ['p', 'padding p', 0, 2]]
  for (const [key, label, lo, hi] of defs) {
    const row = document.createElement('label')
    row.innerHTML = `<span>${label}</span><input type="range" min="${lo}" max="${hi}" step="1" value="${state[key]}"><output>${state[key]}</output>`
    const input = row.querySelector('input')
    const out = row.querySelector('output')
    input.addEventListener('input', () => {
      state[key] = Number(input.value)
      out.textContent = input.value
      state.pos = 0
      render()
    })
    controls.appendChild(row)
  }

  const svgIn = root.querySelector('svg.in')
  const svgKer = root.querySelector('svg.ker')
  const svgOut = root.querySelector('svg.out')
  const formula = root.querySelector('.formula')
  const playBtn = root.querySelector('.play')

  function stop() {
    if (state.timer) {
      clearInterval(state.timer)
      state.timer = null
      playBtn.textContent = '播放'
    }
  }

  function render() {
    const { n, k, s, p } = state
    const np = n + 2 * p
    const o = Math.floor((np - k) / s) + 1
    const x = makeInput(n)
    const w = makeKernel(k)
    svgIn.textContent = svgKer.textContent = svgOut.textContent = ''

    if (o < 1) {
      stop()
      formula.innerHTML = `<span class="warn">kernel（${k}）比加了 padding 的輸入（${np}）還大，沒有任何輸出。</span>`
      for (const svg of [svgIn, svgKer, svgOut]) {
        svg.setAttribute('viewBox', '0 0 1 1')
        svg.setAttribute('width', 1)
        svg.setAttribute('height', 1)
      }
      return
    }
    state.pos = clamp(state.pos, 0, o * o - 1)
    const orow = Math.floor(state.pos / o)
    const ocol = state.pos % o
    const r0 = orow * s
    const c0 = ocol * s

    // 輸入（含 padding）
    svgIn.setAttribute('viewBox', `0 0 ${np * CELL} ${np * CELL}`)
    svgIn.setAttribute('width', np * CELL)
    svgIn.setAttribute('height', np * CELL)
    svgIn.setAttribute('aria-label', `${n}×${n} 的輸入，上下左右各補 ${p} 圈 0`)
    for (let r = 0; r < np; r++) {
      for (let c = 0; c < np; c++) {
        const isPad = r < p || c < p || r >= n + p || c >= n + p
        const inWin = r >= r0 && r < r0 + k && c >= c0 && c < c0 + k
        node('rect', { x: c * CELL, y: r * CELL, width: CELL, height: CELL, class: `cell${isPad ? ' pad' : ''}${inWin ? ' hit' : ''}` }, svgIn)
        const t = node('text', { x: c * CELL + CELL / 2, y: r * CELL + CELL / 2, class: isPad ? 'padtext' : '' }, svgIn)
        t.textContent = isPad ? '0' : x[r - p][c - p]
      }
    }
    node('rect', { x: c0 * CELL, y: r0 * CELL, width: k * CELL, height: k * CELL, class: 'win' }, svgIn)

    // kernel
    svgKer.setAttribute('viewBox', `0 0 ${k * CELL} ${k * CELL}`)
    svgKer.setAttribute('width', k * CELL)
    svgKer.setAttribute('height', k * CELL)
    svgKer.setAttribute('aria-label', `${k}×${k} 的 kernel`)
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        node('rect', { x: c * CELL, y: r * CELL, width: CELL, height: CELL, class: 'cell hit' }, svgKer)
        const t = node('text', { x: c * CELL + CELL / 2, y: r * CELL + CELL / 2 }, svgKer)
        t.textContent = w[r][c]
      }
    }

    // 輸出
    svgOut.setAttribute('viewBox', `0 0 ${o * CELL} ${o * CELL}`)
    svgOut.setAttribute('width', o * CELL)
    svgOut.setAttribute('height', o * CELL)
    svgOut.setAttribute('aria-label', `${o}×${o} 的輸出`)
    let selValue = 0
    for (let r = 0; r < o; r++) {
      for (let c = 0; c < o; c++) {
        let v = 0
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < k; j++) {
            const rr = r * s + i
            const cc = c * s + j
            const isPad = rr < p || cc < p || rr >= n + p || cc >= n + p
            v += (isPad ? 0 : x[rr - p][cc - p]) * w[i][j]
          }
        }
        const idx = r * o + c
        if (idx === state.pos) {
          selValue = v
        }
        const g = node('g', { class: 'out-cell', tabindex: 0, role: 'button', 'aria-label': `輸出第 ${r + 1} 列第 ${c + 1} 行，值 ${v}` }, svgOut)
        node('rect', { x: c * CELL, y: r * CELL, width: CELL, height: CELL, class: `cell${idx === state.pos ? ' sel' : ''}` }, g)
        node('text', { x: c * CELL + CELL / 2, y: r * CELL + CELL / 2 }, g).textContent = v
        const pick = () => {
          stop()
          state.pos = idx
          render()
        }
        g.addEventListener('pointerenter', (e) => {
          if (e.pointerType === 'mouse') {
            pick()
          }
        })
        g.addEventListener('click', pick)
        g.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            pick()
          }
        })
      }
    }

    formula.innerHTML = `輸出大小 O = ⌊(N + 2p − k) / s⌋ + 1 = ⌊(${n} + 2×${p} − ${k}) / ${s}⌋ + 1 = <strong>${o}</strong>　｜　輸出格（${orow + 1}, ${ocol + 1}）= <strong>${selValue}</strong>`
    root.dataset.outSize = o
  }

  const total = () => {
    const o = Math.floor((state.n + 2 * state.p - state.k) / state.s) + 1
    return Math.max(o * o, 1)
  }
  const go = (d) => {
    state.pos = (state.pos + d + total()) % total()
    render()
  }
  root.querySelector('.prev').addEventListener('click', () => {
    stop()
    go(-1)
  })
  root.querySelector('.next').addEventListener('click', () => {
    stop()
    go(1)
  })
  playBtn.addEventListener('click', () => {
    if (state.timer) {
      stop()
      return
    }
    playBtn.textContent = '暫停'
    state.timer = setInterval(() => go(1), 700)
  })

  render()
}
