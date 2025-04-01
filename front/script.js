// ========== 定数定義 ==========
const wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'];

// ========== 波長ごとのカラーマップ定義 ==========
const colormaps = {
  '0094': [  // 94 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 79, 51] },
    { pos: 0.4, color: [41, 121, 102] },
    { pos: 0.6, color: [92, 162, 153] },
    { pos: 0.8, color: [163, 206, 204] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0131': [  // 131 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [0, 73, 73] },
    { pos: 0.4, color: [0, 147, 147] },
    { pos: 0.6, color: [62, 221, 221] },
    { pos: 0.8, color: [158, 255, 255] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0171': [  // 171 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 51, 0] },
    { pos: 0.4, color: [147, 102, 0] },
    { pos: 0.6, color: [221, 153, 0] },
    { pos: 0.8, color: [255, 204, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0193': [  // 193 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 10] },
    { pos: 0.4, color: [161, 102, 41] },
    { pos: 0.6, color: [197, 153, 92] },
    { pos: 0.8, color: [228, 204, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0211': [  // 211 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 79] },
    { pos: 0.4, color: [161, 102, 121] },
    { pos: 0.6, color: [197, 153, 162] },
    { pos: 0.8, color: [228, 204, 206] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0304': [  // 304 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 0, 0] },
    { pos: 0.4, color: [147, 0, 0] },
    { pos: 0.6, color: [221, 62, 0] },
    { pos: 0.8, color: [255, 158, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0335': [  // 335 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 51, 114] },
    { pos: 0.4, color: [41, 102, 161] },
    { pos: 0.6, color: [92, 153, 197] },
    { pos: 0.8, color: [163, 204, 228] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '1600': [  // 1600 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [79, 79, 10] },
    { pos: 0.4, color: [121, 121, 41] },
    { pos: 0.6, color: [162, 162, 92] },
    { pos: 0.8, color: [206, 206, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '4500': [  // 4500 Ångström
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [51, 51, 0] },
    { pos: 0.4, color: [102, 102, 0] },
    { pos: 0.6, color: [153, 153, 0] },
    { pos: 0.8, color: [204, 204, 27] },
    { pos: 1.0, color: [255, 255, 128] }
  ]
};

// ========== カラーマップ補間関数 ==========
function getAIAColorForWavelength(normValue, wavelength) {
  const stops = colormaps[wavelength];
  for (let i = 0; i < stops.length - 1; i++) {
    if (normValue >= stops[i].pos && normValue <= stops[i+1].pos) {
      const range = stops[i+1].pos - stops[i].pos;
      const factor = (normValue - stops[i].pos) / range;
      const r = Math.round(stops[i].color[0] + factor * (stops[i+1].color[0] - stops[i].color[0]));
      const g = Math.round(stops[i].color[1] + factor * (stops[i+1].color[1] - stops[i].color[1]));
      const b = Math.round(stops[i].color[2] + factor * (stops[i+1].color[2] - stops[i].color[2]));
      return [r, g, b];
    }
  }
  return [255, 255, 255];
}

// AIA画像にカラーマップを適用する関数（波長指定付き）
function applyAIAColormap(image, wavelength) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    // グレースケール画像の場合、R=G=Bなのでいずれかで正規化
    const gray = data[i];
    const norm = gray / 255;
    const [r, g, b] = getAIAColorForWavelength(norm, wavelength);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    // data[i+3] はアルファ値
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

// ========== 初期化処理 ==========
window.addEventListener('DOMContentLoaded', () => {
  populateTimeSelectors();
  loadImagesFromSelectedTime();

  document.getElementById('load-button').addEventListener('click', () => {
    loadImagesFromSelectedTime();
  });
});

// ========== セレクタの中身を生成 ==========
function populateTimeSelectors() {
  const now = new Date();
  now.setUTCMinutes(0, 0, 0);

  // 年
  const yearSelect = document.getElementById('year');
  for (let y = 2011; y <= now.getUTCFullYear(); y++) {
    yearSelect.innerHTML += `<option value="${y}">${y}</option>`;
  }
  yearSelect.value = now.getUTCFullYear();

  // 月
  const monthSelect = document.getElementById('month');
  for (let m = 1; m <= 12; m++) {
    monthSelect.innerHTML += `<option value="${m}">${m}</option>`;
  }
  monthSelect.value = now.getUTCMonth() + 1;

  // 日
  updateDaySelector();
  const daySelect = document.getElementById('day');
  daySelect.value = now.getUTCDate();
  document.getElementById('year').addEventListener('change', () => {
    updateDaySelector();
    updateHourSelector();
  });
  document.getElementById('month').addEventListener('change', () => {
    updateDaySelector();
    updateHourSelector();
  });
  document.getElementById('day').addEventListener('change', updateHourSelector);

  // 時間
  updateHourSelector();
}

function updateDaySelector() {
  const year = parseInt(document.getElementById('year').value);
  const month = parseInt(document.getElementById('month').value);
  const daySelect = document.getElementById('day');

  const daysInMonth = new Date(year, month, 0).getDate();
  const prevDay = parseInt(daySelect.value) || 1;
  daySelect.innerHTML = '';
  for (let d = 1; d <= daysInMonth; d++) {
    daySelect.innerHTML += `<option value="${d}">${d}</option>`;
  }
  daySelect.value = Math.min(prevDay, daysInMonth);
}

function updateHourSelector() {
  const now = new Date();
  now.setUTCMinutes(0, 0, 0);
  const maxHour = now.getUTCHours() - ((now.getUTCMinutes() >= 30) ? 0 : 1);

  const selectedYear = parseInt(document.getElementById('year').value);
  const selectedMonth = parseInt(document.getElementById('month').value);
  const selectedDay = parseInt(document.getElementById('day').value);

  const isToday = selectedYear === now.getUTCFullYear() &&
                  selectedMonth === now.getUTCMonth() + 1 &&
                  selectedDay === now.getUTCDate();

  const hourSelect = document.getElementById('hour');
  const currentValue = parseInt(hourSelect.value) || 0;
  hourSelect.innerHTML = '';

  for (let h = 0; h <= 23; h++) {
    const disabled = (isToday && h > maxHour) ? 'disabled' : '';
    hourSelect.innerHTML += `<option value="${h}" ${disabled}>${h}</option>`;
  }
  hourSelect.value = isToday ? maxHour : Math.min(currentValue, 23);
}

// ========== ロジック本体 ==========
let preloadedImages = {};  // 全画像キャッシュ
let timestamps = [];
let imageElements = {};  // 各波長のimgタグ
let frameIndex = 0;
let animationTimer = null;

function loadXRSData(baseTime) {
  const xrsDataURL = 'data/xrs.json';
  return fetch(xrsDataURL)
    .then(res => res.json())
    .then(data => {
      const xrsData = [];
      for (let i = -24; i < 72; i++) {
        const targetTime = new Date(baseTime.getTime() + i * 3600 * 1000);
        const key = `${targetTime.getUTCFullYear()}${String(targetTime.getUTCMonth() + 1).padStart(2, '0')}${String(targetTime.getUTCDate()).padStart(2, '0')}${String(targetTime.getUTCHours()).padStart(2, '0')}`;
        xrsData.push(data[key] !== undefined ? data[key] : null);
      }
      return xrsData;
    })
    .catch(err => {
      console.error("XRSデータ取得中にエラー:", err);
      return Array(96).fill(null); // エラー時はすべてnull
    });
}

function loadImagesFromSelectedTime() {
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = null;
  }

  const year = parseInt(document.getElementById('year').value);
  const month = parseInt(document.getElementById('month').value);
  const day = parseInt(document.getElementById('day').value);
  const hour = parseInt(document.getElementById('hour').value);

  const baseTime = new Date(Date.UTC(year, month - 1, day, hour));

  // 1時間ごと11枚生成（-22h ～ 0h → 12枚）
  timestamps = [];
  for (let h = 22; h >= 0; h -= 2) {
    const t = new Date(baseTime.getTime() - h * 3600 * 1000);
    timestamps.push(t);
  }
  // console.log("Generated timestamps:", timestamps.map(t => t.toISOString()));

  // URL生成
  const aiaUrls = {};
  const hmiUrls = [];
  wavelengths.forEach(wl => {
    aiaUrls[wl] = timestamps.map(t => {
      const m = String(t.getUTCMonth() + 1).padStart(2, '0');
      const d = String(t.getUTCDate()).padStart(2, '0');
      const h = String(t.getUTCHours()).padStart(2, '0');
      return `data/images/${m}${d}/${h}_aia_${wl}.png`;
    });
  });
  hmiUrls.push(...timestamps.map(t => {
    const m = String(t.getUTCMonth() + 1).padStart(2, '0');
    const d = String(t.getUTCDate()).padStart(2, '0');
    const h = String(t.getUTCHours()).padStart(2, '0');
    return `data/images/${m}${d}/${h}_hmi.png`;
  }));

  // 画像キャッシュ初期化
  preloadedImages = {};
  const transparentURL = createTransparentImageURL();

  // AIA画像の読み込み＆波長ごとのカラーマップ変換
  wavelengths.forEach(wl => {
    aiaUrls[wl].forEach((url, i) => {
      const key = `${wl}-${i}`;
      const img = new Image();
      img.onload = () => {
        const coloredDataURL = applyAIAColormap(img, wl);
        const coloredImg = new Image();
        coloredImg.onload = () => { preloadedImages[key] = coloredImg; };
        coloredImg.src = coloredDataURL;
      };
      img.onerror = () => {
        const fallback = new Image();
        fallback.src = transparentURL;
        preloadedImages[key] = fallback;
        console.warn(`❌ AIA image failed to load: ${url}`);
      };
      img.src = url;
    });
  });

  // HMI画像はそのまま読み込み
  hmiUrls.forEach((url, i) => {
    const key = `HMI-${i}`;
    const img = new Image();
    img.onload = () => { preloadedImages[key] = img; };
    img.onerror = () => {
      const fallback = new Image();
      fallback.src = transparentURL;
      preloadedImages[key] = fallback;
      console.warn(`❌ HMI image failed to load: ${url}`);
    };
    img.src = url;
  });

  renderImages();

  // フレアデータ取得＆チャート描画処理（以下は元コードと同様）
  const tY = baseTime.getUTCFullYear();
  const tM = String(baseTime.getUTCMonth() + 1).padStart(2, '0');
  const tD = String(baseTime.getUTCDate()).padStart(2, '0');
  const tH = String(baseTime.getUTCHours()).padStart(2, '0');
  const flareTimeStr = `${tY}${tM}${tD}${tH}`;

  loadXRSData(baseTime).then(flareData => {
    const labels = Array.from({ length: 96 }, (_, i) => `${i > 24 ? '+' : ''}${i - 24}h`);
    const ctx = document.getElementById('flareChart').getContext('2d');
    const pointColors = flareData.map(value => {
      if (value == null) return 'gray';
      if (value < 1e-6) return 'blue';
      if (value < 1e-5) return 'green';
      if (value < 1e-4) return 'orange';
      return 'red';
    });

    if (window.flareChartInstance) {
      window.flareChartInstance.data.labels = labels;
      window.flareChartInstance.data.datasets[0].data = flareData;
      window.flareChartInstance.data.datasets[0].pointBackgroundColor = pointColors;

      const lastTimestamp = timestamps[timestamps.length - 1];
      const formattedTime = `${lastTimestamp.getUTCFullYear()}-${String(lastTimestamp.getUTCMonth() + 1).padStart(2, '0')}-${String(lastTimestamp.getUTCDate()).padStart(2, '0')} ${String(lastTimestamp.getUTCHours()).padStart(2, '0')}:00 UTC`;
      window.flareChartInstance.options.plugins.annotation.annotations.zeroHourLine.label.content = formattedTime;

      window.flareChartInstance.update();
    } else {
        window.flareChartInstance = new Chart(ctx, {
          type: 'line',
          data: {
            labels: labels,
            datasets: [{
              label: 'X-ray Flux (0.1–0.8 nm)',
              data: flareData,
              borderColor: 'black',
              pointBackgroundColor: pointColors,
              fill: false
            }]
          },
          options: {
            scales: {
              y: {
                type: 'logarithmic',
                min: 1e-9,
                max: 1e-3,
                title: { display: true, text: 'Flux (W/m²)' }
              }
            },
            plugins: {
              tooltip: {
                callbacks: {
                  label: (ctx) => {
                    const v = ctx.raw;
                    if (v == null) return '欠損';
                    const cls = v >= 1e-4 ? 'X' :
                                v >= 1e-5 ? 'M' :
                                v >= 1e-6 ? 'C' : 'O';
                    return `Flux: ${v} W/m² (Class ${cls})`;
                  }
                }
              },
              annotation: {
                annotations: {
                  flareBands: {
                    type: 'box',
                    yMin: 1e-4,
                    yMax: 1e-3,
                    backgroundColor: 'rgba(255,0,0,0.05)',
                    label: {
                      enabled: true,
                      content: 'X',
                      position: 'start',
                      xAdjust: 50,
                      backgroundColor: 'transparent',
                      color: 'red',
                      font: { weight: 'bold', size: 14 }
                    }
                  },
                  flareBandM: {
                    type: 'box',
                    yMin: 1e-5,
                    yMax: 1e-4,
                    backgroundColor: 'rgba(255,165,0,0.05)',
                    label: {
                      enabled: true,
                      content: 'M',
                      position: 'start',
                      xAdjust: 50,
                      backgroundColor: 'transparent',
                      color: 'orange',
                      font: { weight: 'bold', size: 14 }
                    }
                  },
                  flareBandC: {
                    type: 'box',
                    yMin: 1e-6,
                    yMax: 1e-5,
                    backgroundColor: 'rgba(0,255,0,0.05)',
                    label: {
                      enabled: true,
                      content: 'C',
                      position: 'start',
                      xAdjust: 50,
                      backgroundColor: 'transparent',
                      color: 'green',
                      font: { weight: 'bold', size: 14 }
                    }
                  },
                  flareBandO: {
                    type: 'box',
                    yMin: 1e-9,
                    yMax: 1e-6,
                    backgroundColor: 'rgba(0,0,255,0.05)',
                    label: {
                      enabled: true,
                      content: 'O',
                      position: 'start',
                      xAdjust: 50,
                      backgroundColor: 'transparent',
                      color: 'blue',
                      font: { weight: 'bold', size: 14 }
                    }
                  },
                  zeroHourLine: {
                    type: 'line',
                    scaleID: 'x',
                    value: 24,
                    borderColor: 'black',
                    borderWidth: 3,
                    label: {
                      enabled: true,
                      content: timestamps.length > 0 
                        ? `${timestamps[timestamps.length - 1].getUTCFullYear()}-${String(timestamps[timestamps.length - 1].getUTCMonth() + 1).padStart(2, '0')}-${String(timestamps[timestamps.length - 1].getUTCDate()).padStart(2, '0')} ${String(timestamps[timestamps.length - 1].getUTCHours()).padStart(2, '0')}:00 UTC`
                        : '0h',
                      position: 'end',
                      backgroundColor: 'black',
                      color: 'white',
                      font: { weight: 'bold', size: 12 }
                    }
                  }
                }
              }
            }
          },
          plugins: [{
            id: 'backgroundZones',
            beforeDraw: (chart) => {
              const { ctx, chartArea, scales } = chart;
              const zones = [
                { from: 1e-4, to: 1e-3, color: 'rgba(255,0,0,0.15)' },
                { from: 1e-5, to: 1e-4, color: 'rgba(255,165,0,0.15)' },
                { from: 1e-6, to: 1e-5, color: 'rgba(0,255,0,0.15)' },
                { from: 1e-9, to: 1e-6, color: 'rgba(0,0,255,0.15)' }
              ];
        
              zones.forEach(zone => {
                const y1 = scales.y.getPixelForValue(zone.from);
                const y2 = scales.y.getPixelForValue(zone.to);
                ctx.fillStyle = zone.color;
                ctx.fillRect(chartArea.left, y2, chartArea.right - chartArea.left, y1 - y2);
              });
            }
          }]
        });
      }
    })
    .catch(err => {
      console.error("フレアデータ取得中にエラー:", err);
    });
}

function createTransparentImageURL(width = 200, height = 200) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.font = "20px sans-serif";
  ctx.fillStyle = "rgba(255, 0, 0, 0.7)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("No data", width / 2, height / 2);
  return canvas.toDataURL('image/png');
}

function renderImages() {
  const grid = document.getElementById('aia-grid');
  grid.innerHTML = '';
  imageElements = {};

  [...wavelengths, 'HMI'].forEach(type => {
    const container = document.createElement('div');
    container.className = 'channel';

    const label = document.createElement('div');
    label.textContent = type === 'HMI' ? 'HMI' : `AIA ${type}`;
    const img = document.createElement('img');
    img.id = `img-${type}`;
    container.appendChild(label);
    container.appendChild(img);
    grid.appendChild(container);

    imageElements[type] = img;
  });

  frameIndex = 0;
  const timestampLabel = document.getElementById('timestamp');
  animationTimer = setInterval(() => {
    wavelengths.forEach(wl => {
      const key = `${wl}-${frameIndex % timestamps.length}`;
      if (preloadedImages[key]) imageElements[wl].src = preloadedImages[key].src;
    });

    const hmiKey = `HMI-${frameIndex % timestamps.length}`;
    if (preloadedImages[hmiKey]) imageElements['HMI'].src = preloadedImages[hmiKey].src;

    const t = timestamps[frameIndex % timestamps.length];
    const timeStr = `${t.getUTCFullYear()}-${String(t.getUTCMonth() + 1).padStart(2, '0')}-${String(t.getUTCDate()).padStart(2, '0')} ${String(t.getUTCHours()).padStart(2, '0')}:00 UTC`;
    timestampLabel.textContent = `現在表示中の時刻: ${timeStr}`;

    if (window.flareChartInstance) {
      const dataset = window.flareChartInstance.data.datasets[0];
      dataset.pointRadius = Array(96).fill(2);
      const hourOffset = Math.floor((t - timestamps[timestamps.length - 1]) / (3600 * 1000));
      if (hourOffset >= -24 && hourOffset < 72) {
        const graphIndex = hourOffset + 24;
        dataset.pointRadius[graphIndex] = 6;
      }
    
      window.flareChartInstance.update('none');
    }

    frameIndex++;
  }, 200);
}
