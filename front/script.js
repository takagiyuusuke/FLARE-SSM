// ========== 定数定義 ==========
const wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'];

// ========== 波長ごとのカラーマップ定義 ==========
const colormaps = {
  '0094': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 79, 51] },
    { pos: 0.4, color: [41, 121, 102] },
    { pos: 0.6, color: [92, 162, 153] },
    { pos: 0.8, color: [163, 206, 204] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0131': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [0, 73, 73] },
    { pos: 0.4, color: [0, 147, 147] },
    { pos: 0.6, color: [62, 221, 221] },
    { pos: 0.8, color: [158, 255, 255] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0171': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 51, 0] },
    { pos: 0.4, color: [147, 102, 0] },
    { pos: 0.6, color: [221, 153, 0] },
    { pos: 0.8, color: [255, 204, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0193': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 10] },
    { pos: 0.4, color: [161, 102, 41] },
    { pos: 0.6, color: [197, 153, 92] },
    { pos: 0.8, color: [228, 204, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0211': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 79] },
    { pos: 0.4, color: [161, 102, 121] },
    { pos: 0.6, color: [197, 153, 162] },
    { pos: 0.8, color: [228, 204, 206] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0304': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 0, 0] },
    { pos: 0.4, color: [147, 0, 0] },
    { pos: 0.6, color: [221, 62, 0] },
    { pos: 0.8, color: [255, 158, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0335': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 51, 114] },
    { pos: 0.4, color: [41, 102, 161] },
    { pos: 0.6, color: [92, 153, 197] },
    { pos: 0.8, color: [163, 204, 228] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '1600': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [79, 79, 10] },
    { pos: 0.4, color: [121, 121, 41] },
    { pos: 0.6, color: [162, 162, 92] },
    { pos: 0.8, color: [206, 206, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '4500': [
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
    const gray = data[i];
    const norm = gray / 255;
    const [r, g, b] = getAIAColorForWavelength(norm, wavelength);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

// ========== 初期化処理 ==========

// flatpickr のインスタンス（カレンダー部分：日付のみ）
let fp = null;

// 予め全体で利用する変数を定義しておく
let preloadedImages = {};  // 全画像キャッシュ
let timestamps = [];
let imageElements = {};    // 各波長のimgタグ
let frameIndex = 0;
let animationTimer = null;

// DOMContentLoaded時の初期処理
window.addEventListener('DOMContentLoaded', () => {
  // 現在の UTC 時刻（分・秒は 00 固定）
  const now = new Date();
  const utcNow = new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
    now.getUTCHours(), 0, 0
  ));

  // pred.json から利用可能な日付を取得し、それをもとにカレンダーを初期化
  fetch('data/pred.json')
    .then(response => response.json())
    .then(predData => {
      // キーから日付部分 ("YYYYMMDD") を抽出し、ユニークな日付リストを作成
      const availableDates = new Set(
        Object.keys(predData).map(key => key.slice(0, 8)) // "YYYYMMDDHH" → "YYYYMMDD"
      );

      // flatpickr の初期化
      fp = flatpickr("#date-picker", {
        inline: true,
        enableTime: false,
        dateFormat: "Y-m-d",
        defaultDate: utcNow.toISOString().slice(0, 10), // 今日の日付 "YYYY-MM-DD"
        maxDate: "today",
        minDate: "2025-01-01", // 必要に応じて変更

        // 有効な日付だけ選択可能にする
        enable: [
          function(date) {
            // 日付を "YYYYMMDD" 形式に変換して存在確認
            const y = date.getFullYear();
            const m = String(date.getMonth() + 1).padStart(2, '0');
            const d = String(date.getDate()).padStart(2, '0');
            return availableDates.has(`${y}${m}${d}`);
          }
        ],
        // カレンダーが描画された後に実行
        onReady: () => {
          // onReady 時に一度だけ実行
          loadImagesFromSelectedTime();
        },
        // ユーザーが日付を変更したとき
        onChange: (selectedDates) => {
          if (selectedDates.length > 0) {
            // 新たに日付をクリックしたら再読み込み
            loadImagesFromSelectedTime();
          }
        }
      });
    })
    .catch(err => {
      console.error("pred.json の取得中にエラー:", err);
      // 取得失敗時でも最低限のflatpickrを起動しておく（全日付有効にするなど）
      fp = flatpickr("#date-picker", {
        inline: true,
        enableTime: false,
        dateFormat: "Y-m-d",
        defaultDate: utcNow.toISOString().slice(0, 10),
        maxDate: "today",
        minDate: "2025-01-01" // 必要に応じて変更
      });
    });

  // 時刻選択用の <select>（#utc-hour） を初期化（偶数時刻のみ）
  const hourSelect = document.getElementById("utc-hour");
  hourSelect.innerHTML = "";
  for (let h = 0; h < 24; h += 2) {
    const option = document.createElement("option");
    option.value = h;
    option.textContent = String(h).padStart(2, "0") + ":00";
    hourSelect.appendChild(option);
  }
  // 現在のUTC時刻の時刻を偶数に丸めた値を初期値に設定
  let initHour = utcNow.getUTCHours();
  initHour = Math.floor(initHour / 2) * 2;
  hourSelect.value = initHour;

  // 「この時刻で読み込む」ボタンをクリックで再読み込み
  document.getElementById('load-button').addEventListener('click', () => {
    loadImagesFromSelectedTime();
  });
});

// ========== ロジック本体 ==========

function loadXRSData(baseTime) {
  const xrsDataURL = 'data/xrs.json';
  return fetch(xrsDataURL)
    .then(res => res.json())
    .then(data => {
      const xrsData = [];
      // -24 ~ +71までで計96個
      for (let i = -24; i < 72; i++) {
        const targetTime = new Date(baseTime.getTime() + i * 3600 * 1000);
        const key = `${targetTime.getUTCFullYear()}${String(targetTime.getUTCMonth() + 1).padStart(2, '0')}${String(targetTime.getUTCDate()).padStart(2, '0')}${String(targetTime.getUTCHours()).padStart(2, '0')}`;
        xrsData.push(data[key] !== undefined ? data[key] : null);
      }
      return xrsData;
    })
    .catch(err => {
      console.error("XRSデータ取得中にエラー:", err);
      return Array(96).fill(null);
    });
}

function loadImagesFromSelectedTime() {
  // flatpickrインスタンスが無い or まだ日付が選択されていない場合は終了
  if (!fp) {
    console.warn("Flatpickrが初期化されていません。");
    return;
  }
  const selectedDate = fp.selectedDates[0];
  if (!selectedDate) {
    console.error("日付が選択されていません");
    return;
  }

  // アニメーションタイマーがあれば停止
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = null;
  }

  // 選択された日付(UTC)を取得
  const year = selectedDate.getUTCFullYear();
  const month = selectedDate.getUTCMonth();
  const day = selectedDate.getUTCDate();

  // 時刻は <select id="utc-hour"> の値
  const hour = parseInt(document.getElementById("utc-hour").value, 10);
  const baseTime = new Date(Date.UTC(year, month, day, hour, 0, 0));

  // 1時間ごとに計12枚生成 （-22h ～ 0hを2h刻み → 12枚）
  // ※元コードでは「-22h, -20h, ... 0h」で 12ステップ分のタイムスタンプを作成
  timestamps = [];
  for (let h = 22; h >= 0; h -= 2) {
    const t = new Date(baseTime.getTime() - h * 3600 * 1000);
    timestamps.push(t);
  }

  // AIA/HMI 画像のURLを生成
  const aiaUrls = {};
  wavelengths.forEach(wl => {
    aiaUrls[wl] = timestamps.map(t => {
      const m = String(t.getUTCMonth() + 1).padStart(2, '0');
      const d = String(t.getUTCDate()).padStart(2, '0');
      const h = String(t.getUTCHours()).padStart(2, '0');
      return `data/images/${m}${d}/${h}_aia_${wl}.png`;
    });
  });
  const hmiUrls = timestamps.map(t => {
    const m = String(t.getUTCMonth() + 1).padStart(2, '0');
    const d = String(t.getUTCDate()).padStart(2, '0');
    const h = String(t.getUTCHours()).padStart(2, '0');
    return `data/images/${m}${d}/${h}_hmi.png`;
  });

  // 画像のプリロード用オブジェクトをクリア
  preloadedImages = {};
  const transparentURL = createTransparentImageURL();

  // 各波長のAIA画像を読み込み＆カラーマップ変換
  wavelengths.forEach(wl => {
    aiaUrls[wl].forEach((url, i) => {
      const key = `${wl}-${i}`;
      const img = new Image();
      img.onload = () => {
        const coloredDataURL = applyAIAColormap(img, wl);
        const coloredImg = new Image();
        coloredImg.onload = () => {
          preloadedImages[key] = coloredImg;
        };
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

  // HMI画像の読み込み
  hmiUrls.forEach((url, i) => {
    const key = `HMI-${i}`;
    const img = new Image();
    img.onload = () => {
      preloadedImages[key] = img;
    };
    img.onerror = () => {
      const fallback = new Image();
      fallback.src = transparentURL;
      preloadedImages[key] = fallback;
      console.warn(`❌ HMI image failed to load: ${url}`);
    };
    img.src = url;
  });

  // 画像表示を開始
  renderImages();

  // pred.json を取得してカードを表示
  const tY = baseTime.getUTCFullYear();
  const tM = String(baseTime.getUTCMonth() + 1).padStart(2, '0');
  const tD = String(baseTime.getUTCDate()).padStart(2, '0');
  const tH = String(baseTime.getUTCHours()).padStart(2, '0');
  const selectedTimeKey = `${tY}${tM}${tD}${tH}`;

  fetch('data/pred.json')
    .then(res => res.json())
    .then(predData => {
      const probabilities = predData[selectedTimeKey];
      const cardContainer = document.getElementById('prediction-cards');
      cardContainer.innerHTML = ''; // 既存のカードをクリア

      const classLabels = ['O class', 'C class', 'M class', 'X class'];

      if (probabilities) {
        const maxProbability = Math.max(...probabilities); // 最大の確率を取得

        probabilities.forEach((prob, index) => {
          const card = document.createElement('div');
          card.className = 'prediction-card';
          if (prob === maxProbability) {
            card.classList.add('highlight'); // 最大の確率を持つカードを強調
          }
          card.innerHTML = `
            <h3><span class="class-label">${classLabels[index][0]}</span><span class="class-small">${classLabels[index].slice(1)}</span></h3>
            <p>Probability: ${(prob * 100).toFixed(2)}%</p>
          `;
          cardContainer.appendChild(card);
        });
      } else {
        console.warn(`❌ No prediction data found for key: ${selectedTimeKey}`);
        // エントリが存在しない場合、??% のカードを表示
        for (let i = 0; i < 4; i++) {
          const card = document.createElement('div');
          card.className = 'prediction-card';
          card.innerHTML = `
            <h3><span class="class-label">${classLabels[i][0]}</span><span class="class-small">${classLabels[i].slice(1)}</span></h3>
            <p>Probability: ??%</p>
          `;
          cardContainer.appendChild(card);
        }
        // エラーメッセージを表示
        const errorMessage = document.createElement('p');
        errorMessage.className = 'error-message';
        errorMessage.textContent = "❌ 推論データが存在しません";
        cardContainer.appendChild(errorMessage);
      }
    })
    .catch(err => {
      console.error("Prediction data fetch error:", err);
      const cardContainer = document.getElementById('prediction-cards');
      cardContainer.innerHTML = ''; // 既存のカードをクリア
      // エラー時も ??% のカードを表示
      const classLabels = ['O class', 'C class', 'M class', 'X class'];
      for (let i = 0; i < 4; i++) {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.innerHTML = `
          <h3><span class="class-label">${classLabels[i][0]}</span><span class="class-small">${classLabels[i].slice(1)}</span></h3>
          <p>Probability: ??%</p>
        `;
        cardContainer.appendChild(card);
      }
      // エラーメッセージを表示
      const errorMessage = document.createElement('p');
      errorMessage.className = 'error-message';
      errorMessage.textContent = "❌ 推論データの取得中にエラーが発生しました";
      cardContainer.appendChild(errorMessage);
    });

  // XRSデータをロードしてチャートを更新
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
      // すでにチャートがあればデータだけ更新
      window.flareChartInstance.data.labels = labels;
      window.flareChartInstance.data.datasets[0].data = flareData;
      window.flareChartInstance.data.datasets[0].pointBackgroundColor = pointColors;

      const lastTimestamp = timestamps[timestamps.length - 1];
      const formattedTime = `${lastTimestamp.getUTCFullYear()}-${String(lastTimestamp.getUTCMonth() + 1).padStart(2, '0')}-${String(lastTimestamp.getUTCDate()).padStart(2, '0')} ${String(lastTimestamp.getUTCHours()).padStart(2, '0')}:00 UTC`;
      window.flareChartInstance.options.plugins.annotation.annotations.zeroHourLine.label.content = formattedTime;

      window.flareChartInstance.update();
    } else {
      // 初回の描画
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
          // 背景色を帯状に塗るデモ用プラグイン
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

// 透明画像（あるいは「No data」と表示される画像）を生成するヘルパー
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

// 画像をループ再生で描画する
function renderImages() {
  const grid = document.getElementById('aia-grid');
  grid.innerHTML = '';
  imageElements = {};

  // AIA 9波長 + HMI で合計10枚を並べる
  [...wavelengths, 'HMI'].forEach(type => {
    const container = document.createElement('div');
    container.className = 'channel';

    const label = document.createElement('div');
    label.textContent = (type === 'HMI') ? 'HMI' : `AIA ${parseInt(type, 10)}Å`;
    const img = document.createElement('img');
    img.id = `img-${type}`;
    container.appendChild(label);
    container.appendChild(img);
    grid.appendChild(container);

    imageElements[type] = img;
  });

  frameIndex = 0;
  const timestampLabel = document.getElementById('timestamp');

  // 0.4秒ごとにフレーム更新
  animationTimer = setInterval(() => {
    // 各波長の画像を切り替え
    wavelengths.forEach(wl => {
      const key = `${wl}-${frameIndex % timestamps.length}`;
      if (preloadedImages[key]) {
        imageElements[wl].src = preloadedImages[key].src;
      }
    });
    // HMIも切り替え
    const hmiKey = `HMI-${frameIndex % timestamps.length}`;
    if (preloadedImages[hmiKey]) {
      imageElements['HMI'].src = preloadedImages[hmiKey].src;
    }

    // 時刻を表示
    const t = timestamps[frameIndex % timestamps.length];
    const timeStr = `${t.getUTCFullYear()}-${String(t.getUTCMonth() + 1).padStart(2, '0')}-${String(t.getUTCDate()).padStart(2, '0')} ${String(t.getUTCHours()).padStart(2, '0')}:00 UTC`;
    timestampLabel.textContent = `現在表示中の時刻: ${timeStr}`;

    // グラフ上の点も合わせてハイライト
    if (window.flareChartInstance) {
      const dataset = window.flareChartInstance.data.datasets[0];
      // 点の大きさをいったん全て2に戻す
      dataset.pointRadius = Array(96).fill(2);

      // タイムスタンプの最後(=基準時)との差 (hourOffset) を計算
      const lastT = timestamps[timestamps.length - 1];
      const hourOffset = Math.floor((t - lastT) / (3600 * 1000)); // 何時間離れているか

      // -24 <= hourOffset < 72 の範囲であれば、その点を大きくする
      if (hourOffset >= -24 && hourOffset < 72) {
        const graphIndex = hourOffset + 24; // 配列上のインデックス
        dataset.pointRadius[graphIndex] = 6;  // ハイライト
      }
      window.flareChartInstance.update('none');
    }

    frameIndex++;
  }, 400);
}
