import os
import requests
import numpy as np
import cv2
import json
import time
import h5py
from io import BytesIO
from datetime import datetime, timedelta
from astropy.io import fits
from matplotlib import pyplot as plt
from dateutil import tz

# ---------- 設定 ----------
AIA_WAVELENGTHS = [
    '0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'
]
SAVE_ROOT = 'data/images'
H5_SAVE_ROOT = 'ml/datasets/all_data_hours'
XRS_PATH = 'data/xrs.json'
XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def fetch_and_process_aia_image(wavelength, dt):
    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    if year >= 2023:
        url = f"https://sdo5.nascom.nasa.gov/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/AIA{ymd}_{hour}0000_{wavelength}.fits"
    else:
        url = f"https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/AIA{ymd}_{hour}00_{wavelength}.fits"

    try:
        response = requests.get(url)
        time.sleep(1)
        response.raise_for_status()
        hdul = fits.open(BytesIO(response.content))
        image_data = hdul[1].data
        if image_data.dtype == 'object':
            image_data = np.array(image_data.tolist())

        image_data = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_AREA)
        image_data = image_data[20: -20, 15: -15]  # 中央クロップ
        image_data = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_AREA)
        image_data = np.flipud(image_data)

        return image_data.astype(np.uint16)
    except Exception as e:
        print(f"❌ AIA {wavelength} の取得失敗: {e}")
        return np.zeros((256, 256), dtype=np.uint16)


def download_hmi_image(dt):
    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    url = f"https://jsoc1.stanford.edu/data/hmi/images/{year}/{month}/{day}/{ymd}_{hour}0000_M_1k.jpg"

    try:
        response = requests.get(url)
        time.sleep(1)
        response.raise_for_status()
        image_data = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        image_data[-30:, :] = 0
        image_data = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_AREA)
        return image_data.astype(np.uint16)
    except Exception as e:
        print(f"❌ HMI画像の取得失敗: {e}")
        return np.zeros((256, 256), dtype=np.uint16)


def save_png(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def save_h5(aia_images, hmi_image, dt):
    os.makedirs(H5_SAVE_ROOT, exist_ok=True)
    filename = dt.strftime("%Y%m%d_%H0000.h5")
    filepath = os.path.join(H5_SAVE_ROOT, filename)

    aia_images_fixed = []
    for img in aia_images:
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aia_images_fixed.append(img.astype(np.uint16).reshape(256, 256))

    if hmi_image.ndim == 3 and hmi_image.shape[-1] == 3:
        hmi_image = cv2.cvtColor(hmi_image, cv2.COLOR_BGR2GRAY)
    hmi_image = hmi_image.astype(np.uint16).reshape(256, 256)

    X = np.stack(aia_images_fixed + [hmi_image])  # shape: (10, 256, 256)
    timestamp = dt.strftime("%Y%m%d_%H0000").encode()

    with h5py.File(filepath, 'w') as f:
        f.create_dataset("X", data=X)
        f.create_dataset("timestamp", data=timestamp)

    print(f"✅ H5保存: {filepath}")


def update_xrs_json(dt):
    try:
        response = requests.get(XRS_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ XRS API取得失敗: {e}")
        return

    max_flux = 0
    for item in data:
        if item.get("energy") != "0.1-0.8nm":
            continue
        ts = item.get("time_tag")
        try:
            obs_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            obs_time = obs_time.replace(tzinfo=tz.tzutc())
        except:
            continue

        if dt - timedelta(hours=1) < obs_time <= dt:
            try:
                flux = float(item.get("flux", 0))
                max_flux = max(max_flux, flux)
            except:
                continue

    time_str = dt.strftime('%Y%m%d%H')
    try:
        if os.path.exists(XRS_PATH):
            with open(XRS_PATH, 'r') as f:
                xrs_data = json.load(f)
        else:
            xrs_data = {}
        xrs_data[time_str] = max_flux
        with open(XRS_PATH, 'w') as f:
            json.dump(xrs_data, f, indent=2)
        print(f"✅ XRS更新: {time_str} → {max_flux}")
    except Exception as e:
        print(f"❌ xrs.jsonの更新失敗: {e}")


def main():
    now_jst = datetime.now(tz=tz.gettz('Asia/Tokyo')) - timedelta(minutes=20)
    now_utc = now_jst.astimezone(tz=tz.tzutc())
    dt = now_utc.replace(minute=0, second=0, microsecond=0)

    while True:
        time_str = dt.strftime('%H')
        date_str = dt.strftime('%m%d')
        hmi_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_hmi.png")

        # HMIファイルが存在するか確認
        if os.path.exists(hmi_path):
            print(f"✅ HMIファイルが既に存在: {hmi_path}")
            break

        aia_images = []
        for wl in AIA_WAVELENGTHS:
            image_data = fetch_and_process_aia_image(wl, dt)
            aia_images.append(image_data)

            image_display = np.log1p(image_data)
            image_display = cv2.normalize(image_display, None, 0, 255, cv2.NORM_MINMAX)
            image_uint8 = image_display.astype(np.uint8)
            png_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_aia_{wl}.png")
            save_png(image_uint8, png_path)
            print(f"✅ AIA保存: {png_path}")

        hmi = download_hmi_image(dt)
        save_png(hmi.astype(np.uint8), hmi_path)
        print(f"✅ HMI保存: {hmi_path}")

        save_h5(aia_images, hmi, dt)
        update_xrs_json(dt)

        # 1時間前に遡る
        dt -= timedelta(hours=1)


if __name__ == '__main__':
    main()
