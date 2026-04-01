import heapq
import math
import io
import json
import os
import random
import tkinter as tk
import zipfile
from collections import Counter
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import numpy as np
    from PIL import Image, ImageTk
    import cv2
    import pywt
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Отсутствуют библиотеки: {e}. Функции сжатия изображений будут отключены.")
    LIBRARIES_AVAILABLE = False
    np = None
    Image = None
    ImageTk = None
    cv2 = None
    pywt = None

try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:
    RSCodec = None

    class ReedSolomonError(Exception):
        pass

def source_report(text="ABRACADABRA"):
    if not text:
        return "Введите текст для анализа."

    freq = Counter(text)
    total = len(text)
    if total == 0:
        return "Текст пустой."

    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())

    lines = ["Анализ источника:"]
    lines.append(f"Текст: {text}")
    lines.append(f"Длина: {total} символов")
    lines.append("")
    lines.append("Частоты символов:")
    for ch, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        prob = count / total
        lines.append(f"  {repr(ch)}: {count} ({prob:.4f})")

    lines.append("")
    lines.append(f"Энтропия H(X) = {entropy:.6f} бит/символ")

    # Простое кодирование с битом четности
    lines.append("")
    lines.append("Кодирование с битом четности:")
    for ch, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        code = bin(ord(ch))[2:].zfill(8)  # 8-bit ASCII
        parity = str(code.count('1') % 2)
        lines.append(f"  {repr(ch)}: {code} -> {code}{parity}")

    l_avg = 9.0  # 8 + 1 parity
    efficiency = entropy / l_avg
    redundancy = 1 - efficiency

    lines.append("")
    lines.append(f"Средняя длина кода L = {l_avg:.1f} бит/символ")
    lines.append(f"Эффективность η = {efficiency:.6f}")
    lines.append(f"Избыточность R = {redundancy:.6f} ({redundancy * 100:.2f}%)")

    return "\n".join(lines)




def build_huffman(text):
    freq = dict(Counter(text))
    if not freq:
        return {}, {}, ""

    heap = []
    uid = 0
    for ch, f in freq.items():
        heapq.heappush(heap, (f, uid, {"ch": ch, "left": None, "right": None}))
        uid += 1

    if len(heap) == 1:
        ch = next(iter(freq))
        codes = {ch: "0"}
        return freq, codes, "".join(codes[c] for c in text)

    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        parent = {"ch": None, "left": left, "right": right}
        heapq.heappush(heap, (f1 + f2, uid, parent))
        uid += 1

    root = heap[0][2]
    codes = {}

    def walk(node, prefix):
        if node["ch"] is not None:
            codes[node["ch"]] = prefix or "0"
            return
        walk(node["left"], prefix + "0")
        walk(node["right"], prefix + "1")

    walk(root, "")
    encoded = "".join(codes[c] for c in text)
    return freq, codes, encoded

def build_shannon_fano(text):
    freq = dict(Counter(text))
    if not freq:
        return {}, {}, ""

    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    codes = {ch: "" for ch, _ in items}

    if len(items) == 1:
        ch = items[0][0]
        codes[ch] = "0"
        return freq, codes, "".join(codes[c] for c in text)

    def split_group(group):
        total = sum(v for _, v in group)
        left_sum = 0
        best_idx = 1
        best_diff = float("inf")
        for i in range(1, len(group)):
            left_sum += group[i - 1][1]
            diff = abs(total - 2 * left_sum)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return group[:best_idx], group[best_idx:]

    def assign(group):
        if len(group) <= 1:
            return
        left, right = split_group(group)
        for ch, _ in left:
            codes[ch] += "0"
        for ch, _ in right:
            codes[ch] += "1"
        assign(left)
        assign(right)

    assign(items)
    for ch in codes:
        if not codes[ch]:
            codes[ch] = "0"

    encoded = "".join(codes[c] for c in text)
    return freq, codes, encoded

def coding_report(title, text, freq, codes, encoded, with_efficiency):
    original_bits = len(text) * 8
    encoded_bits = len(encoded)
    efficiency = (original_bits - encoded_bits) / original_bits if original_bits else 0.0
    compression = (original_bits / encoded_bits) if encoded_bits else 0.0

    lines = [title, "-" * len(title), f"Вход: {text}", "Частоты:"]
    for ch, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {repr(ch)}: {count}")

    lines.append("Коды:")
    for ch, code in sorted(codes.items(), key=lambda x: (len(x[1]), x[0])):
        lines.append(f"  {repr(ch)}: {code}")

    lines.append(f"Закодированное сообщение: {encoded}")
    lines.append(f"Lисходный = {original_bits} бит")
    lines.append(f"Lкодированный = {encoded_bits} бит")

    if with_efficiency:
        lines.append(f"Эффективность E = (Lисх - Lкод)/Lисх = {efficiency:.4f} ({efficiency * 100:.2f}%)")

    lines.append(f"Коэффициент сжатия C = Lисх/Lкод = {compression:.4f}")
    return "\n".join(lines)

def reed_solomon_report(text, nsym):
    if RSCodec is None:
        return "\n".join(
            [
                "Задание 3: Рид-Соломон",
                "-----------------------",
                "Библиотека reedsolo не найдена.",
                "Установите: python -m pip install reedsolo",
            ]
        )

    data = text.encode("utf-8")
    codec = RSCodec(nsym)

    encoded = bytearray(codec.encode(data))
    corrupted = bytearray(encoded)
    idx = 2 if len(corrupted) > 2 else 0
    corrupted[idx] ^= 0xFF

    try:
        decoded = codec.decode(bytes(corrupted))
        decoded_bytes = decoded[0] if isinstance(decoded, tuple) else decoded
        decoded_bytes = bytes(decoded_bytes)
        restored_text = decoded_bytes.decode("utf-8", errors="replace")
        ok = decoded_bytes == data
        status = "успешно" if ok else "частично"
    except ReedSolomonError as err:
        restored_text = f"Ошибка декодирования: {err}"
        ok = False
        status = "не удалось"

    n = len(encoded)
    k = len(data)
    r = n - k
    max_errors = r // 2

    encoded_dec = ", ".join(str(b) for b in encoded)
    corrupted_dec = ", ".join(str(b) for b in corrupted)

    lines = [
        "Задание 3: Рид-Соломон",
        "-----------------------",
        f"Исходное сообщение: {text}",
        f"Параметры [n, k] = [{n}, {k}]",
        f"Избыточные байты r = n - k = {r}",
        f"Максимум исправляемых ошибок = (n-k)/2 = {max_errors}",
        f"Кодированное (dec): {encoded_dec}",
        f"Внесена ошибка в байт: {idx}",
        f"Поврежденное (dec): {corrupted_dec}",
        f"Декодирование: {status}",
        f"Восстановленное сообщение: {restored_text}",
        f"Совпадает с исходным: {'Да' if ok else 'Нет'}",
    ]
    return "\n".join(lines)




def decode_text_bytes(data):
    if not data:
        return "", "utf-8"

    candidate_encodings = ["utf-8-sig", "utf-8"]
    if b"\x00" in data:
        candidate_encodings.extend(["utf-16", "utf-16-le", "utf-16-be"])
    candidate_encodings.extend(["cp1251", "cp866", "latin-1"])

    for encoding in candidate_encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace"), "utf-8"


def try_decode_with_encoding(data, encoding):
    try:
        return data.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return decode_text_bytes(data)[0]


def rle_encode_bytes(data):
    if not data:
        return []

    runs = []
    current = data[0]
    count = 1

    for value in data[1:]:
        if value == current:
            count += 1
        else:
            runs.append((count, current))
            current = value
            count = 1

    runs.append((count, current))
    return runs


def rle_decode_bytes(runs):
    decoded = bytearray()
    for count, value in runs:
        decoded.extend(bytes([value]) * count)
    return bytes(decoded)


def serialize_rle_runs(runs):
    return "".join(f"{count}\t{value}\n" for count, value in runs).encode("utf-8")


def deserialize_rle_runs(payload_text):
    runs = []

    for line_number, line in enumerate(payload_text.splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Некорректная строка RLE #{line_number}: {line!r}")

        count = int(parts[0])
        value = int(parts[1])

        if count <= 0:
            raise ValueError(f"Длина серии должна быть > 0 (строка {line_number})")
        if not 0 <= value <= 255:
            raise ValueError(f"Байт вне диапазона 0..255 (строка {line_number})")

        runs.append((count, value))

    return runs


def build_archive(file_name, data, encoding):
    base_name = os.path.basename(file_name) or "document.txt"
    runs = rle_encode_bytes(data)
    payload_bytes = serialize_rle_runs(runs)
    payload_name = f"{base_name}.rle"

    metadata = {
        "lab": 4,
        "variant": 7,
        "algorithm": "RLE",
        "original_name": base_name,
        "original_extension": os.path.splitext(base_name)[1],
        "encoding": encoding,
        "payload_name": payload_name,
        "original_size_bytes": len(data),
        "rle_payload_size_bytes": len(payload_bytes),
        "run_count": len(runs),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr(payload_name, payload_bytes)

    archive_bytes = buffer.getvalue()
    metadata["zip_size_bytes"] = len(archive_bytes)

    return {
        "archive_bytes": archive_bytes,
        "metadata": metadata,
        "payload_text": payload_bytes.decode("utf-8"),
        "runs": runs,
    }


def unpack_archive(archive_bytes):
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        names = archive.namelist()
        if "metadata.json" not in names:
            raise ValueError("В архиве отсутствует metadata.json")

        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        if metadata.get("algorithm") not in (None, "RLE"):
            raise ValueError("Архив не относится к варианту RLE")
        if metadata.get("variant") not in (None, 7):
            raise ValueError("Архив не относится к варианту 7")
        payload_name = metadata.get("payload_name")

        if not payload_name or payload_name not in names:
            candidates = [name for name in names if name.lower().endswith(".rle")]
            if not candidates:
                raise ValueError("В архиве не найден файл с RLE-данными")
            payload_name = candidates[0]
            metadata["payload_name"] = payload_name

        payload_text = archive.read(payload_name).decode("utf-8")

    runs = deserialize_rle_runs(payload_text)
    restored_bytes = rle_decode_bytes(runs)
    encoding = metadata.get("encoding") or "utf-8"
    restored_text = try_decode_with_encoding(restored_bytes, encoding)

    return {
        "metadata": metadata,
        "payload_text": payload_text,
        "runs": runs,
        "restored_bytes": restored_bytes,
        "restored_text": restored_text,
    }


def format_size(size_bytes):
    if size_bytes is None:
        return "-"
    return f"{size_bytes} байт ({size_bytes / 1024:.2f} КБ)"


def build_rle_preview(runs, limit=18):
    if not runs:
        return "Серии не найдены."

    lines = ["Первые серии RLE (count -> byte):"]
    for index, (count, value) in enumerate(runs[:limit], start=1):
        lines.append(f"{index:>2}. {count} -> {value}")

    if len(runs) > limit:
        lines.append(f"... еще {len(runs) - limit} серий")

    return "\n".join(lines)


def generate_signal(fs, duration, amplitude, f_start, f_end):
    sample_count = max(2, int(fs * duration))
    dt = 1.0 / fs
    slope = (f_end - f_start) / duration if duration > 0 else 0.0

    t_values = [i * dt for i in range(sample_count)]
    signal = [
        amplitude * math.sin(2.0 * math.pi * (f_start * t + 0.5 * slope * t * t))
        for t in t_values
    ]
    return t_values, signal


def add_block_noise(signal, sigma, block_size=10):
    noisy = signal.copy()
    for start in range(0, len(noisy), block_size):
        noise = random.gauss(0.0, sigma)
        end = min(start + block_size, len(noisy))
        for i in range(start, end):
            noisy[i] += noise
    return noisy


def pass_discrete_channel(signal, loss_probability):
    transmitted = signal.copy()
    loss_count = 0
    delay_count = 0

    for i in range(len(transmitted)):
        if random.random() < loss_probability:
            transmitted[i] = 0.0
            loss_count += 1

    # Вариант 7: каждое 5-е значение заменяем предыдущим (имитация задержки).
    for i in range(4, len(transmitted), 5):
        transmitted[i] = transmitted[i - 1]
        delay_count += 1

    return transmitted, loss_count, delay_count


def moving_average_filter(signal, window_size):
    window_size = max(1, int(window_size))
    filtered = []
    for i in range(len(signal)):
        left = max(0, i - window_size + 1)
        chunk = signal[left : i + 1]
        filtered.append(sum(chunk) / len(chunk))
    return filtered


def calc_channel_metrics(original, transmitted, fs, f_max, bandwidth):
    n_total = len(original)
    if n_total == 0:
        return {
            "ber": 0.0,
            "n_err": 0,
            "n_total": 0,
            "snr_linear": float("inf"),
            "snr_db": float("inf"),
            "capacity": float("inf"),
            "nyquist_ok": True,
            "nyquist_need": 0.0,
        }

    original_bits = [1 if x >= 0 else 0 for x in original]
    tx_bits = [1 if x >= 0 else 0 for x in transmitted]
    n_err = sum(1 for a, b in zip(original_bits, tx_bits) if a != b)
    ber = n_err / n_total

    signal_power = sum(x * x for x in original) / n_total
    noise_power = sum((tx - src) ** 2 for src, tx in zip(original, transmitted)) / n_total
    snr_linear = signal_power / noise_power if noise_power > 0 else float("inf")
    snr_db = 10.0 * math.log10(snr_linear) if snr_linear > 0 and math.isfinite(snr_linear) else float("inf")

    capacity = bandwidth * math.log2(1.0 + snr_linear) if math.isfinite(snr_linear) else float("inf")
    nyquist_need = 2.0 * f_max
    nyquist_ok = fs >= nyquist_need

    return {
        "ber": ber,
        "n_err": n_err,
        "n_total": n_total,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "capacity": capacity,
        "nyquist_ok": nyquist_ok,
        "nyquist_need": nyquist_need,
    }




def rgb_to_ycrcb(img):
    if not LIBRARIES_AVAILABLE:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def ycrcb_to_rgb(img):
    if not LIBRARIES_AVAILABLE:
        return img
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def chroma_subsampling(img, ratio='4:2:0'):
    if not LIBRARIES_AVAILABLE:
        return img
    h, w = img.shape[:2]
    y = img[:, :, 0]
    cr = img[:, :, 1]
    cb = img[:, :, 2]
    
    if ratio == '4:2:0':
        cr_sub = cv2.resize(cr, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        cb_sub = cv2.resize(cb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        cr_up = cv2.resize(cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        cb_up = cv2.resize(cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    elif ratio == '4:2:2':
        cr_sub = cv2.resize(cr, (w//2, h), interpolation=cv2.INTER_LINEAR)
        cb_sub = cv2.resize(cb, (w//2, h), interpolation=cv2.INTER_LINEAR)
        cr_up = cv2.resize(cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        cb_up = cv2.resize(cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        cr_up, cb_up = cr, cb
    
    return np.stack([y, cr_up, cb_up], axis=2).astype(np.uint8)

def block_dct(img_channel, block_size=8):
    if not LIBRARIES_AVAILABLE:
        return img_channel.astype(np.float32)
    h, w = img_channel.shape
    dct_img = np.zeros_like(img_channel, dtype=np.float32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_channel[i:i+block_size, j:j+block_size].astype(np.float32)
            if block.shape == (block_size, block_size):
                dct_block = cv2.dct(block)
                dct_img[i:i+block_size, j:j+block_size] = dct_block
    
    return dct_img

def block_idct(dct_channel, block_size=8):
    if not LIBRARIES_AVAILABLE:
        return dct_channel.astype(np.uint8)
    h, w = dct_channel.shape
    img_channel = np.zeros_like(dct_channel, dtype=np.float32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dct_block = dct_channel[i:i+block_size, j:j+block_size]
            if dct_block.shape == (block_size, block_size):
                block = cv2.idct(dct_block)
                img_channel[i:i+block_size, j:j+block_size] = block
    
    return img_channel.astype(np.uint8)

# JPEG quantization matrix
JPEG_QUANTIZATION = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]) if LIBRARIES_AVAILABLE else None

def quantize_dct(dct_channel, quality=50, block_size=8):
    if not LIBRARIES_AVAILABLE or JPEG_QUANTIZATION is None:
        return dct_channel.astype(np.int32)
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100
    
    # Scale quantization matrix based on quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    quant_matrix = (JPEG_QUANTIZATION * scale / 100).astype(np.int32)
    quant_matrix[quant_matrix == 0] = 1
    
    h, w = dct_channel.shape
    quantized = np.zeros_like(dct_channel, dtype=np.int32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_channel[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                quantized_block = np.round(block / quant_matrix).astype(np.int32)
                quantized[i:i+block_size, j:j+block_size] = quantized_block
    
    return quantized

def dequantize_dct(quantized_channel, quality=50, block_size=8):
    if not LIBRARIES_AVAILABLE or JPEG_QUANTIZATION is None:
        return quantized_channel.astype(np.float32)
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    quant_matrix = (JPEG_QUANTIZATION * scale / 100).astype(np.int32)
    quant_matrix[quant_matrix == 0] = 1
    
    h, w = quantized_channel.shape
    dct_channel = np.zeros_like(quantized_channel, dtype=np.float32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_channel[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = block * quant_matrix
                dct_channel[i:i+block_size, j:j+block_size] = dct_block
    
    return dct_channel

def dwt_compress(img, wavelet='haar', level=1):
    if not LIBRARIES_AVAILABLE:
        return img
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    # Simple thresholding - set small coefficients to zero
    threshold = 10
    coeffs_thresholded = [coeffs[0]]
    for c in coeffs[1:]:
        cH, cV, cD = c
        cH[np.abs(cH) < threshold] = 0
        cV[np.abs(cV) < threshold] = 0
        cD[np.abs(cD) < threshold] = 0
        coeffs_thresholded.append((cH, cV, cD))
    
    return pywt.waverec2(coeffs_thresholded, wavelet).astype(np.uint8)

def limit_colors(img, max_colors=256):
    if not LIBRARIES_AVAILABLE:
        return img
    if max_colors >= 256:
        return img
    
    # Convert to float
    img_float = img.astype(np.float32)
    h, w, c = img_float.shape
    
    # Reshape to 2D array of pixels
    pixels = img_float.reshape(-1, c)
    
    # Use k-means to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, max_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstruct image
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, c)
    
    return quantized

def jpeg_compress(img, quality=50):
    if not LIBRARIES_AVAILABLE:
        return img
    # Convert to YCrCb
    ycrcb = rgb_to_ycrcb(img)
    
    # Chroma subsampling
    ycrcb_sub = chroma_subsampling(ycrcb, '4:2:0')
    
    # Apply DCT to each channel
    y_dct = block_dct(ycrcb_sub[:, :, 0])
    cr_dct = block_dct(ycrcb_sub[:, :, 1])
    cb_dct = block_dct(ycrcb_sub[:, :, 2])
    
    # Quantization
    y_quant = quantize_dct(y_dct, quality)
    cr_quant = quantize_dct(cr_dct, quality)
    cb_quant = quantize_dct(cb_dct, quality)
    
    # Dequantization
    y_dct_rec = dequantize_dct(y_quant, quality)
    cr_dct_rec = dequantize_dct(cr_quant, quality)
    cb_dct_rec = dequantize_dct(cb_quant, quality)
    
    # Inverse DCT
    y_rec = block_idct(y_dct_rec)
    cr_rec = block_idct(cr_dct_rec)
    cb_rec = block_idct(cb_dct_rec)
    
    # Combine channels
    ycrcb_rec = np.stack([y_rec, cr_rec, cb_rec], axis=2)
    
    # Convert back to RGB
    rgb_rec = ycrcb_to_rgb(ycrcb_rec)
    
    return rgb_rec

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная: энтропия и алгоритмы кодирования")
        self.root.geometry("1180x860")

        self.input_text = tk.StringVar(value="ABRACADABRA")
        self.nsym = tk.IntVar(value=10)

        self.fs_var = tk.DoubleVar(value=500.0)
        self.duration_var = tk.DoubleVar(value=2.0)
        self.amplitude_var = tk.DoubleVar(value=1.0)
        self.f_start_var = tk.DoubleVar(value=2.0)
        self.f_end_var = tk.DoubleVar(value=20.0)
        self.noise_sigma_var = tk.DoubleVar(value=0.2)
        self.loss_prob_var = tk.DoubleVar(value=0.08)
        self.filter_window_var = tk.IntVar(value=9)
        self.bandwidth_var = tk.DoubleVar(value=200.0)
        self.lab4_method_var = tk.StringVar(value="RLE")
        self.lab4_status_var = tk.StringVar(value="ЛР4: файл не загружен.")

        self.lab4_source_path = ""
        self.lab4_source_bytes = b""
        self.lab4_source_text = ""
        self.lab4_source_encoding = "utf-8"
        self.lab4_archive_bytes = b""
        self.lab4_archive_name = ""
        self.lab4_archive_meta = {}
        self.lab4_decoded_text = ""

        # Image compression variables
        self.img_original = None
        self.img_compressed = None
        self.img_quality = tk.IntVar(value=50)
        self.img_status = tk.StringVar(value="Изображение не загружено")
        self.img_path = ""

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab1 = ttk.Frame(notebook, padding=10)
        tab2 = ttk.Frame(notebook, padding=10)
        tab3 = ttk.Frame(notebook, padding=10)
        tab4 = ttk.Frame(notebook, padding=10)
        tab5 = ttk.Frame(notebook, padding=10)
        notebook.add(tab1, text="Энтропия")
        notebook.add(tab2, text="Алгоритмы")
        notebook.add(tab3, text="Каналы")
        notebook.add(tab4, text="RLE")
        notebook.add(tab5, text="Сжатие изображений")

        self.build_entropy_tab(tab1)
        self.build_algorithms_tab(tab2)
        self.build_channels_tab(tab3)
        self.build_lab4_tab(tab4)
        self.build_img_tab(tab5)

        self.refresh_entropy()
        self.run_algorithms()
        self.run_channel_simulation()

    def build_entropy_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x")

        ttk.Label(top, text="Текст для анализа:").pack(side="left")
        self.entropy_input = tk.StringVar(value="ABRACADABRA")
        ttk.Entry(top, textvariable=self.entropy_input, width=30).pack(side="left", padx=8)
        ttk.Button(top, text="Пересчитать", command=self.refresh_entropy).pack(side="right")

        self.entropy_output = ScrolledText(parent, font=("Consolas", 11), wrap="word")
        self.entropy_output.pack(fill="both", expand=True, pady=(8, 0))
        self.entropy_output.configure(state="disabled")

    def build_algorithms_tab(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill="x")

        ttk.Label(row, text="Строка:").pack(side="left")
        ttk.Entry(row, textvariable=self.input_text, width=45).pack(side="left", padx=8)

        ttk.Label(row, text="nsym (RS):").pack(side="left")
        ttk.Spinbox(row, from_=2, to=64, textvariable=self.nsym, width=6).pack(side="left", padx=8)

        ttk.Button(row, text="Выполнить", command=self.run_algorithms).pack(side="left")

        hint = "Вкладка выполняет: Хаффман, Шеннон-Фано и Рид-Соломон"
        ttk.Label(parent, text=hint, foreground="#333333").pack(fill="x", pady=(8, 4))

        self.alg_output = ScrolledText(parent, font=("Consolas", 10), wrap="word")
        self.alg_output.pack(fill="both", expand=True)
        self.alg_output.configure(state="disabled")

    def build_channels_tab(self, parent):
        controls = ttk.LabelFrame(parent, text="Параметры моделирования", padding=8)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="fs (Гц)").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.fs_var, width=10).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(controls, text="T (с)").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.duration_var, width=10).grid(row=0, column=3, padx=4, pady=2)

        ttk.Label(controls, text="A").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.amplitude_var, width=10).grid(row=0, column=5, padx=4, pady=2)

        ttk.Label(controls, text="f0 (Гц)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.f_start_var, width=10).grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(controls, text="f1 (Гц)").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.f_end_var, width=10).grid(row=1, column=3, padx=4, pady=2)

        ttk.Label(controls, text="σ шума").grid(row=1, column=4, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.noise_sigma_var, width=10).grid(row=1, column=5, padx=4, pady=2)

        ttk.Label(controls, text="p потерь").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.loss_prob_var, width=10).grid(row=2, column=1, padx=4, pady=2)

        ttk.Label(controls, text="Окно фильтра").grid(row=2, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.filter_window_var, width=10).grid(row=2, column=3, padx=4, pady=2)

        ttk.Label(controls, text="B (Гц)").grid(row=2, column=4, sticky="w", padx=4, pady=2)
        ttk.Entry(controls, textvariable=self.bandwidth_var, width=10).grid(row=2, column=5, padx=4, pady=2)

        ttk.Button(controls, text="Запустить моделирование", command=self.run_channel_simulation).grid(
            row=0, column=6, rowspan=3, padx=10, pady=2, sticky="ns"
        )

        graph_frame = ttk.Frame(parent)
        graph_frame.pack(fill="both", expand=True)

        self.canvas_original = tk.Canvas(graph_frame, bg="white", height=180, highlightthickness=1, highlightbackground="#cccccc")
        self.canvas_original.pack(fill="x", pady=(0, 6))

        self.canvas_transmitted = tk.Canvas(graph_frame, bg="white", height=180, highlightthickness=1, highlightbackground="#cccccc")
        self.canvas_transmitted.pack(fill="x", pady=(0, 6))

        self.canvas_filtered = tk.Canvas(graph_frame, bg="white", height=180, highlightthickness=1, highlightbackground="#cccccc")
        self.canvas_filtered.pack(fill="x", pady=(0, 6))

        self.channel_output = ScrolledText(parent, font=("Consolas", 10), wrap="word", height=9)
        self.channel_output.pack(fill="both", expand=False)
        self.channel_output.configure(state="disabled")

    def build_lab4_tab(self, parent):
        controls = ttk.LabelFrame(parent, text="Лабораторная работа №4, вариант 7", padding=8)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Загрузить файл", command=self.load_lab4_source_file).grid(
            row=0, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Button(controls, text="Загрузить ZIP", command=self.load_lab4_archive_file).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(controls, text="Метод сжатия:").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        method_box = ttk.Combobox(
            controls,
            textvariable=self.lab4_method_var,
            values=["RLE"],
            state="readonly",
            width=16,
        )
        method_box.grid(row=0, column=3, padx=4, pady=4, sticky="w")

        ttk.Button(controls, text="Сжать", command=self.compress_lab4_file).grid(
            row=0, column=4, padx=4, pady=4, sticky="w"
        )
        ttk.Button(controls, text="Сохранить", command=self.save_lab4_archive).grid(
            row=0, column=5, padx=4, pady=4, sticky="w"
        )
        ttk.Button(controls, text="Декодировать", command=self.decode_lab4_archive).grid(
            row=0, column=6, padx=4, pady=4, sticky="w"
        )

        ttk.Label(
            controls,
            text="RLE, выходной формат .zip, архив сохраняет исходное расширение файла.",
            foreground="#333333",
        ).grid(row=1, column=0, columnspan=7, sticky="w", padx=4, pady=(2, 0))
        ttk.Label(controls, textvariable=self.lab4_status_var, foreground="#0b5394").grid(
            row=2, column=0, columnspan=7, sticky="w", padx=4, pady=(4, 0)
        )

        preview_row = ttk.Frame(parent)
        preview_row.pack(fill="both", expand=True, pady=(0, 8))

        source_frame = ttk.LabelFrame(preview_row, text="Исходный текст", padding=6)
        source_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.lab4_source_preview = ScrolledText(source_frame, font=("Consolas", 10), wrap="word")
        self.lab4_source_preview.pack(fill="both", expand=True)
        self.lab4_source_preview.configure(state="disabled")

        result_frame = ttk.LabelFrame(preview_row, text="Результат", padding=6)
        result_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.lab4_result_preview = ScrolledText(result_frame, font=("Consolas", 10), wrap="word")
        self.lab4_result_preview.pack(fill="both", expand=True)
        self.lab4_result_preview.configure(state="disabled")

        info_frame = ttk.LabelFrame(parent, text="Информация", padding=6)
        info_frame.pack(fill="both", expand=False)
        self.lab4_info_output = ScrolledText(info_frame, font=("Consolas", 10), wrap="word", height=12)
        self.lab4_info_output.pack(fill="both", expand=True)
        self.lab4_info_output.configure(state="disabled")

    def build_img_tab(self, parent):
        controls = ttk.LabelFrame(parent, text="Сжатие изображений (Вариант 7: PNG, ограничение цветов до 256, DWT + Квантование)", padding=8)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Загрузить изображение", command=self.load_image).grid(row=0, column=0, padx=4, pady=4)
        ttk.Label(controls, text="Качество:").grid(row=0, column=1, padx=4, pady=4)
        ttk.Scale(controls, from_=1, to=100, variable=self.img_quality, orient="horizontal").grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(controls, text="Сжать", command=self.compress_image).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(controls, text="Сохранить", command=self.save_compressed_image).grid(row=0, column=4, padx=4, pady=4)

        ttk.Label(controls, textvariable=self.img_status, foreground="#0b5394").grid(row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(4, 0))

        # Image display area
        img_frame = ttk.Frame(parent)
        img_frame.pack(fill="both", expand=True)

        # Original image
        orig_frame = ttk.LabelFrame(img_frame, text="Исходное изображение", padding=6)
        orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.orig_canvas = tk.Canvas(orig_frame, bg="gray", width=300, height=300)
        self.orig_canvas.pack(fill="both", expand=True)

        # Compressed image
        comp_frame = ttk.LabelFrame(img_frame, text="Сжатое изображение", padding=6)
        comp_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.comp_canvas = tk.Canvas(comp_frame, bg="gray", width=300, height=300)
        self.comp_canvas.pack(fill="both", expand=True)

    def load_image(self):
        if not LIBRARIES_AVAILABLE:
            messagebox.showerror("Ошибка", "Необходимые библиотеки не установлены. Установите: numpy, pillow, opencv-python, pywavelets")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.img_original = Image.open(file_path).convert('RGB')
            self.img_path = file_path
            self.display_image(self.img_original, self.orig_canvas)
            self.img_status.set(f"Загружено: {os.path.basename(file_path)}")
            self.img_compressed = None
            self.comp_canvas.delete("all")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def compress_image(self):
        if not LIBRARIES_AVAILABLE:
            messagebox.showerror("Ошибка", "Необходимые библиотеки не установлены")
            return

        if self.img_original is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            img_array = np.array(self.img_original)
            
            # For variant 7: limit colors to 256, then apply JPEG compression
            img_limited = limit_colors(img_array, 256)
            
            # Apply JPEG compression
            img_compressed = jpeg_compress(img_limited, self.img_quality.get())
            
            self.img_compressed = Image.fromarray(img_compressed)
            self.display_image(self.img_compressed, self.comp_canvas)
            
            # Calculate sizes
            orig_size = len(self.img_original.tobytes())
            comp_size = len(self.img_compressed.tobytes())
            ratio = orig_size / comp_size if comp_size > 0 else 0
            
            self.img_status.set(f"Сжато. Оригинал: {orig_size} байт, Сжатый: {comp_size} байт, Коэффициент: {ratio:.2f}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сжатия: {e}")

    def save_compressed_image(self):
        if self.img_compressed is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните сжатие")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.img_compressed.save(file_path)
                messagebox.showinfo("Успех", "Изображение сохранено")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")

    def display_image(self, img, canvas):
        if not LIBRARIES_AVAILABLE:
            return

        canvas.delete("all")
        # Resize image to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 300
            
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_img)
        
        # Store reference to prevent garbage collection
        if canvas == self.orig_canvas:
            self.orig_img_tk = img_tk
        else:
            self.comp_img_tk = img_tk
        
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor="nw", image=img_tk)

    @staticmethod
    def set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def draw_signal(canvas, data, color, title):
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 600)
        height = max(canvas.winfo_height(), 140)
        margin = 28
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin
        canvas.delete("all")

        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#cccccc")
        canvas.create_text(margin + 4, 12, text=title, anchor="w", fill="#222222", font=("Segoe UI", 9, "bold"))

        if not data:
            return

        min_y = min(data)
        max_y = max(data)
        if abs(max_y - min_y) < 1e-12:
            min_y -= 1.0
            max_y += 1.0

        # Если точек очень много, прореживаем для быстрого рисования.
        max_points = max(300, plot_w)
        step = max(1, len(data) // max_points)
        points = []

        for idx in range(0, len(data), step):
            x = margin + (idx / (len(data) - 1)) * plot_w
            y = margin + (max_y - data[idx]) / (max_y - min_y) * plot_h
            points.extend([x, y])

        if len(points) >= 4:
            canvas.create_line(points, fill=color, width=1.4, smooth=True)

        if min_y <= 0 <= max_y:
            zero_y = margin + (max_y / (max_y - min_y)) * plot_h
            canvas.create_line(margin, zero_y, width - margin, zero_y, fill="#bbbbbb", dash=(3, 2))

    def run_channel_simulation(self):
        try:
            fs = float(self.fs_var.get())
            duration = float(self.duration_var.get())
            amplitude = float(self.amplitude_var.get())
            f_start = float(self.f_start_var.get())
            f_end = float(self.f_end_var.get())
            noise_sigma = float(self.noise_sigma_var.get())
            loss_prob = float(self.loss_prob_var.get())
            filter_window = int(self.filter_window_var.get())
            bandwidth = float(self.bandwidth_var.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Ошибка параметров", "Введите корректные числовые параметры для ЛР3.")
            return

        if fs <= 0 or duration <= 0 or amplitude <= 0 or bandwidth <= 0:
            messagebox.showwarning("Ошибка параметров", "Параметры fs, T, A и B должны быть больше 0.")
            return
        if f_start < 0 or f_end <= 0:
            messagebox.showwarning("Ошибка параметров", "Частоты f0 и f1 должны быть положительными.")
            return
        if not 0 <= loss_prob <= 1:
            messagebox.showwarning("Ошибка параметров", "Вероятность потерь должна быть в диапазоне [0, 1].")
            return
        if filter_window < 1:
            messagebox.showwarning("Ошибка параметров", "Окно фильтра должно быть не меньше 1.")
            return

        _, original = generate_signal(fs, duration, amplitude, f_start, f_end)
        noisy = add_block_noise(original, noise_sigma, block_size=10)
        transmitted, loss_count, delay_count = pass_discrete_channel(noisy, loss_prob)
        filtered = moving_average_filter(transmitted, filter_window)

        metrics = calc_channel_metrics(original, transmitted, fs, max(f_start, f_end), bandwidth)

        self.draw_signal(self.canvas_original, original, "#1f77b4", "Исходный сигнал (частота плавно растет)")
        self.draw_signal(self.canvas_transmitted, transmitted, "#d62728", "Переданный сигнал (шум + потери + задержка)")
        self.draw_signal(self.canvas_filtered, filtered, "#2ca02c", "Сглаженный сигнал (имитация аналогового канала)")

        snr_lin_text = f"{metrics['snr_linear']:.6f}" if math.isfinite(metrics["snr_linear"]) else "inf"
        snr_db_text = f"{metrics['snr_db']:.3f}" if math.isfinite(metrics["snr_db"]) else "inf"
        cap_text = f"{metrics['capacity']:.3f}" if math.isfinite(metrics["capacity"]) else "inf"
        nyq_status = "выполнено" if metrics["nyquist_ok"] else "НЕ выполнено"

        lines = [
            f"Количество точек: {len(original)}",
            f"Потери в дискретном канале: {loss_count}",
            f"Имитация задержки (каждое 5-е): {delay_count}",
            "",
            "Расчеты:",
            f"BER = Nerr / Ntotal = {metrics['n_err']} / {metrics['n_total']} = {metrics['ber']:.6f}",
            f"SNR (линейно): {snr_lin_text}",
            f"SNR (дБ): {snr_db_text}",
            f"C = B*log2(1+SNR) = {bandwidth}*log2(1+SNR) = {cap_text} бит/с",
            f"Проверка Найквиста: fs = {fs}, требуется >= {metrics['nyquist_need']:.3f} -> {nyq_status}",
        ]

        self.set_text(self.channel_output, "\n".join(lines))

    def refresh_entropy(self):
        text = self.entropy_input.get()
        report = source_report(text)
        self.set_text(self.entropy_output, report)

    def run_algorithms(self):
        text = self.input_text.get()
        if not text:
            messagebox.showwarning("Пустой ввод", "Введите строку для кодирования.")
            return

        nsym = self.nsym.get()
        if nsym < 2:
            messagebox.showwarning("Ошибка", "nsym должен быть >= 2")
            return
        h_freq, h_codes, h_encoded = build_huffman(text)
        h_report = coding_report(
            "Задание 1: Хаффман",
            text,
            h_freq,
            h_codes,
            h_encoded,
            with_efficiency=True,
        )

        sf_freq, sf_codes, sf_encoded = build_shannon_fano(text)
        sf_report = coding_report(
            "Задание 2: Шеннон-Фано",
            text,
            sf_freq,
            sf_codes,
            sf_encoded,
            with_efficiency=False,
        )

        rs_report = reed_solomon_report(text, nsym)

        full_text = f"{h_report}\n\n{sf_report}\n\n{rs_report}\n"
        self.set_text(self.alg_output, full_text)

    def load_lab4_source_file(self):
        path = filedialog.askopenfilename(
            title="Выберите текстовый файл",
            filetypes=[
                ("Текстовые файлы", "*.txt *.log *.csv *.md *.html"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        try:
            with open(path, "rb") as src_file:
                data = src_file.read()
        except OSError as err:
            messagebox.showerror("Ошибка чтения", f"Не удалось открыть файл:\n{err}")
            return

        text, encoding = decode_text_bytes(data)

        self.lab4_source_path = path
        self.lab4_source_bytes = data
        self.lab4_source_text = text
        self.lab4_source_encoding = encoding
        self.lab4_archive_bytes = b""
        self.lab4_archive_name = ""
        self.lab4_archive_meta = {}
        self.lab4_decoded_text = ""

        self.set_text(self.lab4_source_preview, text)
        self.set_text(self.lab4_result_preview, "")

        size_note = "Файл укладывается в рекомендуемые 100 КБ."
        if len(data) > 100 * 1024:
            size_note = "Файл больше 100 КБ, но приложение все равно может выполнить сжатие."

        lines = [
            "Файл загружен.",
            f"Путь: {path}",
            f"Имя: {os.path.basename(path)}",
            f"Расширение: {os.path.splitext(path)[1] or '(без расширения)'}",
            f"Размер исходного файла: {format_size(len(data))}",
            f"Определенная кодировка: {encoding}",
            size_note,
        ]

        self.set_text(self.lab4_info_output, "\n".join(lines))
        self.lab4_status_var.set(f"ЛР4: загружен файл {os.path.basename(path)}")

    def load_lab4_archive_file(self):
        path = filedialog.askopenfilename(
            title="Выберите архивN",
            filetypes=[("ZIP архив", "*.zip"), ("Все файлы", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "rb") as archive_file:
                archive_bytes = archive_file.read()
            unpacked = unpack_archive(archive_bytes)
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as err:
            messagebox.showerror("Ошибка архива", f"Не удалось открыть архив:\n{err}")
            return

        self.lab4_archive_bytes = archive_bytes
        self.lab4_archive_name = os.path.basename(path)
        self.lab4_archive_meta = unpacked["metadata"]
        self.lab4_decoded_text = ""

        info_lines = [
            "ZIP-архив загружен.",
            f"Путь: {path}",
            f"Размер ZIP: {format_size(len(archive_bytes))}",
            f"Алгоритм: {self.lab4_archive_meta.get('algorithm', 'RLE')}",
            f"Исходное имя файла: {self.lab4_archive_meta.get('original_name', '-')}",
            f"Исходное расширение: {self.lab4_archive_meta.get('original_extension', '-')}",
            f"Файл с данными внутри ZIP: {self.lab4_archive_meta.get('payload_name', '-')}",
            f"Количество серий RLE: {len(unpacked['runs'])}",
            "Нажмите «Декодировать», чтобы восстановить текст.",
        ]

        self.set_text(self.lab4_result_preview, build_rle_preview(unpacked["runs"]))
        self.set_text(self.lab4_info_output, "\n".join(info_lines))
        self.lab4_status_var.set(f"загружен архив {os.path.basename(path)}")

    def compress_lab4_file(self):
        if not self.lab4_source_path:
            messagebox.showwarning("Нет файла", "Сначала загрузите текстовый файл для сжатия.")
            return

        if self.lab4_method_var.get() != "RLE":
            messagebox.showwarning("Метод недоступен", "доступен только метод RLE.")
            return

        archive_data = build_archive(
            self.lab4_source_path or "document.txt",
            self.lab4_source_bytes,
            self.lab4_source_encoding,
        )
        metadata = archive_data["metadata"]
        zip_size = len(archive_data["archive_bytes"])
        original_size = len(self.lab4_source_bytes)
        compression_ratio = (original_size / zip_size) if zip_size else 0.0
        saved_percent = (1.0 - zip_size / original_size) * 100 if original_size else 0.0

        self.lab4_archive_bytes = archive_data["archive_bytes"]
        self.lab4_archive_meta = metadata
        self.lab4_archive_name = f"{metadata['original_name']}.zip"

        preview_text = build_rle_preview(archive_data["runs"])
        preview_text += "\n\nПервые строки сериализованных данных:\n"
        preview_text += "\n".join(archive_data["payload_text"].splitlines()[:12])
        self.set_text(self.lab4_result_preview, preview_text)

        lines = [
            "Сжатие выполнено успешно.",
            f"Метод: {self.lab4_method_var.get()}",
            f"Исходный файл: {metadata['original_name']}",
            f"Исходное расширение сохранено: {metadata['original_extension'] or '(без расширения)'}",
            f"Выходной архив: {self.lab4_archive_name}",
            f"Имя RLE-файла внутри ZIP: {metadata['payload_name']}",
            f"Размер до сжатия: {format_size(original_size)}",
            f"Размер RLE-данных: {format_size(metadata['rle_payload_size_bytes'])}",
            f"Размер после сжатия (ZIP): {format_size(zip_size)}",
            f"Количество серий RLE: {metadata['run_count']}",
            f"Коэффициент сжатия: {compression_ratio:.4f}",
            f"Изменение размера: {saved_percent:.2f}%",
        ]

        self.set_text(self.lab4_info_output, "\n".join(lines))
        self.lab4_status_var.set(f"ЛР4: архив подготовлен ({self.lab4_archive_name})")

    def save_lab4_archive(self):
        if not self.lab4_archive_bytes:
            messagebox.showwarning("Нет архива", "Сначала выполните сжатие или загрузите ZIP-архив.")
            return

        default_name = self.lab4_archive_name or "compressed.zip"
        path = filedialog.asksaveasfilename(
            title="Сохранить архив",
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[("ZIP архив", "*.zip")],
        )
        if not path:
            return

        try:
            with open(path, "wb") as dst_file:
                dst_file.write(self.lab4_archive_bytes)
        except OSError as err:
            messagebox.showerror("Ошибка записи", f"Не удалось сохранить архив:\n{err}")
            return

        current_info = self.lab4_info_output.get("1.0", "end").strip()
        extra_line = f"\nАрхив сохранен: {path}"
        self.set_text(self.lab4_info_output, (current_info + extra_line).strip())
        self.lab4_status_var.set(f"ЛР4: архив сохранен в {os.path.basename(path)}")

    def decode_lab4_archive(self):
        if not self.lab4_archive_bytes:
            messagebox.showwarning("Нет архива", "Сначала выполните сжатие или загрузите ZIP-архив.")
            return

        try:
            unpacked = unpack_archive(self.lab4_archive_bytes)
        except (ValueError, zipfile.BadZipFile, json.JSONDecodeError) as err:
            messagebox.showerror("Ошибка декодирования", f"Не удалось декодировать архив:\n{err}")
            return

        self.lab4_archive_meta = unpacked["metadata"]
        self.lab4_decoded_text = unpacked["restored_text"]
        restored_size = len(unpacked["restored_bytes"])
        original_size = self.lab4_archive_meta.get("original_size_bytes")

        match_text = "не проверено"
        current_source_name = os.path.basename(self.lab4_source_path) if self.lab4_source_path else ""
        archived_source_name = self.lab4_archive_meta.get("original_name", "")
        if self.lab4_source_path and current_source_name == archived_source_name:
            match_text = "Да" if unpacked["restored_bytes"] == self.lab4_source_bytes else "Нет"

        lines = [
            "Декодирование выполнено успешно.",
            f"Восстановленное имя: {self.lab4_archive_meta.get('original_name', '-')}",
            f"Восстановленное расширение: {self.lab4_archive_meta.get('original_extension', '-')}",
            f"Кодировка текста: {self.lab4_archive_meta.get('encoding', '-')}",
            f"Размер восстановленного файла: {format_size(restored_size)}",
            f"Размер, записанный в метаданных: {format_size(original_size)}",
            f"Количество серий RLE: {len(unpacked['runs'])}",
            f"Совпадение с загруженным исходником: {match_text}",
        ]

        self.set_text(self.lab4_result_preview, self.lab4_decoded_text)
        self.set_text(self.lab4_info_output, "\n".join(lines))
        self.lab4_status_var.set("ЛР4: текст успешно восстановлен из ZIP")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
