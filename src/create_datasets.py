import os
import json
import shutil
import numpy as np
import pandas as pd

# ==========================
# Configuration
# ==========================

INPUT_FOLDER       = "trumpet_frequency/raw"
N_FOURIER          = 4

# Output folders: trumpet_<encoding>/raw
ENCODINGS = ["class", "log_normalised", "sine_cosine", "fourier"]
OUTPUT_FOLDERS = {enc: f"trumpet_{enc}/raw" for enc in ENCODINGS}

for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# Standard chromatic frequencies (equal temperament, A4=440Hz)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)

CHROMATIC = [(midi_to_freq(m), NOTE_NAMES[m % 12], m // 12 - 1, m)
             for m in range(0, 128)]

def nearest_note(freq):
    diffs = [abs(freq - c[0]) for c in CHROMATIC]
    idx   = int(np.argmin(diffs))
    entry = CHROMATIC[idx]
    return entry[1], entry[2], entry[3], entry[0]  # name, octave, midi, freq


# ==========================
# Encoding functions
# ==========================

def encode_log_normalised(freq, f_min, f_max):
    return (np.log2(freq) - np.log2(f_min)) / (np.log2(f_max) - np.log2(f_min))

def encode_sine_cosine(freq, f_min, f_max):
    t     = encode_log_normalised(freq, f_min, f_max)
    angle = 2 * np.pi * t
    return np.sin(angle), np.cos(angle)

def encode_fourier(freq, f_min, f_max, n=N_FOURIER):
    t        = encode_log_normalised(freq, f_min, f_max)
    features = []
    for k in range(1, n + 1):
        features.append(np.sin(2 * np.pi * k * t))
        features.append(np.cos(2 * np.pi * k * t))
    return features


# ==========================
# Main
# ==========================

csv_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")])
if not csv_files:
    raise ValueError(f"No CSV files found in {INPUT_FOLDER}")

# --- Pass 1: collect all unique frequencies ---
all_freqs = set()
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(INPUT_FOLDER, csv_file))
    if "pitch" not in df.columns:
        raise ValueError(f"No 'pitch' column in {csv_file}")
    all_freqs.update(df["pitch"].unique().tolist())

all_freqs      = sorted(all_freqs)
f_min, f_max   = min(all_freqs), max(all_freqs)
freq_to_note   = {f: nearest_note(f) for f in all_freqs}
unique_classes = sorted(set(
    f"{n}{o}" for f in all_freqs
    for n, o, _, _ in [freq_to_note[f]]
))

print(f"Input        : {INPUT_FOLDER}")
print(f"CSV files    : {len(csv_files)}")
print(f"Unique freqs : {all_freqs}")
print(f"Freq range   : {f_min:.2f} Hz – {f_max:.2f} Hz")
print(f"Note classes : {unique_classes}")
print()

# --- Pass 2: process each file ---
for csv_file in csv_files:
    stem   = os.path.splitext(csv_file)[0]
    df     = pd.read_csv(os.path.join(INPUT_FOLDER, csv_file))
    freqs  = df["pitch"].values

    # Copy audio file to all output folders
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio_src = os.path.join(INPUT_FOLDER, stem + ext)
        if os.path.exists(audio_src):
            for folder in OUTPUT_FOLDERS.values():
                shutil.copy2(audio_src, os.path.join(folder, stem + ext))
            break

    # Class (one-hot label)
    pd.DataFrame({
        "pitch_class": [f"{freq_to_note[f][0]}{freq_to_note[f][1]}" for f in freqs],
    }).to_csv(os.path.join(OUTPUT_FOLDERS["class"], csv_file), index=False)

    # Log-normalised
    pd.DataFrame({
        "pitch_log_norm": [round(encode_log_normalised(f, f_min, f_max), 6) for f in freqs],
    }).to_csv(os.path.join(OUTPUT_FOLDERS["log_normalised"], csv_file), index=False)

    # Sine / cosine
    sins, coss = zip(*[encode_sine_cosine(f, f_min, f_max) for f in freqs])
    pd.DataFrame({
        "pitch_sin": [round(s, 6) for s in sins],
        "pitch_cos": [round(c, 6) for c in coss],
    }).to_csv(os.path.join(OUTPUT_FOLDERS["sine_cosine"], csv_file), index=False)

    # Fourier
    fourier_rows = [encode_fourier(f, f_min, f_max) for f in freqs]
    fourier_cols = {}
    for k in range(1, N_FOURIER + 1):
        fourier_cols[f"pitch_sin_{k}"] = [round(r[(k-1)*2],   6) for r in fourier_rows]
        fourier_cols[f"pitch_cos_{k}"] = [round(r[(k-1)*2+1], 6) for r in fourier_rows]
    pd.DataFrame(fourier_cols).to_csv(
        os.path.join(OUTPUT_FOLDERS["fourier"], csv_file), index=False
    )

# --- Load original parameters.json if present ---
orig_params = {}
params_src  = os.path.join(INPUT_FOLDER, "parameters.json")
if os.path.exists(params_src):
    with open(params_src) as f:
        orig_params = json.load(f)

# Remove any existing pitch parameter from originals
non_pitch_params = {k: v for k, v in orig_params.items() if v.get("name") != "pitch"}

def save_params(folder, pitch_descriptors):
    """Write parameters.json preserving non-pitch params and appending pitch ones."""
    params   = dict(non_pitch_params)
    base_idx = len(params) + 1
    for offset, descriptor in enumerate(pitch_descriptors):
        params[f"parameter_{base_idx + offset}"] = descriptor
    with open(os.path.join(folder, "parameters.json"), "w") as f:
        json.dump(params, f, indent=4)

save_params(OUTPUT_FOLDERS["class"], [{
    "name":    "pitch_class",
    "type":    "class",
    "classes": unique_classes,
}])

save_params(OUTPUT_FOLDERS["log_normalised"], [{
    "name": "pitch_log_norm",
    "type": "continuous",
    "unit": "log_normalised",
    "min":  0.0,
    "max":  1.0,
}])

save_params(OUTPUT_FOLDERS["sine_cosine"], [
    {"name": "pitch_sin", "type": "continuous", "unit": "sine",   "min": -1.0, "max": 1.0},
    {"name": "pitch_cos", "type": "continuous", "unit": "cosine", "min": -1.0, "max": 1.0},
])

fourier_descriptors = []
for k in range(1, N_FOURIER + 1):
    fourier_descriptors.append(
        {"name": f"pitch_sin_{k}", "type": "continuous",
         "unit": f"sine_component_{k}", "min": -1.0, "max": 1.0}
    )
    fourier_descriptors.append(
        {"name": f"pitch_cos_{k}", "type": "continuous",
         "unit": f"cosine_component_{k}", "min": -1.0, "max": 1.0}
    )
save_params(OUTPUT_FOLDERS["fourier"], fourier_descriptors)

print("Done. Output folders:")
for enc, folder in OUTPUT_FOLDERS.items():
    print(f"  {folder}/   ({enc})")