import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import wavfile

# ==========================
# Configuration
# ==========================

INPUT_FOLDER       = "raw"
OUTPUT_FOLDER      = "trumpet_frequency/raw"
FRAME_RATE         = 75
MIN_DURATION       = 0.5    # seconds — discard files shorter than this after trimming
SILENCE_RUN        = 50    # consecutive zero samples required to declare silence
                            # 50 samples @ 44100Hz = ~1.1ms — safe for real audio

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================
# Trimming
# ==========================

def find_trim_end(audio, silence_run=SILENCE_RUN):
    """
    Find the last sample index before a run of >= silence_run consecutive zeros.
    Scans backwards looking for the first run of that length.
    Returns the index just after the last non-silent sample.
    """
    n = len(audio)
    run = 0
    for i in range(n - 1, -1, -1):
        if audio[i] == 0:
            run += 1
        else:
            # Found a non-zero sample — check if the run behind us was long enough
            if run >= silence_run:
                # The silence started at i+1, sound ends at i+1
                return i + 1
            else:
                # Short zero run inside real audio — reset
                run = 0
    # No long silence run found — entire file is real audio, do not trim
    return n


# ==========================
# Main
# ==========================

wav_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav")])

print(f"Processing {len(wav_files)} files from '{INPUT_FOLDER}' → '{OUTPUT_FOLDER}'")
print(f"  Trim method  : last non-zero sample preceded by >={SILENCE_RUN} consecutive zeros")
print(f"                 ({SILENCE_RUN/44100*1000:.1f}ms run — safe against zero crossings)")
print(f"  Frame rate   : {FRAME_RATE} fps")
print()

total_original_sec = 0.0
total_trimmed_sec  = 0.0
skipped            = 0
trimmed_count      = 0

for wav_file in wav_files:
    base_name = os.path.splitext(wav_file)[0]
    wav_in    = os.path.join(INPUT_FOLDER, wav_file)
    csv_in    = os.path.join(INPUT_FOLDER, base_name + ".csv")

    if not os.path.exists(csv_in):
        print(f"  [SKIP] {base_name} — no matching CSV found")
        skipped += 1
        continue

    sr, audio = wavfile.read(wav_in)
    original_duration = len(audio) / sr
    total_original_sec += original_duration

    trim_end         = find_trim_end(audio)
    trimmed_duration = trim_end / sr

    if trim_end == 0:
        print(f"  [SKIP] {base_name} — entire file is silent")
        skipped += 1
        total_original_sec -= original_duration
        continue

    if trimmed_duration < MIN_DURATION:
        print(f"  [SKIP] {base_name} — trimmed duration {trimmed_duration:.2f}s "
              f"< {MIN_DURATION}s minimum")
        skipped += 1
        total_original_sec -= original_duration
        continue

    audio_trimmed = audio[:trim_end]

    df         = pd.read_csv(csv_in)
    n_frames   = int(np.floor(trimmed_duration * FRAME_RATE))
    n_frames   = min(n_frames, len(df))
    df_trimmed = df.iloc[:n_frames]

    wavfile.write(os.path.join(OUTPUT_FOLDER, wav_file), sr, audio_trimmed)
    df_trimmed.to_csv(os.path.join(OUTPUT_FOLDER, base_name + ".csv"),
                      index=False, float_format="%.6f")

    total_trimmed_sec += trimmed_duration
    trimmed_count     += 1

    removed_sec = original_duration - trimmed_duration
    print(f"  {base_name:<20}  {original_duration:.1f}s → {trimmed_duration:.2f}s  "
          f"(removed {removed_sec:.1f}s,  {n_frames} frames kept)")

# Copy parameters.json if present
params_src = os.path.join(INPUT_FOLDER, "parameters.json")
if os.path.exists(params_src):
    shutil.copy(params_src, os.path.join(OUTPUT_FOLDER, "parameters.json"))
    print(f"\n  Copied parameters.json → {OUTPUT_FOLDER}/")

# ==========================
# Sanity checks
# ==========================

print()
print("=" * 60)
print("SANITY CHECK")
print("=" * 60)
print(f"  Files processed       : {trimmed_count}")
print(f"  Files skipped         : {skipped}")
print(f"  Total original audio  : {total_original_sec:.1f}s  "
      f"({total_original_sec/60:.2f} min)")
print(f"  Total trimmed audio   : {total_trimmed_sec:.1f}s  "
      f"({total_trimmed_sec/60:.2f} min)")
print(f"  Audio removed         : {total_original_sec - total_trimmed_sec:.1f}s  "
      f"({100*(1 - total_trimmed_sec/max(total_original_sec,1)):.1f}%)")
print(f"  Average file duration : {total_trimmed_sec/max(trimmed_count,1):.2f}s")

print()
print("  Frame alignment check (first 5 files):")
out_wavs = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".wav")])[:5]
for wf in out_wavs:
    bn        = os.path.splitext(wf)[0]
    sr_, sig  = wavfile.read(os.path.join(OUTPUT_FOLDER, wf))
    dur       = len(sig) / sr_
    df_       = pd.read_csv(os.path.join(OUTPUT_FOLDER, bn + ".csv"))
    expected  = int(np.floor(dur * FRAME_RATE))
    match     = "OK" if abs(len(df_) - expected) <= 1 else "MISMATCH"
    print(f"    {bn:<20}  audio={dur:.2f}s  "
          f"csv_rows={len(df_)}  expected~{expected}  [{match}]")

print()
print("Done.")