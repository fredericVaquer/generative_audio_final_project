# RNeNcodec: Generative Audio Conditioning Analysis

This project explores the impact of various latent representations on the frequency fidelity and controllability of an RNN-based audio generator. By leveraging the **EnCodec** neural audio codec, we evaluate how different "mathematical languages" (Linear, Logarithmic, Trigonometric, and Categorical) affect the model's ability to follow precise pitch instructions.

---

## 🚀 Overview
The core of this repository is an **RNN-based Generator** trained to predict EnCodec latent frames. The project includes a real-time interactive GUI and an offline inference pipeline for high-precision frequency analysis.

### Key Objectives:
* **Fidelity:** How accurately does the synthesized audio match the input Hz?
* **Smoothness:** Can the model perform continuous frequency slides (glissandi)?
* **Representation Bias:** How do different encoding geometries (circular vs. linear vs. categorical) change the acoustic output?

---

## 🎹 Conditioning Representations
We compare five distinct methods for injecting frequency information into the model:

| Representation | Dimensions | Description |
| :--- | :--- | :--- |
| **Linear Frequency** | 1 | Raw Hz normalized between [234, 448]. |
| **Log-Normal** | 1 | Frequency mapped to log-space for equal-tempered musicality. |
| **Sine-Cosine** | 2 | A 2D circular embedding: $(\sin(2\pi t), \cos(2\pi t))$. |
| **Fourier Series** | 8 | A multi-harmonic stack of $N=4$ sine-cosine pairs. |
| **Probabilistic Class**| 11 | One-hot/Softmax distribution across 11 chromatic pitch classes. |

---

## 🛠 Features
* **Real-Time Synthesis:** Interactive Jupyter/IPython GUI with Hz-calibrated sliders.
* **Offline Rendering:** Scripted generation of complex pitch slides with manual RNN state management.
* **Fidelity Analysis:** Automated plotting tool comparing **Target Hz** vs. **pYIN Detected Hz** on a spectrogram background.
* **Synchronized Plotting:** Automatic audio truncation to ensure conditioning and analysis match exactly in time.

---

## 📦 Installation
```bash
git clone [https://github.com/fredericVaquer/generative_audio_final_project.git](https://github.com/fredericVaquer/generative_audio_final_project.git)
cd generative_audio_final_project

# Install core dependencies
pip install -r requirements.txt
```

## 📂 Project Structure

The repository is organized into four main stages: Data Exploration, Pre-processing, Model Training, and Inference. There are other folders, but I did minimal modifications and were already provided in other repos (EnCodec exploration and RNeNcodec training)

```text
root
├── audio_examples/                # Set of audio glissando examples generated for each representation
├── latent_visualization/
│   └── latent_exploration.ipynb   # Analysis of EnCodec pitch distribution and latent codebook 
├── src/                           # Core pre-processing scripts
│   ├── trim_dataset.py            # Trims raw audio files to extract active signal (Trumpet) and creates the frequency representation
│   └── create_datasets.py         # Generates the other 4 different conditioning representations based on the frequency one
└── rnencodec_notebooks/           # Main workflow pipeline
    ├── 1_dataset.ipynb            # Formatting data for RNN training
    ├── 2_train.ipynb              # Model training and hyperparameter configuration
    ├── 3_inference.ipynb          # Real-time GUI and interactive parameter exploration
    ├── 4_sonic_examples.ipynb     # Offline rendering of glissandi and fidelity analysis
    └── output/                    # (Created after training) Model checkpoints and configs
```