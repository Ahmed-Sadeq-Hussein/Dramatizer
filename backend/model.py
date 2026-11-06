"""model.py | Every model Dramatizer uses, in one place.

Other files never load a model themselves. They ask this file:

    from backend import model

    cosyvoice = model.voice_model()        # speaks in a cloned voice
    whisper_model = model.recogniser()     # turns speech into text (for WER)
    encoder = model.speaker_encoder()      # turns a voice into numbers (similarity)
    squim = model.quality_model()          # estimates how clean a recording is
    deepfilter = model.denoiser()          # removes background noise from speech
    llm = model.emphasis_model()           # a small LLM that scores how much each word is stressed

Why one file: to change a model (a bigger Whisper, another speaker encoder, a
new TTS model) you edit .env, or the one loader below, and nothing else.

Each model is loaded the first time it is asked for and then kept, so asking
again is free. Loading is thread-safe: the web page and a background job can
ask at the same time. Importing this file loads nothing and imports no model
library, so it is fast.

Settings, from .env (a missing one falls back to the default shown):

    DEVICE                cuda or cpu (cuda falls back to cpu if there is no GPU)
    VOICE_MODEL_DIR       the CosyVoice repository checkout (its code)
    COSYVOICE_MODEL_DIR   the CosyVoice 3 weights, Fun-CosyVoice3-0.5B
    WER_MODEL             whisper-base   (whisper-tiny, whisper-small, ...)
    SIMILARITY_MODEL      resemblyzer
    PERQ_MODEL            squim
    DENOISE_MODEL         deepfilternet3
    LLM_LIBRARY_DIR       the folder with one sub-folder per downloaded LLM
    EMPHASIS_MODEL        Qwen3-4B       (Qwen3-1.7B, gemma-4-E2B-it, gemma-4-E4B-it)
"""
import logging
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

# The project's root: .env, output/, measurements/ and certs/ stay there, one per machine
PROJECT_DIR = Path(__file__).resolve().parent.parent
# The data that is committed with the code: the manifests and the emphasis test set
DATA_DIR = Path(__file__).resolve().parent / "data"

# The settings of this machine
load_dotenv(PROJECT_DIR / ".env")

# CosyVoice turns on debug logging for everything, and numba (used inside
# librosa) then prints pages of compiler internals. Keep those quiet.
for noisy in ("numba", "matplotlib", "PIL", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

_loaded = {}
_lock = threading.RLock()


def setting(name, default=None):
    """A value from .env (or the environment). Without a default, it must exist."""
    value = os.environ.get(name, default)

    if value is None or not str(value).strip():
        raise KeyError(f"{name} is not set. Add it to .env (see requirements.txt and README.md).")

    return str(value).strip()


def device():
    """cuda if .env asks for it and a GPU is there, otherwise cpu."""
    wanted = setting("DEVICE", "cpu").lower()

    if wanted == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cpu"


def _load_once(key, loader):
    """Run loader the first time `key` is asked for, and keep what it returns."""
    with _lock:
        if key not in _loaded:
            _loaded[key] = loader()
        return _loaded[key]


def is_loaded(key):
    """True once the model is in memory. Keys: voice, recogniser, speaker, quality, denoiser, emphasis:<name>."""
    return key in _loaded


def unload(key):
    """Forget a loaded model and free its GPU memory (e.g. to compare LLMs one after another)."""
    with _lock:
        if _loaded.pop(key, None) is not None:
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass


# --------------------------------------------------------------------------- #
# The voice model: CosyVoice 3
# --------------------------------------------------------------------------- #


def _wetext_from_cache():
    """
    CosyVoice's text normaliser (wetext: numbers, dates and the like into words)
    asks ModelScope for its files every time it loads. ModelScope now refuses part
    of that request (403), so each load spent about a minute retrying and then went
    on with no normaliser at all. Once the files are on this PC, use them as they are;
    the first time, they are still downloaded.
    """
    try:
        import wetext.wetext as wetext
    except ImportError:
        return

    download = wetext.snapshot_download
    if getattr(download, "from_cache", False):
        return

    def from_cache(model_id, *args, **kwargs):
        try:
            folder = download(model_id, *args, local_files_only=True, **kwargs)
            if all((Path(folder) / lang / "tn" / "tagger.fst").is_file() for lang in ("en", "zh")):
                return folder
        except Exception:
            pass
        return download(model_id, *args, **kwargs)

    from_cache.from_cache = True
    wetext.snapshot_download = from_cache


def voice_model():
    """
    CosyVoice 3 with the Fun-CosyVoice3-0.5B-2512 weights. Takes about a minute
    the first time. Its methods (inference_zero_shot, ...) and sample_rate are
    CosyVoice's own; Voice_model.generate_speech is the way to use them.
    """

    def load():
        code = Path(setting("VOICE_MODEL_DIR"))
        weights = Path(setting("COSYVOICE_MODEL_DIR"))

        if not weights.is_dir():
            raise FileNotFoundError(f"CosyVoice weights not found: {weights} (COSYVOICE_MODEL_DIR in .env)")

        # CosyVoice is not installed with pip: its code is imported from the
        # checkout, together with Matcha-TTS, which it ships as a submodule
        for folder in (code, code / "third_party" / "Matcha-TTS"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))

        from cosyvoice.cli.cosyvoice import AutoModel

        _wetext_from_cache()
        print("Loading CosyVoice 3...")
        return AutoModel(model_dir=str(weights))

    return _load_once("voice", load)


# --------------------------------------------------------------------------- #
# The speech recogniser: Whisper (for the word error rate)
# --------------------------------------------------------------------------- #


def recogniser():
    """A Whisper model. WER_MODEL=whisper-base picks the size (tiny, base, small, medium, ...)."""

    def load():
        name = setting("WER_MODEL", "whisper-base").lower()
        family, _, size = name.partition("-")

        if family != "whisper" or not size:
            raise ValueError(f"Unknown WER_MODEL {name!r}. Use whisper-<size>, e.g. whisper-base.")

        import whisper

        print(f"Loading Whisper ({size})...")
        return whisper.load_model(size, device=device())

    return _load_once("recogniser", load)


# --------------------------------------------------------------------------- #
# The speaker encoder: Resemblyzer (for the speaker similarity)
# --------------------------------------------------------------------------- #


def speaker_encoder():
    """A speaker encoder: embed_utterance(wav) -> 256 numbers that describe a voice."""

    def load():
        name = setting("SIMILARITY_MODEL", "resemblyzer").lower()

        if name != "resemblyzer":
            raise ValueError(f"Unknown SIMILARITY_MODEL {name!r}. Supported: resemblyzer.")

        from resemblyzer import VoiceEncoder

        print("Loading the speaker encoder (Resemblyzer)...")
        return VoiceEncoder(device=device(), verbose=False)

    return _load_once("speaker", load)


# --------------------------------------------------------------------------- #
# The quality model: SQUIM (for the perceptual quality)
# --------------------------------------------------------------------------- #


def quality_model():
    """
    torchaudio's SQUIM objective model. From one 16 kHz recording it estimates
    three scores with no reference: STOI, PESQ and SI-SDR (see PERQ.py).
    """

    def load():
        name = setting("PERQ_MODEL", "squim").lower()

        if name != "squim":
            raise ValueError(f"Unknown PERQ_MODEL {name!r}. Supported: squim.")

        from torchaudio.pipelines import SQUIM_OBJECTIVE

        print("Loading the quality model (SQUIM)...")
        return SQUIM_OBJECTIVE.get_model().to(device()).eval()

    return _load_once("quality", load)


# --------------------------------------------------------------------------- #
# The denoiser: DeepFilterNet 3 (for Clean_Audio_Distortion.py)
# --------------------------------------------------------------------------- #


def denoiser():
    """
    DeepFilterNet 3: removes background noise from speech, at 48 kHz.
    Returns (network, df_state); Clean_Audio_Distortion.py is the way to use them.
    The weights (about 8 MB) are downloaded from the DeepFilterNet GitHub
    releases the first time and cached.
    """

    def load():
        name = setting("DENOISE_MODEL", "deepfilternet3").lower()

        if name != "deepfilternet3":
            raise ValueError(f"Unknown DENOISE_MODEL {name!r}. Supported: deepfilternet3.")

        import warnings

        # DeepFilterNet imports a torchaudio path that moved; it still works
        warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
        try:
            from df.enhance import init_df
        except ImportError as error:
            raise ImportError(
                "DeepFilterNet is not installed in this environment. Run: "
                "pip install --no-deps deepfilternet==0.5.6 deepfilterlib==0.5.6 "
                "and pip install appdirs==1.4.4 loguru==0.7.3 (step 3c in requirements.txt)"
            ) from error

        print("Loading the denoiser (DeepFilterNet 3)...")
        network, df_state, _ = init_df(model_base_dir="DeepFilterNet3", log_level="WARNING", log_file=None)
        return network, df_state

    return _load_once("denoiser", load)


# --------------------------------------------------------------------------- #
# The emphasis LLM: a small language model (for emphasis.py)
# --------------------------------------------------------------------------- #


def emphasis_model(name=None):
    """
    A small instruction-tuned LLM from LLM_LIBRARY_DIR, loaded in 4-bit so it
    fits on the GPU next to CosyVoice. Returns (tokenizer, model).
    name: a folder in LLM_LIBRARY_DIR; default EMPHASIS_MODEL from .env.

    Needs transformers, accelerate and bitsandbytes. Gemma 4 needs a newer
    transformers than CosyVoice allows, so it runs in its own environment
    (see requirements.txt).
    """
    name = name or setting("EMPHASIS_MODEL", "Qwen3-4B")

    def load():
        folder = Path(setting("LLM_LIBRARY_DIR")) / name
        if not folder.is_dir():
            raise FileNotFoundError(f"LLM not found: {folder} (LLM_LIBRARY_DIR and EMPHASIS_MODEL in .env)")

        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading the emphasis LLM ({name}, 4-bit)...")
        gpu = device()
        config = AutoConfig.from_pretrained(folder)

        # Gemma 3n / Gemma 4 carry a per-layer embedding table bigger than the
        # model's layers (2.35 B numbers in E2B), plus audio and vision parts
        # that text never uses. 4-bit does not shrink embeddings, so together
        # they overflow a 10 GB card. Google meant the table to live in CPU
        # memory (it is a lookup, not computation), so it and the unused parts
        # go to the CPU, unquantized; everything else goes to the GPU in 4-bit.
        cpu_parts = ("embed_tokens_per_layer", "audio_tower", "vision_tower", "embed_audio", "embed_vision")
        with init_empty_weights():
            names = [n for n, _ in AutoModelForCausalLM.from_config(config).named_modules()]
        offload = [n for n in names if n.split(".")[-1] in cpu_parts]

        device_map = _split_map(names, offload) if offload and gpu == "cuda" else gpu

        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          llm_int8_skip_modules=[*cpu_parts, "lm_head"],
                                          llm_int8_enable_fp32_cpu_offload=bool(offload))
        tokenizer = AutoTokenizer.from_pretrained(folder)
        llm = AutoModelForCausalLM.from_pretrained(folder, quantization_config=quantization,
                                                   device_map=device_map, dtype=torch.bfloat16)
        if offload and gpu == "cuda":
            _lookup_on_cpu(llm, folder, [n for n in offload if n.endswith("embed_tokens_per_layer")])
        return tokenizer, llm.eval()

    return _load_once(f"emphasis:{name}", load)


def _lookup_on_cpu(llm, folder, modules):
    """
    Make offloaded embedding tables look rows up ON the CPU. By default an
    offloaded module is copied to the GPU for every call, which for Gemma's
    4.7 GB per-layer table means every generated token (about 2 tokens/s).
    Here the table stays in CPU memory with its real weights, and only the
    rows asked for travel to the GPU.
    """
    import json

    import torch
    from accelerate.hooks import remove_hook_from_module
    from safetensors import safe_open

    index = folder / "model.safetensors.index.json"
    files = json.loads(index.read_text())["weight_map"] if index.exists() else None

    for name in modules:
        table = llm.get_submodule(name)
        remove_hook_from_module(table)
        table.to_empty(device="cpu")

        key = f"{name}.weight"
        with safe_open(str(folder / (files[key] if files else "model.safetensors")), "pt", device="cpu") as f:
            table.weight.data = f.get_tensor(key).to(torch.bfloat16)
        if hasattr(table, "scalar_embed_scale"):
            table.embed_scale = torch.tensor(table.scalar_embed_scale)

        lookup = table.forward

        def forward(input_ids, lookup=lookup):
            return lookup(input_ids.to("cpu")).to(input_ids.device)

        table.forward = forward


def _split_map(names, offload):
    """A device_map: the `offload` modules on the CPU, every other branch on GPU 0, as high in the tree as possible."""
    offload = set(offload)
    device_map = {}

    def children(parent):
        prefix = f"{parent}." if parent else ""
        return [n for n in names if n.startswith(prefix) and n != parent and "." not in n[len(prefix):]]

    def visit(module):
        if module in offload:
            device_map[module] = "cpu"
        elif module and not any(o.startswith(f"{module}.") for o in offload):
            device_map[module] = 0
        else:
            for child in children(module):
                visit(child)

    visit("")
    return device_map
