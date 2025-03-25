# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


MODEL = None  # Last used model
MODEL_PATH = "checkpoints/YodelGen"

SPACE_ID = os.environ.get("SPACE_ID", "")
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None

# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def load_model(version: str = "facebook/musicgen-melody"):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        # TODO: change device
        MODEL = MusicGen.get_pretrained(version, device="cpu")


def _do_predictions(
    texts,
    melodies,
    duration,
    progress=False,
    gradio_progress=None,
    **gen_kwargs,
):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print(
        "new batch",
        len(texts),
        texts,
        [None if m is None else (m[0], m[1].shape) for m in melodies],
    )
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = (
                melody[0],
                torch.from_numpy(melody[1]).to(MODEL.device).float().t(),
            )
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., : int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=False,
            )
        else:
            outputs = MODEL.generate(
                texts,
                progress=progress,
                return_tokens=False,
            )
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name,
                output,
                MODEL.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_wavs


def predict_full(
    model_path: str,
    text: str,
    melody,
    duration,
    topk,
    topp,
    temperature,
    cfg_coef,
    progress=gr.Progress(),
):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")

    model_path = model_path.strip()
    if not Path(model_path).exists():
        raise gr.Error(f"Model path {model_path} doesn't exist.")
    if not Path(model_path).is_dir():
        raise gr.Error(
            f"Model path {model_path} must be a folder containing "
            "state_dict.bin and compression_state_dict_.bin."
        )

    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model_path)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    MODEL.set_custom_progress_callback(_progress)

    wavs = _do_predictions(
        [text],
        [melody],
        duration,
        progress=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        gradio_progress=progress,
    )
    # TODO: remove the "videos"
    return wavs[0]


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # YodelGen
            Top-k: Top-k is a parameter used in text generation models, including music generation models. It determines the number of most likely next tokens to consider at each step of the generation process. The model ranks all possible tokens based on their predicted probabilities, and then selects the top-k tokens from the ranked list. The model then samples from this reduced set of tokens to determine the next token in the generated sequence. A smaller value of k results in a more focused and deterministic output, while a larger value of k allows for more diversity in the generated music.

            Top-p (or nucleus sampling): Top-p, also known as nucleus sampling or probabilistic sampling, is another method used for token selection during text generation. Instead of specifying a fixed number like top-k, top-p considers the cumulative probability distribution of the ranked tokens. It selects the smallest possible set of tokens whose cumulative probability exceeds a certain threshold (usually denoted as p). The model then samples from this set to choose the next token. This approach ensures that the generated output maintains a balance between diversity and coherence, as it allows for a varying number of tokens to be considered based on their probabilities.

            Temperature: Temperature is a parameter that controls the randomness of the generated output. It is applied during the sampling process, where a higher temperature value results in more random and diverse outputs, while a lower temperature value leads to more deterministic and focused outputs. In the context of music generation, a higher temperature can introduce more variability and creativity into the generated music, but it may also lead to less coherent or structured compositions. On the other hand, a lower temperature can produce more repetitive and predictable music.

            Classifier-Free Guidance: Classifier-Free Guidance refers to a technique used in some music generation models where a separate classifier network is trained to provide guidance or control over the generated music. This classifier is trained on labeled data to recognize specific musical characteristics or styles. During the generation process, the output of the generator model is evaluated by the classifier, and the generator is encouraged to produce music that aligns with the desired characteristics or style. This approach allows for more fine-grained control over the generated music, enabling users to specify certain attributes they want the model to capture.

            """
        )
        with gr.Row():
            with gr.Row():
                with gr.Column():
                    text = gr.Text(
                        label="Input Text",
                        placeholder="A swiss alpine yodeling song from ...",
                        interactive=True,
                    )
                    with gr.Column():
                        radio = gr.Radio(
                            ["file", "mic"],
                            value="file",
                            label="Condition on a melody (optional) File or Mic",
                        )
                        melody = gr.Audio(
                            sources=["upload"],
                            type="numpy",
                            label="File",
                            interactive=True,
                            elem_id="melody-input",
                        )
            with gr.Column():
                duration = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=10,
                    label="Duration",
                    interactive=True,
                )
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(
                        label="Temperature", value=1.0, interactive=True
                    )
                    cfg_coef = gr.Number(
                        label="Classifier Free Guidance", value=3.0, interactive=True
                    )
        with gr.Row():
            submit = gr.Button("Submit")
            # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
            _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
        with gr.Row():
            audio_output = gr.Audio(label="Generated Music (wav)", type="filepath")

        model_path = gr.State(value=MODEL_PATH)
        submit.click(
            predict_full,
            inputs=[
                model_path,
                text,
                melody,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
            ],
            outputs=[audio_output],
            queue=True,
        )
        radio.change(
            toggle_audio_src, radio, [melody], queue=False, show_progress=False
        )

        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=0,
        help="Port to run the server listener on",
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs["server_name"] = args.listen

    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser
    if args.share:
        launch_kwargs["share"] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    ui_full(launch_kwargs)
