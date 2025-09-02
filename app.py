import gradio as gr
import torch
import torchaudio
from pprint import pformat
import numpy as np
from transformers import ClapModel, ClapProcessor

# --- Initialize Models and Processors ---
torch.set_num_threads(1)

# Silero VAD
try:
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
except Exception:
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)

(get_speech_timestamps, _, read_audio, VADIterator, *_) = vad_utils
VAD_SAMPLING_RATE = 16000
CHUNK_SIZE = 512

# CLAP
print("Loading the CLAP model and processor...")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
device = "cpu"
clap_model.to(device)
CLAP_SAMPLING_RATE = 48000
print(f"Model and processor loaded successfully. Using device: {device}")

# --- Define Class Labels ---
class_labels_human = [
    "Clear human speech", "Coughing", "Sneezing", "Yelling or Shouting",
    "Crying", "Laughing", "Sighing", "Heavy breathing or panting",
]
class_labels_background = [
    "Clear human speech", "Television in the background", "Vacuum cleaner", "Music playing",
    "Dog barking", "Baby crying", "Children playing or shouting", "Another person talking in the background",
    "Street traffic, car horn, or sirens", "Wind noise", "Keyboard typing",
    "Public announcement or PA system", "Restaurant or cafe chatter",
]
class_labels_technical = [
    "Clear human speech", "Static or crackling on the line", "Digital artifacts or garbled audio",
    "Echo or feedback", "DTMF tones (keypad pressing)", "Silence"
]
class_labels_emotion = [
    "Neutral emotion", "Happiness or joy", "Sadness", "Anger",
    "Fear or anxiety", "Surprise", "Disgust", "Confusion or uncertainty",
]

label_categories = {
    "Human Sounds": class_labels_human,
    "Background Noise": class_labels_background,
    "Technical Issues": class_labels_technical,
    "Emotional Tone": class_labels_emotion,
}

def run_clap_inference(audio_segment, class_labels, model, processor, device, target_sampling_rate):
    """Run CLAP inference on an audio segment for a specific set of labels."""
    inputs = processor(
        text=class_labels,
        audios=[audio_segment],
        return_tensors="pt",
        padding=True,
        sampling_rate=target_sampling_rate,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_audio = outputs.logits_per_audio

    probs = logits_per_audio.softmax(dim=-1).cpu().numpy()[0]
    results = {label: prob for label, prob in zip(class_labels, probs)}
    return results

def get_empty_clap_results():
    return {cat: {label: 0.0 for label in labels} for cat, labels in label_categories.items()}

def process_streaming(state, new_chunk):
    sample_rate, data = new_chunk
    
    if state is None:
        state = {
            "vad_iterator": VADIterator(vad_model),
            "timestamps": [],
            "resampler_vad": None,
            "resampler_clap": torchaudio.transforms.Resample(orig_freq=VAD_SAMPLING_RATE, new_freq=CLAP_SAMPLING_RATE),
            "audio_buffer": torch.tensor([]),
            "speech_segment": torch.tensor([]),
            "is_speaking": False,
            "clap_results": get_empty_clap_results()
        }
    
    audio_chunk = torch.from_numpy(data.astype(np.float32) / 32768.0)
    
    if sample_rate != VAD_SAMPLING_RATE:
        if state["resampler_vad"] is None:
            state["resampler_vad"] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SAMPLING_RATE)
        audio_chunk = state["resampler_vad"](audio_chunk)

    state["audio_buffer"] = torch.cat([state["audio_buffer"], audio_chunk])

    clap_outputs = list(state["clap_results"].values())

    while state["audio_buffer"].shape[0] >= CHUNK_SIZE:
        chunk_to_process = state["audio_buffer"][:CHUNK_SIZE]
        state["audio_buffer"] = state["audio_buffer"][CHUNK_SIZE:]
        
        speech_dict = state["vad_iterator"](chunk_to_process, return_seconds=True)
        if speech_dict:
            if 'start' in speech_dict:
                state["is_speaking"] = True
                state["timestamps"].append(speech_dict)
            
            if state["is_speaking"]:
                state["speech_segment"] = torch.cat([state["speech_segment"], chunk_to_process])

            if 'end' in speech_dict:
                state["is_speaking"] = False
                state["timestamps"].append(speech_dict)
                
                if state["speech_segment"].shape[0] > 0:
                    resampled_segment = state["resampler_clap"](state["speech_segment"])
                    
                    for category, labels in label_categories.items():
                        state["clap_results"][category] = run_clap_inference(
                            resampled_segment.numpy(), labels, clap_model, clap_processor, device, CLAP_SAMPLING_RATE
                        )
                    
                    clap_outputs = list(state["clap_results"].values())
                    state["speech_segment"] = torch.tensor([])
    
    return [state, pformat(state["timestamps"])] + clap_outputs

# --- Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # Live VAD + CLAP Classification
        Speak into your microphone. The system will detect speech (VAD) and classify the audio content (CLAP) across different categories.
        """
    )
    
    state = gr.State()
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Speak Here")
        vad_output = gr.Textbox(label="VAD Timestamps")

    with gr.Row():
        clap_output_human = gr.Label(label="Human Sounds", num_top_classes=3)
        clap_output_background = gr.Label(label="Background Noise", num_top_classes=3)
    
    with gr.Row():
        clap_output_technical = gr.Label(label="Technical Issues", num_top_classes=3)
        clap_output_emotion = gr.Label(label="Emotional Tone", num_top_classes=3)

    outputs_list = [state, vad_output, clap_output_human, clap_output_background, clap_output_technical, clap_output_emotion]
    audio_input.stream(
        fn=process_streaming,
        inputs=[state, audio_input],
        outputs=outputs_list
    )

iface.launch()
