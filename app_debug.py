import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import ClapModel, ClapProcessor
from pprint import pprint

# --- Initialize Models and Processors ---
torch.set_num_threads(1)

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

# --- Pre-compute Text Embeddings ---
print("Pre-computing text embeddings for all labels...")
text_embeddings_per_category = {}
with torch.no_grad():
    for category, labels in label_categories.items():
        inputs = clap_processor(text=labels, return_tensors="pt", padding=True).to(device)
        text_embeddings = clap_model.get_text_features(**inputs)
        text_embeddings_per_category[category] = text_embeddings
print("Text embeddings pre-computed.")


def process_recording(audio_path):
    """
    Processes a single audio recording file by comparing its embedding against
    pre-computed text embeddings.
    """
    if audio_path is None:
        empty_results = [{label: 0.0 for label in labels} for labels in label_categories.values()]
        return empty_results

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=CLAP_SAMPLING_RATE)
        resampled_waveform = resampler(waveform).mean(dim=0)

        print(f"\n--- Processing recording: {audio_path} ---")
        print(f"Waveform duration: {resampled_waveform.shape[0] / CLAP_SAMPLING_RATE:.2f} seconds")

        # 1. Get audio embeddings
        audio_inputs = clap_processor(audios=[resampled_waveform.numpy()], return_tensors="pt", sampling_rate=CLAP_SAMPLING_RATE).to(device)
        with torch.no_grad():
            audio_embeddings = clap_model.get_audio_features(**audio_inputs)

        final_results = []
        # 2. Compare with pre-computed text embeddings for each category
        for category, labels in label_categories.items():
            text_embeddings = text_embeddings_per_category[category]
            
            # 3. Calculate similarity and probabilities
            with torch.no_grad():
                # Note: We use the model's learned logit scale for accurate similarity
                logits = torch.matmul(audio_embeddings, text_embeddings.T) * clap_model.logit_scale_a.exp()
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            cat_results = {label: prob for label, prob in zip(labels, probs)}
            final_results.append(cat_results)
            
            print(f"\n--- Category: {category} ---")
            sorted_results = sorted(cat_results.items(), key=lambda item: item[1], reverse=True)
            pprint(sorted_results)

        print("--- END OF PROCESSING ---")
        return final_results

    except Exception as e:
        print(f"Error processing audio file: {e}")
        empty_results = [{label: 0.0 for label in labels} for labels in label_categories.values()]
        return empty_results


# --- Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # Manual Audio Classification with CLAP
        Record audio using the microphone or upload a file. The classification will run automatically when you stop recording or upload a file.
        The results from the CLAP model will be displayed below for different categories.
        """
    )
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio")

    with gr.Row():
        clap_output_human = gr.Label(label="Human Sounds", num_top_classes=3)
        clap_output_background = gr.Label(label="Background Noise", num_top_classes=3)
    
    with gr.Row():
        clap_output_technical = gr.Label(label="Technical Issues", num_top_classes=3)
        clap_output_emotion = gr.Label(label="Emotional Tone", num_top_classes=3)

    outputs_list = [
        clap_output_human, clap_output_background, 
        clap_output_technical, clap_output_emotion
    ]
    
    audio_input.change(
        fn=process_recording,
        inputs=audio_input,
        outputs=outputs_list
    )

iface.launch(debug=True)