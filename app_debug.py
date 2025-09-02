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

class_labels_all = ['Speech', 'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue', 'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell', 'Battle cry', 'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter', 'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing', 'Baby cry, infant cry', 'Whimper', 'Wail, moan', 'Sigh', 'Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Rapping', 'Humming', 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 'Gasp', 'Pant', 'Snort', 'Cough', 'Throat clearing', 'Sneeze', 'Sniff', 'Run', 'Shuffle', 'Walk, footsteps', 'Chewing, mastication', 'Biting', 'Gargling', 'Stomach rumble', 'Burping, eructation', 'Hiccup', 'Fart', 'Hands', 'Finger snapping', 'Clapping', 'Heart sounds, heartbeat', 'Heart murmur', 'Cheering', 'Applause', 'Chatter', 'Crowd', 'Hubbub, speech noise, speech babble', 'Children playing', 'Animal', 'Domestic animals, pets', 'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper (dog)', 'Cat', 'Purr', 'Meow', 'Hiss', 'Caterwaul', 'Livestock, farm animals, working animals', 'Horse', 'Clip-clop', 'Neigh, whinny', 'Cattle, bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'Chicken, rooster', 'Cluck', 'Crowing, cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk', 'Wild animals', 'Roaring cats (lions, tigers)', 'Roar', 'Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet', 'Squawk', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot', 'Bird flight, flapping wings', 'Canidae, dogs, wolves', 'Rodents, rats, mice', 'Mouse', 'Patter', 'Insect', 'Cricket', 'Mosquito', 'Fly, housefly', 'Buzz', 'Bee, wasp, etc.', 'Frog', 'Croak', 'Snake', 'Rattle', 'Whale vocalization', 'Music', 'Musical instrument', 'Plucked string instrument', 'Guitar', 'Electric guitar', 'Bass guitar', 'Acoustic guitar', 'Steel guitar, slide guitar', 'Tapping (guitar technique)', 'Strum', 'Banjo', 'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Keyboard (musical)', 'Piano', 'Electric piano', 'Organ', 'Electronic organ', 'Hammond organ', 'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Drum kit', 'Drum machine', 'Drum', 'Snare drum', 'Rimshot', 'Drum roll', 'Bass drum', 'Timpani', 'Tabla', 'Cymbal', 'Hi-hat', 'Wood block', 'Tambourine', 'Rattle (instrument)', 'Maraca', 'Gong', 'Tubular bells', 'Mallet percussion', 'Marimba, xylophone', 'Glockenspiel', 'Vibraphone', 'Steelpan', 'Orchestra', 'Brass instrument', 'French horn', 'Trumpet', 'Trombone', 'Bowed string instrument', 'String section', 'Violin, fiddle', 'Pizzicato', 'Cello', 'Double bass', 'Wind instrument, woodwind instrument', 'Flute', 'Saxophone', 'Clarinet', 'Harp', 'Bell', 'Church bell', 'Jingle bell', 'Bicycle bell', 'Tuning fork', 'Chime', 'Wind chime', 'Change ringing (campanology)', 'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 'Theremin', 'Singing bowl', 'Scratching (performance technique)', 'Pop music', 'Hip hop music', 'Beatboxing', 'Rock music', 'Heavy metal', 'Punk rock', 'Grunge', 'Progressive rock', 'Rock and roll', 'Psychedelic rock', 'Rhythm and blues', 'Soul music', 'Reggae', 'Country', 'Swing music', 'Bluegrass', 'Funk', 'Folk music', 'Middle Eastern music', 'Jazz', 'Disco', 'Classical music', 'Opera', 'Electronic music', 'House music', 'Techno', 'Dubstep', 'Drum and bass', 'Electronica', 'Electronic dance music', 'Ambient music', 'Trance music', 'Music of Latin America', 'Salsa music', 'Flamenco', 'Blues', 'Music for children', 'New-age music', 'Vocal music', 'A capella', 'Music of Africa', 'Afrobeat', 'Christian music', 'Gospel music', 'Music of Asia', 'Carnatic music', 'Music of Bollywood', 'Ska', 'Traditional music', 'Independent music', 'Song', 'Background music', 'Theme music', 'Jingle (music)', 'Soundtrack music', 'Lullaby', 'Video game music', 'Christmas music', 'Dance music', 'Wedding music', 'Happy music', 'Funny music', 'Sad music', 'Tender music', 'Exciting music', 'Angry music', 'Scary music', 'Wind', 'Rustling leaves', 'Wind noise (microphone)', 'Thunderstorm', 'Thunder', 'Water', 'Rain', 'Raindrop', 'Rain on surface', 'Stream', 'Waterfall', 'Ocean', 'Waves, surf', 'Steam', 'Gurgling', 'Fire', 'Crackle', 'Vehicle', 'Boat, Water vehicle', 'Sailboat, sailing ship', 'Rowboat, canoe, kayak', 'Motorboat, speedboat', 'Ship', 'Motor vehicle (road)', 'Car', 'Vehicle horn, car horn, honking', 'Toot', 'Car alarm', 'Power windows, electric windows', 'Skidding', 'Tire squeal', 'Car passing by', 'Race car, auto racing', 'Truck', 'Air brake', 'Air horn, truck horn', 'Reversing beeps', 'Ice cream truck, ice cream van', 'Bus', 'Emergency vehicle', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Motorcycle', 'Traffic noise, roadway noise', 'Rail transport', 'Train', 'Train whistle', 'Train horn', 'Railroad car, train wagon', 'Train wheels squealing', 'Subway, metro, underground', 'Aircraft', 'Aircraft engine', 'Jet engine', 'Propeller, airscrew', 'Helicopter', 'Fixed-wing aircraft, airplane', 'Bicycle', 'Skateboard', 'Engine', 'Light engine (high frequency)', "Dental drill, dentist's drill", 'Lawn mower', 'Chainsaw', 'Medium engine (mid frequency)', 'Heavy engine (low frequency)', 'Engine knocking', 'Engine starting', 'Idling', 'Accelerating, revving, vroom', 'Door', 'Doorbell', 'Ding-dong', 'Sliding door', 'Slam', 'Knock', 'Tap', 'Squeak', 'Cupboard open or close', 'Drawer open or close', 'Dishes, pots, and pans', 'Cutlery, silverware', 'Chopping (food)', 'Frying (food)', 'Microwave oven', 'Blender', 'Water tap, faucet', 'Sink (filling or washing)', 'Bathtub (filling or washing)', 'Hair dryer', 'Toilet flush', 'Toothbrush', 'Electric toothbrush', 'Vacuum cleaner', 'Zipper (clothing)', 'Keys jangling', 'Coin (dropping)', 'Scissors', 'Electric shaver, electric razor', 'Shuffling cards', 'Typing', 'Typewriter', 'Computer keyboard', 'Writing', 'Alarm', 'Telephone', 'Telephone bell ringing', 'Ringtone', 'Telephone dialing, DTMF', 'Dial tone', 'Busy signal', 'Alarm clock', 'Siren', 'Civil defense siren', 'Buzzer', 'Smoke detector, smoke alarm', 'Fire alarm', 'Foghorn', 'Whistle', 'Steam whistle', 'Mechanisms', 'Ratchet, pawl', 'Clock', 'Tick', 'Tick-tock', 'Gears', 'Pulleys', 'Sewing machine', 'Mechanical fan', 'Air conditioning', 'Cash register', 'Printer', 'Camera', 'Single-lens reflex camera', 'Tools', 'Hammer', 'Jackhammer', 'Sawing', 'Filing (rasp)', 'Sanding', 'Power tool', 'Drill', 'Explosion', 'Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun', 'Fireworks', 'Firecracker', 'Burst, pop', 'Eruption', 'Boom', 'Wood', 'Chop', 'Splinter', 'Crack', 'Glass', 'Chink, clink', 'Shatter', 'Liquid', 'Splash, splatter', 'Slosh', 'Squish', 'Drip', 'Pour', 'Trickle, dribble', 'Gush', 'Fill (with liquid)', 'Spray', 'Pump (liquid)', 'Stir', 'Boiling', 'Sonar', 'Arrow', 'Whoosh, swoosh, swish', 'Thump, thud', 'Thunk', 'Electronic tuner', 'Effects unit', 'Chorus effect', 'Basketball bounce', 'Bang', 'Slap, smack', 'Whack, thwack', 'Smash, crash', 'Breaking', 'Bouncing', 'Whip', 'Flap', 'Scratch', 'Scrape', 'Rub', 'Roll', 'Crushing', 'Crumpling, crinkling', 'Tearing', 'Beep, bleep', 'Ping', 'Ding', 'Clang', 'Squeal', 'Creak', 'Rustle', 'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Clickety-clack', 'Rumble', 'Plop', 'Jingle, tinkle', 'Hum', 'Zing', 'Boing', 'Crunch', 'Silence', 'Sine wave', 'Harmonic', 'Chirp tone', 'Sound effect', 'Pulse', 'Inside, small room', 'Inside, large room or hall', 'Inside, public space', 'Outside, urban or manmade', 'Outside, rural or natural', 'Reverberation', 'Echo', 'Noise', 'Environmental noise', 'Static', 'Mains hum', 'Distortion', 'Sidetone', 'Cacophony', 'White noise', 'Pink noise', 'Throbbing', 'Vibration', 'Television', 'Radio', 'Field recording']

class_labels_human = ["Clear human speech", "Coughing", "Sneezing", "Yelling or Shouting", "Crying", "Laughing", "Sighing", "Heavy breathing or panting"]
class_labels_background = ["Clear human speech", "Television in the background", "Vacuum cleaner", "Music playing", "Dog barking", "Baby crying", "Children playing or shouting", "Another person talking in the background", "Street traffic, car horn, or sirens", "Wind noise", "Keyboard typing", "Public announcement or PA system", "Restaurant or cafe chatter"]
class_labels_technical = ["Clear human speech", "Static or crackling on the line", "Digital artifacts or garbled audio", "Echo or feedback", "DTMF tones (keypad pressing)", "Silence"]
class_labels_emotion = ["Neutral emotion", "Happiness or joy", "Sadness", "Anger", "Fear or anxiety", "Surprise", "Disgust", "Confusion or uncertainty"]

class_labels_gender = ["Male voice", "Female voice", "Child voice"]

label_categories = {
    "All:" :class_labels_all,
    "Human Sounds": class_labels_human,
    "Background Noise": class_labels_background,
    "Technical Issues": class_labels_technical,
    "Emotional Tone": class_labels_emotion,
    "Gender": class_labels_gender,
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
            
            # Format for printing
            formatted_print_results = [(label, f"{prob * 100:.2f}%") for label, prob in sorted_results]
            pprint(formatted_print_results)

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
        clap_output_all = gr.Label(label="All", num_top_classes=3)
    with gr.Row():
        clap_output_human = gr.Label(label="Human Sounds", num_top_classes=3)
        clap_output_background = gr.Label(label="Background Noise", num_top_classes=3)
    
    with gr.Row():
        clap_output_technical = gr.Label(label="Technical Issues", num_top_classes=3)
        clap_output_emotion = gr.Label(label="Emotional Tone", num_top_classes=3)

    with gr.Row():
        clap_output_gender = gr.Label(label="Gender", num_top_classes=3)

    outputs_list = [
        clap_output_all,
        clap_output_human, clap_output_background, 
        clap_output_technical, clap_output_emotion,
        clap_output_gender,
    ]
    
    audio_input.change(
        fn=process_recording,
        inputs=audio_input,
        outputs=outputs_list
    )

iface.launch(debug=True)