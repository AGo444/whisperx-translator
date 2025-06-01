import os
import subprocess
import pysrt
import sys
import argparse
import re

# --- Configuration ---
DEFAULT_TARGET_LANGUAGE = "nl"
WHISPERX_MODEL = "large-v3"
HF_TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-nl"
INPUT_DIR = "/data"

# --- Helper Functions ---
def clean_filename(filename):
    """
    Removes potentially problematic characters from a filename,
    especially useful for paths in shell commands.
    """
    return re.sub(r'[^\w\s\-\.\_]', '', filename).strip()

def is_video_file(filepath):
    """Checks if a file is a common video format."""
    return filepath.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'))

def get_target_language_prefix(target_language):
    """Returns the correct prefix for the Hugging Face translation model."""
    return f">>{target_language}<<"

def generate_subtitles(video_path, output_dir, target_language):
    """
    Generates English subtitles using WhisperX and then translates them.
    Handles existing SRT files to resume processing.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base_path = os.path.join(output_dir, base_name)
    english_srt_path = f"{output_base_path}.en.srt"
    target_srt_path = f"{output_base_path}.{target_language}.srt"

    # --- Logging for ALL existing SRT files related to this video ---
    found_any_srt = False
    for filename in os.listdir(output_dir):
        if filename.startswith(base_name) and filename.endswith('.srt'):
            full_srt_path = os.path.join(output_dir, filename)
            if os.path.isfile(full_srt_path) and os.path.getsize(full_srt_path) > 0:
                print(f"Found existing SRT: {filename}")
                found_any_srt = True
    if not found_any_srt:
        print("Found no existing .srt files for this video.")
    # --- End logging for ALL existing SRT files ---

    # Check if target language SRT already exists and is not empty
    if os.path.exists(target_srt_path) and os.path.getsize(target_srt_path) > 0:
        print(f"✅ Skipping '{video_path}': '{target_language}' subtitles already exist and are not empty.")
        return

    # Check if English SRT exists and is not empty
    if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0:
        print(f"➡️ English subtitles already exist for '{video_path}'. Proceeding to translate.")
    else:
        print(f"🎧 Generating EN subtitles for: {video_path}")
        whisperx_command = [
            "whisperx",
            video_path,
            "--model", WHISPERX_MODEL,
            "--output_dir", output_dir,
            "--output_format", "srt",
            "--language", "en", # Force English transcription
            "--batch_size", "8",
            "--compute_type", "float16",
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"
        ]
        print(f"➡️ Executing WhisperX command: {' '.join(whisperx_command)}")
        try:
            process = subprocess.Popen(whisperx_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, " ".join(whisperx_command))
            print(f"✅ WhisperX successfully executed for {video_path}")
            # --- Log that English file is created ---
            print(f"Created {os.path.basename(english_srt_path)}")
            # --- End log ---

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during WhisperX processing for '{video_path}':")
            print(f"Command: {e.cmd}")
            print(f"Return Code: {e.returncode}")
            print(f"Output: {e.output}")
            return
        except FileNotFoundError:
            print(f"❌ Error: 'whisperx' command not found. Is WhisperX installed correctly in the container?")
            return

    # Translate the English SRT to the target language
    if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0:
        print(f"🌐 Translating '{english_srt_path}' to '{target_language}' with Hugging Face model '{HF_TRANSLATE_MODEL}'...")
        try:
            translate_srt(english_srt_path, target_srt_path, target_language)
            print(f"✅ Translated subtitles saved as: {target_srt_path}")
        except Exception as e:
            print(f"❌ Error during translation for '{video_path}': {e}")
    else:
        print(f"⚠️ No English SRT found for '{video_path}' after transcription. Skipping translation.")

    print(f"\n--- Summary for {os.path.basename(video_path)} ---")
    print(f"English SRT: {english_srt_path} {'(Generated)' if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0 and not os.path.exists(target_srt_path) else '(Exists)' if os.path.exists(english_srt_path) and os.path.getsize(english_srt_path) > 0 else '(Failed/Missing)'}")
    print(f"Target ({target_language}) SRT: {target_srt_path} {'(Generated)' if os.path.exists(target_srt_path) and os.path.getsize(target_srt_path) > 0 else '(Failed/Missing)'}")
    print("------------------------------------------\n")


def translate_srt(input_srt_path, output_srt_path, target_language):
    """Translates an SRT file using Hugging Face's Transformers."""
    from transformers import pipeline
    print("🌐 Loading Hugging Face translation model...")
    translator = pipeline("translation", model=HF_TRANSLATE_MODEL, device=0)
    print("✅ Hugging Face translation model successfully loaded.")

    subs = pysrt.open(input_srt_path, encoding='utf-8')
    translated_subs = pysrt.SubRipFile()

    lang_prefix = get_target_language_prefix(target_language)

    texts_to_translate = [f"{lang_prefix} {sub.text}" for sub in subs]
    
    batch_size = 16
    for i in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[i:i + batch_size]
        print(f"➡️ Processing translation batch {i//batch_size + 1} of {len(texts_to_translate)//batch_size + 1}...")
        try:
            if not batch:
                continue
            
            translated_batch = translator(batch)
            translated_texts = [item['translation_text'] for item in translated_batch]
            
            for j, sub_text in enumerate(translated_texts):
                original_sub = subs[i+j]
                new_sub = pysrt.SubRipItem(index=original_sub.index, start=original_sub.start, end=original_sub.end, text=sub_text)
                translated_subs.append(new_sub)
        except Exception as e:
            print(f"❌ Error during translation batch processing (batch {i//batch_size + 1}): {e}")
            raise

    translated_subs.save(output_srt_path, encoding='utf-8')

# --- Main Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and translate subtitles for video files.")
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"Target language for translation (e.g., 'nl', 'de', 'fr'). Defaults to '{DEFAULT_TARGET_LANGUAGE}'."
    )
    args = parser.parse_args()

    target_language = args.language.lower()
    print(f"Starting subtitle generation and translation process for target language: {target_language}")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory '{INPUT_DIR}' does not exist. Please ensure your volume is mounted correctly.")
        sys.exit(1)

    found_videos = False
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if is_video_file(file):
                found_videos = True
                video_full_path = os.path.join(root, file)
                print(f"\n--- Processing video: {video_full_path} ---")
                generate_subtitles(video_full_path, root, target_language)

    if not found_videos:
        print(f"No supported video files (.mp4, .mkv, etc.) found in '{INPUT_DIR}' or its subdirectories.")
        print("Please ensure your video files are in the mounted /data directory.")