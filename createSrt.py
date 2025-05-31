#!/usr/bin/env python3

import os
import glob
import subprocess
import pysrt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuratie ---
WHISPER_MODEL = "large-v3" # Het grote Whisper model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face Model voor Engels naar Nederlands vertaling
HF_TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-nl"

# Lees de doeltaal uit een omgevingsvariabele, standaard naar 'nl'
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", "nl")
print(f"⚙️ Doeltaal ingesteld op: {TARGET_LANGUAGE.upper()}")

# --- Initialiseer Hugging Face Model (wordt aangeroepen in main) ---
hf_tokenizer = None
hf_model = None

def load_hf_translate_model():
    """Laadt het Hugging Face vertaalmodel en tokenizer."""
    global hf_tokenizer, hf_model
    try:
        print(f"🌐 Laden van Hugging Face vertaalmodel: {HF_TRANSLATE_MODEL} op apparaat: {DEVICE}...")
        hf_tokenizer = AutoTokenizer.from_pretrained(HF_TRANSLATE_MODEL)
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_TRANSLATE_MODEL).to(DEVICE)
        print("✅ Hugging Face vertaalmodel succesvol geladen.")
    except Exception as e:
        print(f"❌ Fout bij het laden van Hugging Face vertaalmodel: {e}")
        hf_tokenizer = None
        hf_model = None

# --- Functies ---

def run_whisperx(audio_path, output_dir, lang_code="en"):
    """
    Voert WhisperX uit om ondertitels te genereren.
    Zorgt ervoor dat de output bestandsnaam .lang_code.srt is.
    """
    print(f"🎧 Genereren van {lang_code.upper()} ondertitels voor: {os.path.basename(audio_path)}")
    
    batch_size = 1 # Batch size is 1 voor het large-v3 model
    compute_type = "float16" # Behoud float16, dit is geheugenvriendelijker

    # WhisperX genereert standaard [basenaam].srt als output_format srt is
    # We laten het dit eerst doen, en hernoemen het daarna
    whisperx_raw_output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".mkv", ".srt").replace(".mp4", ".srt"))
    
    # De uiteindelijke gewenste naam (bijv. .en.srt)
    final_output_srt_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".mkv", f".{lang_code}.srt").replace(".mp4", f".{lang_code}.srt"))

    command = [
        "whisperx",
        audio_path,
        "--model", WHISPER_MODEL, # Gebruikt het large-v3 model
        "--output_dir", output_dir,
        "--output_format", "srt",
        "--task", "transcribe",
        "--language", lang_code,
        "--batch_size", str(batch_size),
        "--compute_type", compute_type,
        "--device", DEVICE
    ]
    
    print(f"    ➡️ Executing WhisperX command: {' '.join(command)}")
    print("    Wachten op WhisperX output...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"✅ WhisperX succesvol uitgevoerd voor {os.path.basename(audio_path)}")
        
        if result.stdout:
            print(f"    WhisperX stdout:\n{result.stdout}")
        if result.stderr:
            print(f"    WhisperX stderr:\n{result.stderr}")

        if os.path.exists(whisperx_raw_output_path):
            if whisperx_raw_output_path != final_output_srt_path:
                os.rename(whisperx_raw_output_path, final_output_srt_path)
                print(f"✅ Hernoemd van '{os.path.basename(whisperx_raw_output_path)}' naar '{os.path.basename(final_output_srt_path)}'")
            return final_output_srt_path
        else:
            print(f"❌ Fout: Verwachte WhisperX output '{os.path.basename(whisperx_raw_output_path)}' niet gevonden na uitvoering.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Fout bij WhisperX uitvoering voor {os.path.basename(audio_path)}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Onverwachte fout bij WhisperX uitvoering voor {os.path.basename(audio_path)}: {e}")
        return None

def translate_srt(english_srt_path, target_lang): # target_lang komt nu van de env var
    """Vertaalt een SRT-bestand met het Hugging Face model."""
    global hf_tokenizer, hf_model
    
    if hf_model is None or hf_tokenizer is None:
        print("❌ Hugging Face vertaalmodel niet geladen, kan niet vertalen.")
        return None

    print(f"🌐 Vertalen '{os.path.basename(english_srt_path)}' naar {target_lang.upper()} met Hugging Face...")
    
    try:
        subs = pysrt.open(english_srt_path, encoding='utf-8')
        translated_subs = pysrt.SubRipFile()

        texts_to_translate = [sub.text for sub in subs]
        translated_texts = []

        batch_size = 32 # Vertaal in batches van 32 zinnen
        total_batches = (len(texts_to_translate) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts_to_translate), batch_size):
            batch_num = (i // batch_size) + 1
            print(f"    ➡️ Verwerken van vertaalbatch {batch_num} van {total_batches}...", end='\r')
            batch = texts_to_translate[i:i+batch_size]
            
            # Belangrijk: De prefix voor het vertaalmodel moet overeenkomen met het model.
            # Voor 'Helsinki-NLP/opus-mt-en-nl' is '>>nl<<' correct.
            # Als je naar een andere taal vertaalt, moet dit '>>[taalcode]<<' zijn.
            # We voegen de prefix toe aan elke tekst in de batch.
            prefixed_batch = [f">>{target_lang}<< {text}" for text in batch]

            inputs = hf_tokenizer(prefixed_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                translated_ids = hf_model.generate(inputs.input_ids, num_beams=5, early_stopping=True, max_length=hf_tokenizer.model_max_length)
            
            translated_batch = hf_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            translated_texts.extend(translated_batch)
        print() # Nieuwe regel na de voortgang

        if len(translated_texts) != len(subs):
            print("WAARSCHUWING: Aantal vertaalde zinnen komt niet overeen met origineel.")

        for i, sub in enumerate(subs):
            new_sub = pysrt.SubRipItem(index=sub.index, start=sub.start, end=sub.end, text=translated_texts[i])
            translated_subs.append(new_sub)

        base_name_no_ext = os.path.splitext(english_srt_path)[0]
        if base_name_no_ext.endswith(".en"):
            base_name_no_ext = base_name_no_ext[:-3]
        dutch_srt_path = f"{base_name_no_ext}.{target_lang}.srt" # Gebruik target_lang hier
        
        translated_subs.save(dutch_srt_path, encoding='utf-8')
        print(f"✅ Vertaalde ondertitels opgeslagen als: {dutch_srt_path}")
        return dutch_srt_path

    except Exception as e:
        print(f"❌ Fout bij vertalen van ondertitels: {e}")
        base_name_no_ext = os.path.splitext(english_srt_path)[0]
        if base_name_no_ext.endswith(".en"):
            base_name_no_ext = base_name_no_ext[:-3]
        dutch_srt_path = f"{base_name_no_ext}.{target_lang}.srt" # Gebruik target_lang hier
        with open(dutch_srt_path, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"❌ Vertaalproces mislukt, leeg SRT-bestand gemaakt: {dutch_srt_path}")
        return None

# --- Hoofdlogica ---
if __name__ == "__main__":
    print("📝 Start transcriptie- en vertaalscript...")

    load_hf_translate_model()

    # Recursief zoeken in /data en subdirectories
    video_files = []
    video_files.extend(glob.glob("/data/**/*.mkv", recursive=True))
    video_files.extend(glob.glob("/data/**/*.mp4", recursive=True))
    
    video_files.sort()

    if not video_files:
        print("Geen MKV- of MP4-bestanden gevonden in de /data map of subdirectories.")
        print("---------------------------------------")
        print("Geen bestanden meer om te verwerken. Container stopt.")
        exit(0) # Container stopt hier als er niets te doen is

    for video_file in video_files:
        print(f"\n🎧 Verwerken van: {os.path.basename(video_file)}")

        base_name = os.path.splitext(video_file)[0]
        english_srt_path_expected = f"{base_name}.en.srt" 
        destination_srt_path_expected = f"{base_name}.{TARGET_LANGUAGE}.srt" # Dit is de doeltaal SRT

        # --- NIEUWE LOGICA: Controleer EERST op de aanwezigheid van de doeltaal SRT ---
        if os.path.exists(destination_srt_path_expected) and os.path.getsize(destination_srt_path_expected) > 0:
            print(f"✅ {TARGET_LANGUAGE.upper()} ondertitels al aanwezig: {destination_srt_path_expected}. Overslaan.")
            continue # Sla dit bestand volledig over en ga naar het volgende video bestand.

        # Als de doeltaal SRT niet bestaat of leeg is, gaan we verder met transcriberen/vertalen
        en_status = "Niet gegenereerd"
        nl_status = "Niet gegenereerd"
        english_srt_for_translation = None # Initialiseer dit als None

        # Controleer vervolgens op de Engelse (bron) SRT
        if not os.path.exists(english_srt_path_expected) or os.path.getsize(english_srt_path_expected) == 0:
            print(f"🔄 Genereert Engelse ondertitels voor: {os.path.basename(video_file)}")
            generated_srt = run_whisperx(video_file, os.path.dirname(video_file), lang_code="en") # Output in dezelfde map als video
            if not generated_srt:
                print(f"❌ Kon geen Engelse ondertitels genereren voor {os.path.basename(video_file)}. Overslaan vertaling.")
                en_status = "Transcriptiefout"
                # Als de transcriptie mislukt, blijft english_srt_for_translation None, en wordt de vertaling overgeslagen.
            else:
                english_srt_for_translation = generated_srt
                en_status = "Gegenereerd"
        else:
            print(f"✅ Engelse ondertitels al aanwezig: {english_srt_path_expected}. Direct vertalen.")
            english_srt_for_translation = english_srt_path_expected
            en_status = "Al aanwezig"

        # Ga verder met vertalen als er een Engelse SRT beschikbaar is (bestaand of zojuist gegenereerd)
        if english_srt_for_translation: # Alleen proberen te vertalen als we een Engelse SRT hebben
            if hf_model is not None and hf_tokenizer is not None:
                # De controle op de aanwezigheid van destination_srt_path_expected is al aan het begin van de loop gedaan.
                # Dus hier proberen we altijd te vertalen als english_srt_for_translation beschikbaar is.
                translated_srt = translate_srt(english_srt_for_translation, TARGET_LANGUAGE)
                if not translated_srt:
                    print(f"❌ Kon geen {TARGET_LANGUAGE.upper()} ondertitels vertalen voor {os.path.basename(video_file)}.")
                    nl_status = "Vertaalfout"
                else:
                    nl_status = "Vertaald"
            else:
                print("Skippen van vertaling omdat het Hugging Face vertaalmodel niet geladen kon worden.")
                nl_status = "Vertaalmodel ontbreekt"
        else:
            print(f"⚠️ Geen Engelse ondertitels beschikbaar voor vertaling voor {os.path.basename(video_file)}.")
            nl_status = "Geen EN SRT om te vertalen"

        print(f"\n--- Samenvatting voor {os.path.basename(video_file)} ---")
        print(f"  Engelse SRT status: {en_status}")
        print(f"  {TARGET_LANGUAGE.upper()} SRT status: {nl_status}")
        print("-----------------------------------\n")

    print("\n---------------------------------------")
    print("Alle bestanden verwerkt. Container stopt.")
    exit(0) # Container stopt na verwerking van alle bestanden
