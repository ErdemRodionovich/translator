import pandas as pd
import requests
import json
import time
from tqdm import tqdm # Using tqdm for a nice progress bar (install with pip install tqdm)

# ========================================================
# --- CONFIGURATION SECTION ---
# !!! YOU MUST UPDATE THESE VALUES BASED ON YOUR LM STUDIO SETUP !!!
# ========================================================

# The URL and port where your local model server is running 
LOCAL_API_URL = "http://localhost:1234/v1/chat/completions" 
MODEL_NAME = "mistralai/ministral-3-14b-reasoning" # Optional, but good practice for the request body

# Input file path (the CSV you want to process)
INPUT_FILE = "translation.csv" 

TARGET_LANGUAGES = {
    'ru': "Русский",
    'es': "Español",
    'fr': "Français",
    'bg': "Български",
    'ar': "العربية",
    'de': "Deutsch",
    'hi': "हिन्दी",
    'id': "Bahasa Indonesia",
    'it': "Italiano",
    'ja': "日本語",
    'ko': "한국어",
    'pl': "Polski",
    'pt': "Português",
    'th': "ภาษาไทย",
    'tr': "Türkçe",
    'vi': "Tiếng Việt",
    'cmn': "普通话"
}

# The column name in your CSV that contains the text needing translation
SOURCE_COLUMN = "keys" 

# The file where the results will be saved
OUTPUT_FILE = "translated_output_local.csv" 

# System Prompt: Guides the model's behavior (CRITICAL for good quality)
SYSTEM_PROMPT = "You are a professional and highly accurate translator. Your only task is to translate the provided text into the target language. Do not add any commentary, introductions, or extra formatting. Don't touch parameters in curle braces, for example {m}. If you see phrases with verbs, then it is for button, keep the verb tense as for action to do, for example, 'Buy moves' translated to Русский as 'Купить ходы'. If the source text is empty or only contains whitespace, return an empty string. Be kind and respectful, for example 'You have moves' is translated into Русский as 'Вы имеете ходы'. Keep length of phrase the same or shorter, longer at most by 10%. If phrase is in all caps, such as 'CONGRATS!, translate it in all caps as well, for example 'CONGRATS!' is translated into Русский as 'ПОЗДРАВЛЯЕМ!'."

def call_llm(prompt: str, system_promts: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_promts},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # Lower temperature for factual tasks like translation
        "max_tokens": 10000 # Set a reasonable limit
    }

    try:
        # Send the request to your local server endpoint
        response = requests.post(LOCAL_API_URL, json=payload)
        response.raise_for_status() # Raises an HTTPError if status code is bad (4xx or 5xx)
        
        data = response.json()
        # print(f"API Response for '{text}' -> {target_language_code}: {data}")
        # Extract the translation text from the complex API response structure
        translation = data['choices'][0]['message']['content'].strip()
        return translation

    except requests.exceptions.ConnectionError:
        print("\n[!!! CONNECTION ERROR !!!]")
        print("Could not connect to the local API server.")
        print(f"Ensure LM Studio is running and that your API URL ({LOCAL_API_URL}) is correct.")
        return "[CONNECTION FAILED]"
    except requests.exceptions.RequestException as e:
        print(f"\n[!!! REQUEST ERROR !!!] An error occurred communicating with the API: {e}")
        return f"[REQUEST ERROR: {type(e).__name__}]"


def translate_text_via_local_model(text: str, target_language_code: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return ""
    prompt = (f"Translate the following text into {target_language_code}. "
              f"Source Text: '{text}'")
    return call_llm(prompt, SYSTEM_PROMPT)

def enhance_keys_of_phrases(text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return ""
    prompt = (f"Check given phrase and give better version of this phrase: more kind, more respectful, more human. "
              f"Source Text: '{text}'")
    system_prompt = "You are a helpful assistant that improves the tone of phrases, making them more kind, respectful, and human. Phrases are done by russian speaking developer, there can be slang, like 'Buy moves via AD', this can be improved as 'Buy moves by watching advertisement'. Your task is to take the provided text and enhance it while preserving its original meaning. Do not add any commentary or extra formatting. Don't change the meaning of the phrase. Don't change parameters in curle braces, for example {m}. If the source text is empty or only contains whitespace, return an empty string. Keep length of phrase the same or shorter, longer at most by 10%."
    return call_llm(prompt, system_prompt)

def process_csv_translation():
    """Main function to read CSV, translate column by column, and save results."""
    print("===============================================")
    print("=== STARTING LOCAL LLM TRANSLATION PROCESS ===")
    print("===============================================")

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n!!! ERROR: Input file '{INPUT_FILE}' not found. Please check the path.")
        return
    
    en_column = 'en'
    if not en_column in df.columns:
        df[en_column] = []
    i = 0
    for original_text in df[SOURCE_COLUMN]:
        i += 1
        while len(df[en_column]) < 1:
            df[en_column].append('')
        #enhanced_text = enhance_keys_of_phrases(original_text)
        #print(f"Original: '{original_text}' -> Enhanced: '{enhanced_text}', previous: '{df.at[i-1, en_column]}'")
        if df.at[i-1, en_column] == '' or pd.isna(df.at[i-1, en_column]):
            df.at[i-1, en_column] = original_text

    # Loop through each language we need to translate into
    for lang_code in TARGET_LANGUAGES:
        print(f"\n\n[ STARTING TRANSLATIONS FOR LANGUAGE: {lang_code.upper()} ]")
        new_column_name = f"{lang_code}"
        if not new_column_name in df.columns:
            df[new_column_name] = []

        i = 0
        for original_text in df[en_column]:
            i += 1
            # Call the translation function
            translation = translate_text_via_local_model(original_text, TARGET_LANGUAGES[lang_code])
            if translation == "[CONNECTION FAILED]" or translation.startswith("[REQUEST ERROR"):
                print(f"\n!!! Stopping translation for '{original_text}' due to API issues.")
                return
            else:
                while len(df[new_column_name]) < 1:
                    df[new_column_name].append('')
                
                print(f"Original: '{original_text}' -> Translated ({lang_code}): '{translation}', previous: '{df.at[i-1, new_column_name]}'")
                df.at[i-1, new_column_name] = translation

        print(f"\n✅ Finished translations for {lang_code.upper()}.")

    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print("\n===============================================")
        print(f"✨ SUCCESS! All translations saved to '{OUTPUT_FILE}'")
        print(f"Don't forget to translate to Buryad by hand.")
        print("===============================================")
    except Exception as e:
        print(f"\n!!! FATAL ERROR during saving: {e}")


if __name__ == "__main__":
    process_csv_translation()

