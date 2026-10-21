# TODO translate via script

import argparse
import os
import re
import tempfile

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TARGET_LANGUAGES = {
    'nl': "Nederlands",
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
    'cmn': "普通话",
    'sv': "Svenska",
    'da': "Dansk",
    'fi': "Suomi",
    'cs': "Čeština",
    'hu': "Magyar",
    'ro': "Română",
    'el': "Ελληνικά",
    'he': "עברית",
    'ms': "Bahasa Melayu",
    'no': "Norsk",
    'uk': "Українська",
    'ca': "Català",
    'lt': "Lietuvių",
    'lv': "Latviešu",
    'sk': "Slovenčina",
    'sl': "Slovenščina",
    'et': "Eesti",
    'hr': "Hrvatski",
    'sr': "Српски",
    'mk': "Македонски",
    'mt': "Malti",
    'sq': "Shqip",
    'hy': "Հայերեն",
    'af': "Afrikaans",
    'sw': "Kiswahili",
    'fil': "Filipino",
    'fa': "فارسی",
    'ml': "മലയാളം",
    'bxr': "Буряад",
    'mn': "Монгол",
    'sah': "Саха",
    'evn': "Эвэн",
    'kjh': "Хакас",
    'tyv': "Тыва",
    'yrk': "Ненц"
}

SOURCE_COLUMN = "keys" 

model_name = "facebook/nllb-200-distilled-600M"

# Mapping from language names to NLLB codes
LANGUAGE_CODES = {
    "Nederlands": "nld_Latn",
    "Русский": "rus_Cyrl",
    "Español": "spa_Latn",
    "Français": "fra_Latn",
    "Български": "bul_Cyrl",
    "العربية": "arb_Arab",
    "Deutsch": "deu_Latn",
    "हिन्दी": "hin_Deva",
    "Bahasa Indonesia": "ind_Latn",
    "Italiano": "ita_Latn",
    "日本語": "jpn_Jpan",
    "한국어": "kor_Hang",
    "Polski": "pol_Latn",
    "Português": "por_Latn",
    "ภาษาไทย": "tha_Thai",
    "Türkçe": "tur_Latn",
    "Tiếng Việt": "vie_Latn",
    "普通话": "zho_Hans",
    "Svenska": "swe_Latn",
    "Dansk": "dan_Latn",
    "Suomi": "fin_Latn",
    "Čeština": "ces_Latn",
    "Magyar": "hun_Latn",
    "Română": "ron_Latn",
    "Ελληνικά": "ell_Grek",
    "עברית": "heb_Hebr",
    "Bahasa Melayu": "msa_Latn",
    "Norsk": "nno_Latn",
    "Українська": "ukr_Cyrl",
    "Català": "cat_Latn",
    "Lietuvių": "lit_Latn",
    "Latviešu": "lav_Latn",
    "Slovenčina": "slk_Latn",
    "Slovenščina": "slv_Latn",
    "Eesti": "est_Latn",
    "Hrvatski": "hrv_Latn",
    "Српски": "srp_Cyrl",
    "Македонски": "mkd_Cyrl",
    "Malti": "mlt_Latn",
    "Shqip": "als_Latn",
    "Հայերեն": "hye_Armn",
    "Afrikaans": "afr_Latn",
    "Kiswahili": "swh_Latn",
    "Filipino": "fil_Latn",
    "فارسی": "fas_Arab",
    "മലയാളം": "mal_Mlym",
    "Буряад": "bxr_Cyrl",
    "Монгол": "mon_Cyrl",
    "Саха": "sah_Cyrl",
    "Эвэн": "evn_Cyrl",
    "Хакас": "kjh_Cyrl",
    "Тыва": "tyv_Cyrl",
    "Ненц": "yrk_Cyrl"
}

PROTECTED_TEXT_PATTERN = re.compile(
    r"\[/?url(?:=[^\]]*)?\]|https?://[^\s\]\)]+|\{[^{}]*\}"
)


def protect_text(text: str):
    protected = {}

    def replace_match(match):
        marker = f"ZXQPROTECTEDTOKEN{len(protected):04d}QXZ"
        protected[marker] = match.group(0)
        return marker

    return PROTECTED_TEXT_PATTERN.sub(replace_match, text), protected


def restore_protected_text(text: str, protected: dict) -> str | None:
    for marker in protected:
        if text.count(marker) != 1:
            return None
        text = text.replace(marker, protected[marker])
    return text


def translate_text_via_local_model(text: str, tokenizer, model, target_language_name) -> str | None:
    if not text.strip():
        return ""

    try:
        protected_text, protected = protect_text(text)
        target_token_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES[target_language_name])
        if target_token_id == tokenizer.unk_token_id:
            print(f"Unsupported NLLB target language: {target_language_name}")
            return None

        inputs = tokenizer(protected_text, return_tensors="pt")
        with torch.inference_mode():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_new_tokens=256,
                num_beams=4,
            )

        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()
        restored_text = restore_protected_text(translated_text, protected)
        if restored_text is None:
            print(f"Translation changed or lost protected markup for {target_language_name}: {text!r}")
            return None

        if is_uppercase_ignore_non_alpha(text):
            pieces = re.split(f"({'|'.join(re.escape(marker) for marker in protected)})", translated_text) if protected else [translated_text]
            translated_text = "".join(piece if piece in protected else piece.upper() for piece in pieces)
            restored_text = restore_protected_text(translated_text, protected)
        return restored_text
    except Exception as e:
        print(f"Translation error for '{text}' to {target_language_name}: {e}")
        return None

def is_uppercase_ignore_non_alpha(s):
    if not isinstance(s, str):
        return False
    s = PROTECTED_TEXT_PATTERN.sub("", s)
    alpha_chars = [c for c in s if c.isalpha()]
    return len(alpha_chars) > 0 and all(c.isupper() for c in alpha_chars)


def save_csv_atomically(df: pd.DataFrame, output_file: str):
    output_path = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_dir, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            df.to_csv(temporary_file, index=False)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def process_csv_translation(input_file: str, output_file: str, overwrite: bool = False):
    """Main function to read CSV, translate column by column, and save results."""
    print("===============================================")
    print("=== STARTING LOCAL LLM TRANSLATION PROCESS ===")
    print("===============================================")

    try:
        df = pd.read_csv(input_file, keep_default_na=False, dtype=str)
    except FileNotFoundError:
        print(f"\n!!! ERROR: Input file '{input_file}' not found. Please check the path.")
        return

    if SOURCE_COLUMN not in df.columns:
        print(f"\n!!! ERROR: Required source column '{SOURCE_COLUMN}' is missing.")
        return

    en_column = 'en'
    if en_column not in df.columns:
        df[en_column] = ""
    missing_english = df[en_column].str.strip().eq("")
    df.loc[missing_english, en_column] = df.loc[missing_english, SOURCE_COLUMN]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "eng_Latn"
    supported_languages = {}
    for lang_code in TARGET_LANGUAGES:
        language_name = TARGET_LANGUAGES[lang_code]
        target_token_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES[language_name])
        if target_token_id == tokenizer.unk_token_id:
            print(f"Skipping {lang_code}: NLLB does not support {language_name} ({LANGUAGE_CODES[language_name]}).")
        else:
            supported_languages[lang_code] = language_name
        if lang_code not in df.columns:
            df[lang_code] = ""

    work_exists = any(
        str(df.at[row_index, lang_code]).strip() == ""
        or overwrite
        for lang_code in supported_languages
        for row_index in df.index
        if str(df.at[row_index, en_column]).strip()
    )
    model = None
    if work_exists:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

    for lang_code, language_name in supported_languages.items():
        print(f"\n\n[ STARTING TRANSLATIONS FOR LANGUAGE: {lang_code.upper()} ]")
        for row_index in df.index:
            original_text = str(df.at[row_index, en_column])
            if not original_text.strip():
                continue

            current_translation = str(df.at[row_index, lang_code])
            if not overwrite and current_translation.strip():
                continue

            if original_text.strip() == "English":
                translation = language_name
            else:
                translation = translate_text_via_local_model(
                    original_text, tokenizer, model, language_name
                )

            if translation is None or (original_text.strip() and not translation.strip()):
                print(f"Leaving {lang_code} row {row_index + 2} unchanged; translation failed validation.")
                continue

            df.at[row_index, lang_code] = translation
            print(f"Translated {lang_code} row {row_index + 2}.")

        try:
            save_csv_atomically(df, output_file)
        except Exception as e:
            print(f"\n!!! FATAL ERROR during saving: {e}")
            return
        print(f"\nFinished translations for {lang_code.upper()}.")

    try:
        save_csv_atomically(df, output_file)
        print("\n===============================================")
        print(f"✨ SUCCESS! All translations saved to '{output_file}'")
        print("Unsupported languages were skipped; translate those columns separately.")
        print("===============================================")
    except Exception as e:
        print(f"\n!!! FATAL ERROR during saving: {e}")

def main():
    parser = argparse.ArgumentParser(description='Translate text from a CSV file using a local LLM model.')

    parser.add_argument('-i', '--input', 
                        type=str, 
                        default='translation.csv',
                        help='Path to the input CSV file containing text to translate')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        default='translated_output.csv',
                        help='Path where the translated output CSV will be saved')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing non-empty translations in supported language columns')

    args = parser.parse_args()

    process_csv_translation(args.input, args.output, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
