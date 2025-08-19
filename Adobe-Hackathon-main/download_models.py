import os
import shutil
from pathlib import Path
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def download_and_save_models(save_directory="models"):
    try:
        # Ensure the save directory exists
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------
        # 1. Download & Save Flan-T5
        # ------------------------------
        flan_t5_path = save_path / "flan-t5-small"
        print("Downloading google/flan-t5-small tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        print("Downloading google/flan-t5-small model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        print(f"Saving google/flan-t5-small tokenizer to {flan_t5_path}...")
        tokenizer.save_pretrained(flan_t5_path)

        print(f"Saving google/flan-t5-small model to {flan_t5_path}...")
        model.save_pretrained(flan_t5_path)

        # ------------------------------
        # 2. Download & Save Whisper
        # ------------------------------
        whisper_path = save_path / "whisper-base"
        print("Downloading Whisper base model...")
        whisper.load_model("base", download_root=str(whisper_path))
        print(f"Whisper base model successfully saved to {whisper_path}")

        print(f"\n✅ All models (Flan-T5, Whisper) successfully saved to {save_directory}\n")

    except ImportError as e:
        print(f"❌ Import error: {e}. Ensure all dependencies are installed (openai-whisper, transformers).")
    except Exception as e:
        print(f"❌ An error occurred: {e}. Check network, disk space, or model availability.")


def main():
    save_directory = "models"
    download_and_save_models(save_directory)


if __name__ == "__main__":
    main()
