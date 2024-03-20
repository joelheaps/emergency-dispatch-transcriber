#!/usr/bin/env python3
import tomllib
from faster_whisper import WhisperModel
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
import requests
from typing import Any

from watchdog.observers.api import BaseObserver


# Load configuration from TOML file
def load_config(file_path="config.toml") -> dict[str, Any]:
    with open(file_path, "rb") as config_file:
        return tomllib.load(config_file)


config: dict[str, Any] = load_config()

AUDIO_DIR: Path = Path(config["audio_dir"])
MODEL_SIZE: str = config["model_size"]
WEBHOOK_URL: str = config["webhook_url"]


def load_model(model_size: str) -> WhisperModel:
    """Load and return the Whisper model based on the model size."""
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def send_discord_message(text: str, webhook_url: str) -> None:
    """Send a message to the Discord webhook URL."""
    data: dict[str, Any] = {"content": text}
    response: requests.Response = requests.post(webhook_url, json=data)
    response.raise_for_status()


def transcribe_audio(file_path: Path, model: WhisperModel) -> str:
    """Transcribe audio file using the Whisper model and return the text."""
    segments, _ = model.transcribe(str(file_path), vad_filter=True)
    return "".join(segment.text for segment in segments)


class AudioFileHandler(FileSystemEventHandler):
    """Handles new audio files and processes them using the Whisper model."""

    def __init__(self, model: WhisperModel, webhook_url: str) -> None:
        super().__init__()
        self.model: WhisperModel = model
        self.webhook_url: str = webhook_url

    def process_file(self, file_path: Path) -> None:
        """Process the specified audio file and send the transcription via Discord."""
        print(f"Processing new audio file: {file_path}")
        transcription: str = transcribe_audio(file_path, self.model)
        print(f"Transcription: {transcription}")
        send_discord_message(transcription, self.webhook_url)

    def on_created(self, event: FileSystemEvent) -> None:
        if Path(event.src_path).suffix == ".mp3":
            self.process_file(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if Path(event.src_path).suffix == ".mp3":
            self.process_file(Path(event.src_path))


if __name__ == "__main__":
    print("Initializing text-to-speech model...")
    model: WhisperModel = load_model(MODEL_SIZE)

    event_handler: FileSystemEventHandler = AudioFileHandler(model, WEBHOOK_URL)
    observer: BaseObserver = Observer()
    observer.schedule(event_handler, str(AUDIO_DIR), recursive=True)

    print("Starting audio directory watcher...")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        observer.stop()
        observer.join()
