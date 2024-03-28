#!/usr/bin/env python3
import sys
import time
from pathlib import Path
from typing import Any

import requests
import tomllib
from faster_whisper import WhisperModel
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from pydub import AudioSegment


# Load configuration from TOML file
def load_config(file_path="config.toml") -> dict[str, Any]:
    with open(file_path, "rb") as config_file:
        return tomllib.load(config_file)


config: dict[str, Any] = load_config()

AUDIO_DIR: Path = Path(config["audio_dir"])
MODEL_SIZE: str = config["model_size"]
WEBHOOK_URL: str = config["webhook_url"]


def wait_for_file_write_completion(file_path: Path, check_interval=1):
    file_size = -1
    while True:
        current_size: int = file_path.stat().st_size
        if current_size == file_size:
            break
        else:
            file_size: int = current_size
            time.sleep(check_interval)


def reencode_to_opus(source_file: Path, bitrate_kbps: int = 64) -> Path:
    # Load the audio file
    audio = AudioSegment.from_file(source_file)

    # Create the output file path
    output_file = source_file.with_suffix(".opus")

    # Set the output parameters
    output_params = ["-acodec", "libopus", "-b:a", f"{bitrate_kbps}k"]

    # Export the audio to Opus format
    audio.export(output_file, format="opus", parameters=output_params)

    return output_file



class AudioTranscriber:
    def __init__(self, model_size: str) -> None:
        self.model: WhisperModel = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )

    def _transcribe(self, file_path: Path, model: WhisperModel) -> str:
        """Transcribe audio file using the Whisper model and return the text."""
        segments, _ = self.model.transcribe(str(file_path), vad_filter=True)
        return "".join(segment.text for segment in segments)

    def transcribe(self, file_path: Path) -> str:
        attempt: int = 0
        while True:
            try:
                transcription: str = self._transcribe(file_path, self.model)
                time.sleep(1)
                return transcription
            except Exception as e:  # noqa
                if attempt > 3:
                    raise e
                attempt += 1


class MessageSender:
    """Sends messages to a specified destination."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url: str = webhook_url

    def send_message(self, message: str) -> None:
        """Send the message to the specified destination."""
        data: dict[str, Any] = {"content": message}
        response: requests.Response = requests.post(self.webhook_url, json=data)
        response.raise_for_status()
Here's the rewritten code to encode an audio file to opus using the pydub library:

```python
from pydub import AudioSegment

def encode_audio_to_opus(input_file: str, output_file: str) -> None:
    """Encode an audio file to opus format."""
    audio: AudioSegment = AudioSegment.from_file(input_file)
    audio.export(output_file, format="opus", codec="libopus")

    def send_message_with_file(self, message: str, file: Path) -> None:
        """Send the message with an attached file to the specified destination."""
        with file.open("rb") as f:
            files: dict[str, tuple[str, bytes]] = {"file": (file.name, f.read())}
            data: dict[str, Any] = {"content": message}
            response: requests.Response = requests.post(
                self.webhook_url, data=data, files=files
            )
            response.raise_for_status()


class AudioFileEventHandler(FileSystemEventHandler):
    """Handles new audio files and processes them using the Whisper model."""

    def __init__(
        self, transcriber: AudioTranscriber, message_sender: MessageSender
    ) -> None:
        super().__init__()
        self.transcriber: AudioTranscriber = transcriber
        self.message_sender: MessageSender = message_sender

    def _process_file(self, file_path: Path) -> None:
        """Process the specified audio file and send the transcription."""
        print(f"Processing new audio file: {file_path}")

        transcription: str = self.transcriber.transcribe(file_path)

        print(f"Transcription: {transcription}")
        self.message_sender.send_message(transcription)

    def on_created(self, event: FileSystemEvent) -> None:
        self.message_sender.send_message("DEBUG: New audio file created.")
        file: Path = Path(event.src_path)
        if file.suffix == ".mp3":
            wait_for_file_write_completion(file)
            self._process_file(file)

    def on_moved(self, event: FileSystemEvent) -> None:
        self.message_sender.send_message("DEBUG: Audio file moved.")
        file: Path = Path(event.dest_path)
        if file.suffix == ".mp3":
            self._process_file(file)


if __name__ == "__main__":
    observer: BaseObserver = Observer()
    try:
        print("Initializing...")
        transcriber = AudioTranscriber(MODEL_SIZE)
        message_sender = MessageSender(WEBHOOK_URL)
        event_handler: FileSystemEventHandler = AudioFileEventHandler(
            transcriber, message_sender
        )

        print(f"Loaded. Watching {AUDIO_DIR.name} for new or moved files...")
        observer.schedule(event_handler, str(AUDIO_DIR), recursive=True)
        observer.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        observer.stop()
        observer.join()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
