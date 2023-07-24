import whisper

model = whisper.load_model("base")


def transcribe(file: str):
    return model.transcribe(file)
