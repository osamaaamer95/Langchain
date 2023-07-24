"""
Youtube Video Summarizer

Step A: Download and transcribe video
Step B: Chunking and embed in vector db
Step C: Generate summary and return
"""
from Projects.YoutubeVideoSummarizer.download_yt_vid import download_yt_vid
from Projects.YoutubeVideoSummarizer.summarize import split_from_file, summarize, summarize_into_bullet_points
from Projects.YoutubeVideoSummarizer.transcribe import transcribe

filename = "video.mp4"
transcribed_textfile_name = "transcription.txt"
video_url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"


def transcribe_and_write_to_file():
    transcription = transcribe(filename)
    # write transcription to a text file
    with open(transcribed_textfile_name, 'w') as file:
        file.write(transcription['text'])


download_yt_vid(video_url, filename)

transcribe_and_write_to_file()

documents = split_from_file(transcribed_textfile_name)

print(documents)

summarize(documents)

summarize_into_bullet_points(documents)
