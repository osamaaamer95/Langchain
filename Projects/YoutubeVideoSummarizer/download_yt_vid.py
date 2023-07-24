import yt_dlp


def download_yt_vid(url: str, filename: str):
    # Set the options for the download
    ydl_opts = {
        'format': 'best.2[ext=mp4]+best.2[ext=m4a]/best.2[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
