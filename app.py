import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import yt_dlp
import re

# Load biến môi trường từ .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt tóm tắt
PROMPT = """Summarize the key points of the following YouTube transcript into bullet points, within 250 words. Keep it concise and informative:\n\n"""

# Hàm trích xuất video_id từ URL YouTube
def extract_video_id(youtube_url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_url)
    return match.group(1) if match else None

# Hàm lấy transcript bằng yt-dlp
def extract_transcript(video_url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["en", "vi"],  # Ưu tiên tiếng Anh, sau đó tiếng Việt
        "writeautomaticsub": True,  # Lấy auto-generated subtitles nếu không có bản gốc
        "outtmpl": "%(id)s.%(ext)s",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if not subtitles:
                return "No subtitles available for this video."

            # Ưu tiên phụ đề tiếng Anh hoặc tiếng Việt
            if "en" in subtitles:
                subtitle_url = subtitles["en"][0]["url"]
            elif "vi" in subtitles:
                subtitle_url = subtitles["vi"][0]["url"]
            else:
                return "No suitable subtitles found."

            # Tải phụ đề từ URL
            import requests
            response = requests.get(subtitle_url)
            return response.text if response.status_code == 200 else "Failed to fetch subtitles."

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# Hàm tóm tắt nội dung bằng Gemini AI
def generate_summary(transcript_text):
    if "Error" in transcript_text or "No subtitles" in transcript_text:
        return transcript_text  # Trả về lỗi thay vì gọi API
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(PROMPT + transcript_text)
    return response.text

# Giao diện Streamlit
st.title("YouTube Video Summarizer using Gemini AI")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = extract_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

# Nút để lấy tóm tắt
if st.button("Summarize Video"):
    if not youtube_link:
        st.warning("Please enter a valid YouTube link.")
    elif not video_id:
        st.error("Invalid YouTube link format.")
    else:
        transcript_text = extract_transcript(youtube_link)
        summary = generate_summary(transcript_text)

        st.markdown("## Summary:")
        st.write(summary)
