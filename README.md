# 🎥 YouTube Video Summarizer with AI

A Streamlit-based web app that summarizes YouTube videos using **Gemini AI**, generates a voiceover, and creates a highlight video with the most important moments. Perfect for turning long videos into short, informative clips in just a few minutes!

---

## 🚀 Features

- 📄 **Transcript Extraction**  
  Retrieves transcript using YouTubeTranscriptAPI or auto-generated captions via `yt-dlp`.

- 🧠 **AI Summarization (Gemini)**  
  Uses Google’s Gemini AI to generate a title, bullet-point summary, and key timestamps from the transcript.

- 🔍 **Key Moment Detection**  
  Identifies important timestamps from the summary and extracts related video clips.

- 🔈 **Voiceover Generation**  
  Converts the AI-generated summary into natural speech using Google Text-to-Speech (gTTS).

- 🎬 **Video Editing with FFmpeg**  
  Clips and merges relevant video parts to create a short highlight summary video.

- 🌐 **Simple Web Interface**  
  Built with Streamlit for ease of use and fast deployment.

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **AI Summarization:** Gemini API (`google.generativeai`)  
- **Transcript Extraction:** YouTubeTranscriptAPI, yt-dlp  
- **Voiceover:** gTTS  
- **Video Editing:** FFmpeg  
- **Utilities:** Python, Regex, subprocess, re, urllib

---
1. Install Dependencies
   
pip install -r requirements.txt
---
3. Add Your Gemini API Key
Create a .env file in the root directory:

GEMINI_API_KEY=your_google_genai_api_key
---
5. Run the App

streamlit run main.py
---
📦 Requirements
Python 3.8+

FFmpeg (must be installed and added to system PATH)

Google Gemini API access

gTTS

yt-dlp
---

📝  Improvements
 Add multi-language support for voiceover

 Custom summary length option

 Drag-and-drop video input

 Embed summarized video in UI

 Add download buttons for all output files
---
