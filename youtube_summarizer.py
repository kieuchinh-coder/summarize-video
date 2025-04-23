import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import re
import math
import subprocess
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import requests
from gtts import gTTS
import urllib.parse
import shutil
from pathlib import Path


# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable. Please add it to your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Summary prompt template
PROMPT = """Summarize the key points of the following YouTube transcript into:
1. A concise title that captures the main topic.
2. 5-8 bullet points highlighting the main ideas (this section should be around 250 words to 300 words).
3. Identify 4-5 key timestamps/sections that are most important.

Maintain the original meaning and ensure clarity in the summary:

"""

# Define permanent download directory
PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = PROJECT_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Check if FFmpeg is installed
def check_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def run_ffmpeg_command(command, description=""):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"FFmpeg success: {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed while {description}")
        return False, f"FFmpeg error: {e.stderr}"


# Function to extract video_id from YouTube URL
def extract_video_id(youtube_url):
    # Handle different URL formats (regular, shortened, embedded, etc.)
    query = urllib.parse.urlparse(youtube_url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urllib.parse.parse_qs(query.query)
            return p['v'][0]
        if query.path.startswith(('/embed/', '/v/')):
            return query.path.split('/')[2]
    return None

# Function to get video details (title, duration, etc.)
def get_video_details(video_id):
    try:
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return {
                'title': info.get('title', 'Unknown title'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', None),
                'channel': info.get('channel', 'Unknown channel'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown date')
            }
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return {
            'title': 'Unknown title',
            'duration': 0,
            'thumbnail': None,
            'channel': 'Unknown channel',
            'view_count': 0,
            'upload_date': 'Unknown date'
        }

# Primary function to get transcript with YouTubeTranscriptApi
def get_transcript_primary(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript_list
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.warning(f"Primary transcript extraction failed: {str(e)}")
        return None

# Backup function to get transcript with yt-dlp
def get_transcript_backup(video_url):
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": ["en", "vi"],  # Prioritize English, then Vietnamese
            "writeautomaticsub": True,
            "outtmpl": "%(id)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if not subtitles:
                return None

            # Prioritize English or Vietnamese subtitles
            lang = next((l for l in ["en", "vi"] if l in subtitles), None)
            if not lang:
                lang = next(iter(subtitles))  # Get the first available language

            subtitle_url = subtitles[lang][0]["url"]
            
            # Download subtitles
            response = requests.get(subtitle_url)
            if response.status_code == 200:
                # Parse the subtitle content to match the format from YouTubeTranscriptApi
                # This is a simplified approach and might need adjustment based on the format
                lines = response.text.split("\n")
                transcript_list = []
                for i in range(1, len(lines), 4):
                    if i+2 < len(lines):
                        try:
                            time_line = lines[i]
                            time_parts = time_line.split(" --> ")
                            start_parts = time_parts[0].replace(",", ".").split(":")
                            start = float(start_parts[0])*3600 + float(start_parts[1])*60 + float(start_parts[2])
                            duration = 5.0  # Approximation
                            text = lines[i+1]
                            transcript_list.append({
                                "text": text,
                                "start": start,
                                "duration": duration
                            })
                        except (IndexError, ValueError):
                            continue
                return transcript_list
        return None
    except Exception as e:
        st.warning(f"Backup transcript extraction failed: {str(e)}")
        return None

# Combined function to get transcript
def get_transcript(video_id, video_url):
    transcript_list = get_transcript_primary(video_id)
    if not transcript_list:
        st.info("Primary transcript extraction failed. Trying backup method...")
        transcript_list = get_transcript_backup(video_url)
    
    if not transcript_list:
        st.error("Could not extract transcript from this video. Please try another video.")
        return None
    
    # Convert transcript to plain text
    full_transcript = " ".join([item['text'] for item in transcript_list])
    return {"text": full_transcript, "segments": transcript_list}

# Generate summary with Gemini AI
def generate_summary(transcript_data):
    if not transcript_data:
        return None
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        full_prompt = PROMPT + transcript_data["text"]
        
        # Generate the summary
        response = model.generate_content(full_prompt)
        
        # Process the summary to extract title, bullet points and key timestamps
        summary = response.text
        
        # Return both the full summary and the transcript segments for timestamp detection
        return {
            "summary": summary,
            "transcript_segments": transcript_data["segments"]
        }
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Basic sentence tokenizer using regex
def split_into_sentences(text):
    # Split on sentence-ending punctuation followed by space and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Function to find key moments in the video based on summary
def find_key_moments(summary_data, video_duration):
    if not summary_data:
        return []
    
    # Extract sentences from the summary
    summary_text = summary_data["summary"]
    sentences = split_into_sentences(summary_text)
    
    # Find timestamps for important moments
    key_moments = []
    transcript_segments = summary_data["transcript_segments"]
    
    # Check for explicitly mentioned timestamps in the summary
    timestamp_pattern = r'(\d+:\d+|\d+\s*minutes?|\d+\s*seconds?)'
    explicit_timestamps = re.findall(timestamp_pattern, summary_text)
    
    if explicit_timestamps:
        # Process explicit timestamps
        for timestamp in explicit_timestamps:
            try:
                # Convert various formats to seconds
                if ':' in timestamp:
                    parts = timestamp.split(':')
                    time_seconds = int(parts[0]) * 60 + int(parts[1])
                elif 'minute' in timestamp:
                    time_seconds = int(re.search(r'\d+', timestamp).group()) * 60
                elif 'second' in timestamp:
                    time_seconds = int(re.search(r'\d+', timestamp).group())
                
                if time_seconds < video_duration:
                    key_moments.append(time_seconds)
            except:
                continue
    
    return key_moments

# Function to download the video
def download_video(video_id, output_path):
    try:
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit quality to 720p
            'outtmpl': output_path,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        return True
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return False

# Add this new function to your code
def prepare_voiceover_text(summary_text):
    # Split the summary by common section markers
    sections = re.split(r'Key Timestamps/Sections:', summary_text, flags=re.IGNORECASE)
    
    # The first part should contain the title and main bullet points
    if len(sections) > 0:
        # Further clean up the text to ensure good voiceover quality
        voiceover_text = sections[0].strip()
        
        # Optional: remove bullet points and numbered list markers for smoother reading
        voiceover_text = re.sub(r'^\s*[\•\-\*\d]+\.?\s+', '', voiceover_text, flags=re.MULTILINE)
        
        return voiceover_text
    
    # If no clear section dividers found, return the original
    return summary_text

# Function to create text-to-speech audio from summary (modified)
def create_voiceover(summary_text, output_path):
    try:
        # Filter the summary to exclude timestamp sections
        voiceover_text = prepare_voiceover_text(summary_text)
        
        # Create TTS with the filtered text
        tts = gTTS(text=voiceover_text, lang='en', slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        st.error(f"Error creating voiceover: {str(e)}")
        return False

    
# Function to create clips from key moments using FFmpeg
def create_clip_from_moment(input_video, output_clip, start_time, duration):
    """Create a video clip from a specific time point using FFmpeg."""
    command = [
        'ffmpeg', '-y',  # Overwrite output files without asking
        '-i', input_video,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'fast',
        output_clip
    ]
    
    return run_ffmpeg_command(command, f"creating clip at {start_time}")

# Function to create concatenation file for FFmpeg
def create_concat_file(clips, concat_file_path):
    """Create a text file for FFmpeg concatenation."""
    with open(concat_file_path, 'w') as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")
    return True

def get_media_duration(file_path):
    """Get the duration of a media file using ffprobe with proper error handling."""
    if not os.path.exists(file_path):
        st.error(f"File does not exist: {file_path}")
        return None

    try:
        probe_command = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(probe_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.warning(f"FFprobe error: {result.stderr}")
            return None
            
        # Check if output is not empty and is a valid float
        if result.stdout and result.stdout.strip():
            try:
                return float(result.stdout.strip())
            except ValueError:
                st.warning(f"Could not convert duration to float: {result.stdout}")
                return None
        else:
            st.warning("FFprobe returned empty duration")
            return None
            
    except Exception as e:
        st.error(f"Error getting media duration: {str(e)}")
        return None

def create_summary_video(video_path,summary_text, key_moments, output_path, audio_path, temp_dir):
    try:
        # Check if FFmpeg is installed
        if not check_ffmpeg_installed():
            st.error("FFmpeg is not installed. Please install FFmpeg to create summary videos.")
            return False
        
        # Validate inputs
        if not os.path.exists(video_path):
            st.error(f"Video file not found: {video_path}")
            return False
            
        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return False
            
        if not key_moments or len(key_moments) == 0:
            st.warning("No key moments detected. Using default moments.")
            # Create some default moments if none were found
            video_duration = get_media_duration(video_path)
            if video_duration:
                key_moments = [
                    video_duration * 0.1,
                    video_duration * 0.3,
                    video_duration * 0.5,
                    video_duration * 0.7
                ]
            else:
                key_moments = [10, 30, 60, 90]  # Fallback default moments
        
        # Create clips from key moments
        clip_paths = []
        for i, moment in enumerate(key_moments):
            # Define parameters for each clip
            start_time = max(0, moment - 2)  # 5 seconds before the key moment
            duration = 15
            
            # Create output path for this clip
            clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
            
            # Create the clip
            success = create_clip_from_moment(video_path, clip_path, start_time, duration)
            if not success:
                st.error(f"Failed to create clip at moment {moment}")
                continue  # Try next clip instead of failing completely
            
            # Verify the clip was created
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clip_paths.append(clip_path)
            else:
                st.warning(f"Clip file is missing or empty: {clip_path}")
        
        # Ensure we have at least one clip
        if not clip_paths:
            st.error("Failed to create any valid clips")
            return False
        
        # Create concatenation file
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        create_concat_file(clip_paths, concat_file)
        
        if not os.path.exists(concat_file):
            st.error("Failed to create concatenation file")
            return False
        
        # Concatenate all clips
        concatenated_path = os.path.join(temp_dir, "concatenated.mp4")
        concat_command = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            concatenated_path
        ]
        success = run_ffmpeg_command(concat_command, "concatenating clips")
        if not success:
            st.error("Failed to concatenate video clips")
            return False
        
        # Verify concatenated file exists
        if not os.path.exists(concatenated_path) or os.path.getsize(concatenated_path) == 0:
            st.error("Concatenated video file is missing or empty")
            return False
        
        # Get duration of concatenated video
        video_duration = get_media_duration(concatenated_path)
        if video_duration is None:
            st.error("Could not determine video duration")
            return False

        # Get duration of audio
        audio_duration = get_media_duration(audio_path)
        if audio_duration is None:
            st.error("Could not determine audio duration")
            return False
        
        st.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        
        # Handle duration mismatch
        if abs(video_duration - audio_duration) > 1:  # More than 1 second difference
            if video_duration > audio_duration:
                # Trim video to match audio
                trimmed_path = os.path.join(temp_dir, "trimmed.mp4")
                trim_command = [
                    'ffmpeg', '-y',
                    '-i', concatenated_path,
                    '-t', str(audio_duration),
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    trimmed_path
                ]
                success = run_ffmpeg_command(trim_command, "trimming video")
                if not success:
                    st.warning("Failed to trim video, using original concatenated video")
                else:
                    if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                        concatenated_path = trimmed_path
                    else:
                        st.warning("Trimmed video file is missing or empty")
            else:
                # Extend video by looping the last part
                st.info("Audio is longer than video, extending video...")
                # Simple approach: loop the whole video if needed
                extended_path = os.path.join(temp_dir, "extended.mp4")
                loop_count = math.ceil(audio_duration / video_duration)
                
                # Create a file with repeated entries
                loop_list = os.path.join(temp_dir, "loop_list.txt")
                with open(loop_list, 'w') as f:
                    for _ in range(loop_count):
                        f.write(f"file '{os.path.basename(concatenated_path)}'\n")
                
                # Concatenate the repeated entries
                loop_command = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', loop_list,
                    '-c', 'copy',
                    extended_path
                ]
                success = run_ffmpeg_command(loop_command, "extending video")
                if success and os.path.exists(extended_path) and os.path.getsize(extended_path) > 0:
                    # Trim to exact audio duration
                    final_trim_path = os.path.join(temp_dir, "final_trim.mp4")
                    trim_command = [
                        'ffmpeg', '-y',
                        '-i', extended_path,
                        '-t', str(audio_duration),
                        '-c:v', 'copy',
                        '-c:a', 'copy',
                        final_trim_path
                    ]
                    if run_ffmpeg_command(trim_command, "final trimming") and os.path.exists(final_trim_path):
                        concatenated_path = final_trim_path
                    else:
                        concatenated_path = extended_path
                else:
                    st.warning("Failed to extend video, using original")
        
        # Add audio to the final video
        final_command = [
            'ffmpeg', '-y',
            '-i', concatenated_path,
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        success = run_ffmpeg_command(final_command, "adding audio to video")
        if not success:
            st.error("Failed to add audio to video")
            return False
        
        # Verify final output exists
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            st.error("Final video file is missing or empty")
            return False
        
        st.success("Summary video created successfully!")
        return True
    
    except Exception as e:
        st.error(f"Error creating summary video: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

# Main function for Streamlit app
def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 YouTube Video Summarizer")
    st.markdown("Automatically generate concise video summaries with AI")
    
    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        st.error("FFmpeg is not installed. Please install FFmpeg to use this application.")
        st.info("Installation instructions: https://ffmpeg.org/download.html")
        return
    
    # Input for YouTube URL
    youtube_url = st.text_input("Enter YouTube Video URL:")
    
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid YouTube video link.")
        else:
            # Get video details and display thumbnail
            video_details = get_video_details(video_id)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
            with col2:
                st.subheader(video_details['title'])
                st.write(f"Channel: {video_details['channel']}")
                st.write(f"Duration: {video_details['duration'] // 60}m {video_details['duration'] % 60}s")
                st.write(f"Views: {video_details['view_count']:,}")
            
            # Process button
            if st.button("Generate Summary Video"):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a temporary directory for processing files
                temp_dir = str(DOWNLOAD_DIR)
                    # Step 1: Extract transcript
                status_text.text("Extracting transcript...")
                transcript_data = get_transcript(video_id, youtube_url)
                progress_bar.progress(20)
                    
                if transcript_data:
                    # Step 2: Generate summary
                    status_text.text("Generating AI summary...")
                    summary_data = generate_summary(transcript_data)
                    progress_bar.progress(40)
                        
                    if summary_data:
                        # Display the text summary
                        st.markdown("### Summary")
                        st.markdown(summary_data["summary"])
                            
                        # Step 3: Find key moments
                        status_text.text("Identifying key moments in video...")
                        key_moments = find_key_moments(summary_data, video_details['duration'])
                        progress_bar.progress(50)
                            
                        # Step 4: Download video
                        video_path = os.path.join(temp_dir, f"{video_id}.mp4")
                        audio_path = os.path.join(temp_dir, f"{video_id}_voiceover.mp3")
                        output_path = os.path.join(temp_dir, f"{video_id}_summary.mp4")
                        final_output_path = f"{video_id}_summary.mp4"
                            
                        status_text.text("Downloading video...")
                        download_success = download_video(video_id, video_path)
                        progress_bar.progress(70)
                            
                        if download_success:
                            # Step 5: Create voiceover
                            status_text.text("Creating voiceover...")
                            voiceover_success = create_voiceover(summary_data["summary"], audio_path)
                            progress_bar.progress(80)
                                
                            if voiceover_success:
                                # Step 6: Create summary video
                                status_text.text("Creating summary video...")
                                video_result = create_summary_video(
                                        video_path, 
                                        summary_data["summary"],
                                        key_moments,
                                        output_path,
                                        audio_path,
                                        temp_dir
                                )
                                progress_bar.progress(95)
                                    
                                if isinstance(video_result, str) and video_result.startswith("Error"):
                                    st.error(video_result)
                                else:
                                    # Copy the output file to a permanent location
                                    shutil.copy(output_path, final_output_path)
                                        
                                    # Read the video file
                                    with open(final_output_path, 'rb') as file:
                                        video_bytes = file.read()
                                        
                                        # Step 7: Display final video
                                    status_text.text("Summary video created successfully!")
                                    progress_bar.progress(100)
                                        
                                    st.markdown("### Summary Video")
                                    st.video(video_bytes)
                                        
                                    # Provide download button
                                    st.download_button(
                                            label="Download Summary Video",
                                            data=video_bytes,
                                            file_name=f"{video_id}_summary.mp4",
                                            mime="video/mp4"
                                    )
                            else:
                                status_text.text("Failed to create voiceover.")
                        else:
                            status_text.text("Failed to download video.")
                    else:
                        status_text.text("Failed to generate summary.")
                else:
                    status_text.text("Failed to extract transcript.")
    
    # Add information about the app at the bottom
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
    """
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 20px;">
        <div style="background-color: #f0f2f6; padding: 15px 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 250px;">
            <strong>1. Enter a YouTube URL</strong><br>
            Provide the link to any YouTube video.
        </div>
        <div style="background-color: #f0f2f6; padding: 15px 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 250px;">
            <strong>2. Extract the transcript</strong><br>
            The app extracts the video's subtitles.
        </div>
        <div style="background-color: #f0f2f6; padding: 15px 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 250px;">
            <strong>3. Generate AI summary</strong><br>
            Gemini AI creates a concise summary.
        </div>
        <div style="background-color: #f0f2f6; padding: 15px 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 250px;">
            <strong>4. Create summary video</strong><br>
            The app selects key moments and adds voiceover.
        </div>
        <div style="background-color: #f0f2f6; padding: 15px 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 250px;">
            <strong>5. Download or share</strong><br>
            Save the summarized video for later use.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Run the app
if __name__ == "__main__":
    main()