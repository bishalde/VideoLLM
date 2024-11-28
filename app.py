import os
import yt_dlp
import whisper
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle

# Initialize Whisper and SentenceTransformer models
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create directories
DOWNLOAD_DIR = "./downloads"
EMBEDDINGS_DIR = "./embeddings"
TRANSCRIPTS_DIR = "./transcripts"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# Function to download audio
def download_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "outtmpl": f"{DOWNLOAD_DIR}/%(id)s.%(ext)s",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info["id"], f"{DOWNLOAD_DIR}/{info['id']}.mp3"

# Function to transcribe audio
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    transcript = [seg["text"] for seg in result["segments"]]
    segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result["segments"]]
    return transcript, segments

# Function to generate embeddings
def generate_embeddings(transcript):
    return embedding_model.encode(transcript)

# Save transcript
def save_transcript(video_id, transcript):
    with open(f"{TRANSCRIPTS_DIR}/{video_id}.txt", "w") as f:
        f.write("\n".join(transcript))

# Save embeddings and segments
def save_data(video_id, embeddings, segments):
    with open(f"{EMBEDDINGS_DIR}/{video_id}.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "segments": segments}, f)

# Load embeddings and segments
def load_data(video_id):
    try:
        with open(f"{EMBEDDINGS_DIR}/{video_id}.pkl", "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["segments"]
    except FileNotFoundError:
        return None, None

# Query embeddings
def query_transcript(embeddings, segments, query, k=3):
    query_embedding = embedding_model.encode([query])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(query_embedding, k)
    results = [{"start": segments[i]["start"], "text": segments[i]["text"]} for i in indices[0]]
    return results

# Streamlit app
def main():
    st.title("Local Video Query App")
    
    # Initialize session state for video path and embeddings
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    
    # Video upload
    video_path = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])
    
    # Display uploaded video
    if video_path:
        st.session_state.video_path = video_path
        st.video(video_path)
        
        # Transcribe and process video if not already done
        if st.session_state.embeddings is None:
            with st.spinner("Transcribing audio..."):
                # Save uploaded video to local file
                video_id = video_path.name.split('.')[0]
                video_file_path = f"{DOWNLOAD_DIR}/{video_path.name}"
                with open(video_file_path, "wb") as f:
                    f.write(video_path.getbuffer())
                
                transcript, segments = transcribe_audio(video_file_path)
                save_transcript(video_id, transcript)
                embeddings = generate_embeddings([seg["text"] for seg in segments])
                save_data(video_id, embeddings, segments)
                
                # Store in session state
                st.session_state.embeddings = embeddings
                st.session_state.segments = segments
    
    # Query input and processing
    query = st.text_input("Enter your query")
    
    if st.session_state.video_path and query:
        with st.spinner("Processing query..."):
            results = query_transcript(
                st.session_state.embeddings, 
                st.session_state.segments, 
                query
            )
            st.success("Query processed successfully!")
            st.write("### Query Results:")
            
            # Create buttons for each result
            for i, result in enumerate(results):
                start_time = result["start"]
                st.write(f"**Match at {start_time:.2f}s:** {result['text']}")
                
                # Use a unique key for each button to prevent Streamlit state issues
                if st.button(f"Play at {int(start_time)}s", key=f"play_{i}"):
                    # Use session state video path to replay at specific timestamp
                    st.video(
                        st.session_state.video_path, 
                        start_time=start_time
                    )

if __name__ == "__main__":
    main()