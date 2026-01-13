import streamlit as st
import requests
import json
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
import os

# Configuration
st.set_page_config(
    page_title="Voice Identification System",
    page_icon="🎙️",
    layout="wide"
)

# Constants
API_BASE_URL = "http://localhost:8000/api/v1" 

# Load from environment or .env file
def load_config():
    """Load configuration from environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    config = {
        'aws_access_key': os.getenv('AWS_ACCESS_KEY', ''),
        'aws_secret_key': os.getenv('AWS_SECRET_KEY', ''),
        'aws_region': os.getenv('AWS_REGION', 'ap-southeast-1'),
        's3_bucket': os.getenv('S3_BUCKET_NAME', 's3-voice-identification'),
        'aws_endpoint': os.getenv('AWS_ENDPOINT_URL', '')
    }
    
    return config

def get_s3_client(config):
    """Create S3 client with proper credentials"""
    try:
        if not config['aws_access_key'] or not config['aws_secret_key']:
            raise ValueError("AWS credentials not found in environment")
        
        session = boto3.Session(
            aws_access_key_id=config['aws_access_key'],
            aws_secret_access_key=config['aws_secret_key'],
            region_name=config['aws_region']
        )
        
        client_config = {
            'config': Config(signature_version='s3v4')
        }
        
        if config['aws_endpoint']:
            st.info(f"🐳 Using LocalStack: {config['aws_endpoint']}")
        else:
            st.info(f"☁️ Using AWS: {config['aws_region']}")
        
        s3_client = session.client('s3', **client_config)
        
        # Test connection by accessing specific bucket
        try:
            s3_client.head_bucket(Bucket=config['s3_bucket'])
        except Exception as e:
            return None
            
        return s3_client
        
    except Exception as e:
        st.error(f"Failed to create S3 client: {e}")
        return None

def generate_presigned_url(s3_client, s3_key: str, bucket_name: str, expires_in: int = 3600) -> str:
    """Generate presigned URL with error handling"""
    try:
        # Check if object exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                st.error(f"Object not found: {s3_key}")
                return None
            else:
                st.error(f"Object access error: {e}")
                return None
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=expires_in
        )
        
        return url
        
    except Exception as e:
        st.error(f"Failed to generate presigned URL: {str(e)}")
        return None

def call_upload_api(api_key: str, file_content: bytes, filename: str) -> dict:
    """Call upload endpoint"""
    try:
        files = {'file': (filename, file_content)}
        data = {'api_key': api_key}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def call_process_api(api_key: str, audio_id: str) -> dict:
    """Call process endpoint"""
    try:
        data = {
            'api_key': api_key,
            'audio_id': audio_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/process",
            data=data,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Process failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        return None

def call_transcribe_api(api_key: str, audio_id_clean: str) -> dict:
    """Call transcribe endpoint"""
    try:
        data = {
            'api_key': api_key,
            'audio_id_clean': audio_id_clean
        }
        
        response = requests.post(
            f"{API_BASE_URL}/transcribe",
            data=data,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Transcribe failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Transcribe error: {str(e)}")
        return None

def call_get_full_transcript_api(api_key: str, audio_id_clean: str) -> dict:
    """Call get_full_transcript endpoint"""
    try:
        data = {
            'api_key': api_key,
            'audio_id_clean': audio_id_clean
        }
        
        response = requests.post(
            f"{API_BASE_URL}/get_full_transcript",
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Get transcript failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Get transcript error: {str(e)}")
        return None

def display_upload_result(result: dict, s3_client, bucket_name: str):
    """Display upload results with audio player"""
    if not result or not result.get('success'):
        return
        
    data = result.get('data', {})
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Audio ID:** `{data.get('audio_id', 'N/A')}`")
        st.info(f"**File Type:** {data.get('file_type', 'N/A')}")
        st.info(f"**Status:** {data.get('status', 'N/A')}")
    
    with col2:
        st.info(f"**File Name:** {data.get('file_name', 'N/A')}")
        st.info(f"**S3 Key:** `{data.get('s3_key', 'N/A')}`")
    
    # Audio player
    s3_key = data.get('s3_key')
    if s3_key and s3_client:
        st.subheader("Original Audio Player")
        presigned_url = generate_presigned_url(s3_client, s3_key, bucket_name)
        
        if presigned_url:
            st.audio(presigned_url)
            with st.expander("🔗 Presigned URL"):
                st.code(presigned_url)

def display_process_result(result: dict, s3_client, bucket_name: str):
    """Display process results with cleaned audio player"""
    if not result or not result.get('success'):
        return
        
    data = result.get('data', {})
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Audio ID", data.get('audio_id', 'N/A'))
    with col2:
        st.metric("Cleaned Audio ID", data.get('cleaned_audio_id', 'N/A'))
    with col3:
        st.metric("Tracks Count", data.get('tracks_count', 0))
    
    # NEW: Cleaned Audio Player
    cleaned_s3_key = data.get('cleaned_s3_key')
    if cleaned_s3_key and s3_client:
        st.subheader("Cleaned Audio Player")
        st.success("Cleaned audio available for playback")
        
        presigned_url = generate_presigned_url(s3_client, cleaned_s3_key, bucket_name)
        
        if presigned_url:
            st.audio(presigned_url, format='audio/wav')
            st.info("🔥 This is the processed audio with noise removed and only voice content")
            
            with st.expander("🔗 Cleaned Audio URL"):
                st.code(presigned_url)
        else:
            st.warning("Cannot generate presigned URL for cleaned audio")
    else:
        st.warning("Cleaned audio S3 key not available")
    
    # Processing details if available
    processing = data.get('processing', {})
    if processing:
        st.subheader("Processing Details")
        
        # Duration metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Duration", f"{processing.get('original_duration', 0):.2f}s")
        with col2:
            st.metric("Final Duration", f"{processing.get('final_duration', 0):.2f}s")
        with col3:
            st.metric("Duration Reduction", f"{processing.get('duration_reduction', 0):.2f}s")
        
        # Analysis details
        with st.expander("Noise Analysis"):
            noise = processing.get('noise_analysis', {})
            st.write(f"Noise segments: {noise.get('noise_segments_count', 0)}")
            st.write(f"Noise ratio: {noise.get('noise_ratio', 0):.2%}")
        
        with st.expander("VAD Analysis"):
            vad = processing.get('vad_analysis', {})
            st.write(f"Voice segments: {vad.get('original_voice_segments', 0)}")
            st.write(f"Voice activity ratio: {vad.get('voice_activity_ratio', 0):.2%}")
        
        with st.expander("Speaker Tracks"):
            tracks = processing.get('tracks', {}).get('tracks_detail', [])
            if tracks:
                for track in tracks:
                    st.write(f"**Speaker {track.get('speaker_id', 'Unknown')}**: {track.get('duration', 0):.2f}s ({track.get('coverage', 0):.1%} coverage)")
            else:
                st.write("No track details available")

def display_transcribe_result(result: dict, api_key: str):
    """Display transcription results with option to get full transcript"""
    if not result or not result.get('success'):
        return
        
    data = result.get('data', {})
    tracks = data.get('tracks', [])
    
    # Summary info
    summary = data.get('summary', {})
    if summary:
        st.subheader("Transcription Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tracks", summary.get('total_tracks', len(tracks)))
        with col2:
            st.metric("Total Words", summary.get('total_words', 0))
        with col3:
            st.metric("Total Characters", summary.get('total_characters', 0))
    
    if tracks:
        st.subheader("Transcription Results")
        
        for i, track in enumerate(tracks):
            speaker_id = track.get('speaker_id', i)
            words_count = track.get('words_count', 0)
            transcript = track.get('transcript', '')
            
            with st.expander(f"Speaker {speaker_id} ({words_count} words)", expanded=True):
                st.text_area(
                    f"Speaker {speaker_id} Transcript",
                    value=transcript,
                    height=150,
                    disabled=True
                )
                
                # Show word-level timings if available
                words = track.get('words', [])
                if words:
                    with st.expander("Word-level Timings"):
                        # Display first 10 words with timings
                        sample_words = words[:10]
                        for word_info in sample_words:
                            if isinstance(word_info, dict):
                                word_text = word_info.get('word', word_info.get('text', ''))
                                start_time = word_info.get('start', 0)
                                end_time = word_info.get('end', 0)
                                confidence = word_info.get('confidence', 0)
                                st.write(f"**{word_text}** ({start_time:.2f}s - {end_time:.2f}s) conf: {confidence:.2f}")
                        
                        if len(words) > 10:
                            st.write(f"... and {len(words) - 10} more words")
    else:
        st.warning("No transcription results found")
    
    # Get Full Transcript Button
    audio_id_clean = data.get('audio_id_clean')
    if audio_id_clean:
        st.divider()
        if st.button("Get Full Transcript with Details", type="secondary"):
            with st.spinner("Getting full transcript from database..."):
                full_result = call_get_full_transcript_api(api_key, audio_id_clean)
                
                if full_result and full_result.get('success'):
                    display_full_transcript_result(full_result)

def display_full_transcript_result(result: dict):
    """Display full transcript with detailed information"""
    if not result or not result.get('success'):
        st.error("Failed to get full transcript")
        return
    
    data = result.get('data', {})
    
    st.success("Full transcript retrieved from database")
    
    # Summary stats
    summary = data.get('summary', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tracks", summary.get('total_tracks', 0))
    with col2:
        st.metric("Total Words", summary.get('total_words', 0))
    with col3:
        st.metric("Total Characters", summary.get('total_characters', 0))
    with col4:
        st.metric("Transcribed Tracks", summary.get('transcribed_tracks', 0))
    
    # Full combined transcript
    full_transcript = data.get('full_transcript', '')
    if full_transcript:
        st.subheader("Complete Transcript")
        st.text_area(
            "Full Conversation",
            value=full_transcript,
            height=300,
            disabled=True
        )
        
        # Download button
        st.download_button(
            label="📥 Download Transcript (.txt)",
            data=full_transcript,
            file_name=f"transcript_{data.get('audio_id_clean', 'unknown')}.txt",
            mime="text/plain"
        )
    
    # Detailed per-track information
    tracks = data.get('tracks', [])
    if tracks:
        st.subheader("Detailed Track Information")
        
        for track in tracks:
            speaker_id = track.get('speaker_id', 'Unknown')
            transcript_data = track.get('transcript', {})
            segments = track.get('segments', [])
            
            with st.expander(f"Speaker {speaker_id} - Complete Details"):
                
                # Track summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{track.get('total_duration', 0):.2f}s")
                with col2:
                    st.metric("Coverage", f"{track.get('coverage', 0):.1f}%")
                with col3:
                    st.metric("Words", transcript_data.get('word_count', 0))
                
                # Full transcript for this speaker
                full_text = transcript_data.get('full_text', '')
                if full_text:
                    st.text_area(
                        f"Speaker {speaker_id} - Full Text",
                        value=full_text,
                        height=200,
                        disabled=True
                    )
                
                # Word-level data
                words = transcript_data.get('words', [])
                if words:
                    with st.expander("📝 All Words with Timings"):
                        # Create a formatted view of words
                        words_text = []
                        for i, word_info in enumerate(words):
                            if isinstance(word_info, dict):
                                word_text = word_info.get('word', word_info.get('text', ''))
                                start_time = word_info.get('start', 0)
                                end_time = word_info.get('end', 0)
                                words_text.append(f"{i+1:3d}. {word_text:15s} ({start_time:6.2f}s - {end_time:6.2f}s)")
                        
                        if words_text:
                            st.text_area(
                                f"Word Timings ({len(words_text)} words)",
                                value='\n'.join(words_text),
                                height=200,
                                disabled=True
                            )
                
                # Segment information
                if segments:
                    with st.expander("🔧 Audio Segments"):
                        for seg in segments:
                            st.write(f"**{seg.get('segment_type', 'unknown')}**: {seg.get('start_time', 0):.2f}s - {seg.get('end_time', 0):.2f}s (confidence: {seg.get('confidence', 0):.2f})")

def main():
    st.title("🎙️ Voice Identification System")
    st.markdown("**Enhanced Vietnamese Voice Processing & Recognition**")
    
    # Load configuration
    config = load_config()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API settings
        api_key = st.text_input(
            "API Key", 
            type="password",
            placeholder="sk_live_..."
        )
    
    if not api_key:
        st.warning("Please enter API Key to continue")
        return
    
    if not config['aws_access_key'] or not config['aws_secret_key']:
        st.error("AWS credentials not configured")
        st.info("Please set AWS_ACCESS_KEY and AWS_SECRET_KEY in your .env file")
        return
    
    # Create S3 client
    s3_client = get_s3_client(config)
    if not s3_client:
        st.error("Cannot connect to AWS S3")
        return
    
    # Initialize session state
    if 'upload_result' not in st.session_state:
        st.session_state.upload_result = None
    if 'process_result' not in st.session_state:
        st.session_state.process_result = None
    if 'transcribe_result' not in st.session_state:
        st.session_state.transcribe_result = None
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Upload", "Process", "Transcribe"])
    
    # === TAB 1: UPLOAD ===
    with tab1:
        st.header("Upload Audio/Video")
        
        uploaded_file = st.file_uploader(
            "Choose audio or video file",
            type=['mp3', 'wav', 'ogg', 'mp4', 'avi', 'mov', 'm4a', 'flac'],
            help="Supported: mp3, wav, ogg, m4a, flac, mp4, avi, mov"
        )
        
        if uploaded_file:
            st.success(f"File selected: {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            
            if st.button("Upload File", type="primary"):
                with st.spinner("Uploading file..."):
                    file_content = uploaded_file.read()
                    result = call_upload_api(api_key, file_content, uploaded_file.name)
                    
                    if result:
                        st.session_state.upload_result = result
                        display_upload_result(result, s3_client, config['s3_bucket'])
        
        # Show previous result
        if st.session_state.upload_result:
            st.divider()
            st.subheader("Previous Upload")
            data = st.session_state.upload_result.get('data', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"Audio ID: {data.get('audio_id', 'N/A')}")
            with col2:
                if st.button("Play Again"):
                    s3_key = data.get('s3_key')
                    if s3_key:
                        presigned_url = generate_presigned_url(s3_client, s3_key, config['s3_bucket'])
                        if presigned_url:
                            st.audio(presigned_url)
    
    # === TAB 2: PROCESS ===
    with tab2:
        st.header("Process Audio")
        
        # Get audio_id from upload or manual input
        default_audio_id = ""
        if (st.session_state.upload_result and 
            st.session_state.upload_result.get('success') and 
            st.session_state.upload_result.get('data')):
            default_audio_id = st.session_state.upload_result['data'].get('audio_id', '')
        
        audio_id_input = st.text_input(
            "Audio ID to process",
            value=default_audio_id,
            placeholder="Enter audio_id or upload file in previous tab"
        )
        
        if audio_id_input:
            if st.button("Process Audio", type="primary"):
                with st.spinner("Processing audio (noise reduction, VAD, OSD, separation)..."):
                    result = call_process_api(api_key, audio_id_input)
                    
                    if result:
                        st.session_state.process_result = result
                        display_process_result(result, s3_client, config['s3_bucket'])
        
        # Show previous result
        if st.session_state.process_result:
            st.divider()
            st.subheader("Previous Process")
            data = st.session_state.process_result.get('data', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.code(f"Cleaned Audio ID: {data.get('cleaned_audio_id', 'N/A')}")
            with col2:
                st.code(f"Tracks: {data.get('tracks_count', 0)}")
            with col3:
                if data.get('cleaned_s3_key') and st.button("Play Cleaned"):
                    presigned_url = generate_presigned_url(s3_client, data['cleaned_s3_key'], config['s3_bucket'])
                    if presigned_url:
                        st.audio(presigned_url, format='audio/wav')
    
    # === TAB 3: TRANSCRIBE ===
    with tab3:
        st.header("Transcribe Audio")
        
        # Get cleaned_audio_id from process or manual input
        default_cleaned_id = ""
        if (st.session_state.process_result and 
            st.session_state.process_result.get('success') and 
            st.session_state.process_result.get('data')):
            default_cleaned_id = st.session_state.process_result['data'].get('cleaned_audio_id', '')
        
        cleaned_audio_id_input = st.text_input(
            "Cleaned Audio ID to transcribe",
            value=default_cleaned_id,
            placeholder="Enter cleaned_audio_id or process audio in previous tab"
        )
        
        if cleaned_audio_id_input:
            if st.button("Transcribe Audio", type="primary"):
                with st.spinner("Transcribing audio with Whisper..."):
                    result = call_transcribe_api(api_key, cleaned_audio_id_input)
                    
                    if result:
                        st.session_state.transcribe_result = result
                        display_transcribe_result(result, api_key)
        
        # Show previous result
        if st.session_state.transcribe_result:
            st.divider()
            st.subheader("Previous Transcribe")
            data = st.session_state.transcribe_result.get('data', {})
            
            tracks = data.get('tracks', [])
            if tracks:
                for track in tracks:
                    speaker_id = track.get('speaker_id', 'Unknown')
                    transcript = track.get('transcript', '')
                    preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
                    st.info(f"**Speaker {speaker_id}:** {preview}")
            
            # Quick access to full transcript
            audio_id_clean = data.get('audio_id_clean')
            if audio_id_clean and st.button("Get Full Transcript", key="prev_transcript"):
                with st.spinner("Getting full transcript..."):
                    full_result = call_get_full_transcript_api(api_key, audio_id_clean)
                    if full_result:
                        display_full_transcript_result(full_result)

if __name__ == "__main__":
    main()