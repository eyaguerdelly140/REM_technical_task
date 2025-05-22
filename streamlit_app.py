import streamlit as st
import os
import json
import tempfile
import subprocess
import requests
from typing import Dict, Optional
import openai
import yt_dlp

# Page config
st.set_page_config(
    page_title="REM Waste Accent Detection Tool",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎙️ REM Waste Accent Detection Tool</h1>
    <h3>Automated English Accent Analysis for Hiring Decisions</h3>
    <p>Built with OpenAI GPT-4o & Whisper | Professional Candidate Evaluation</p>
</div>
""", unsafe_allow_html=True)

class REMWasteAccentDetector:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.temp_dir = "/tmp/rem_waste_accent"
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_audio_from_url(self, video_url: str) -> str:
        """Extract audio from public video URL"""
        st.info(f"🎬 Extracting audio from: {video_url[:50]}...")
        
        try:
            output_template = os.path.join(self.temp_dir, 'video.%(ext)s')
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3'
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

                for file in os.listdir(self.temp_dir):
                    if file.startswith('video.'):
                        input_path = os.path.join(self.temp_dir, file)
                        output_path = os.path.join(self.temp_dir, 'extracted_audio.mp3')

                        subprocess.run([
                            'ffmpeg', '-i', input_path, '-vn', '-codec:a', 'mp3',
                            '-b:a', '128k', '-ac', '1', '-ar', '16000',
                            output_path, '-y'
                        ], check=True, capture_output=True)

                        os.remove(input_path)
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        st.success(f"✅ Audio extracted: {file_size:.1f} MB")
                        return output_path

        except Exception as e:
            st.error(f"⚠️ Audio extraction failed: {str(e)}")
            raise Exception("❌ Could not extract audio from URL")

    def transcribe_speech(self, audio_path: str) -> str:
        """Transcribe audio using OpenAI Whisper"""
        st.info("🎙️ Transcribing speech with OpenAI Whisper...")
        
        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        
        if file_size > 24:
            st.warning("⚠️ File too large, compressing...")
            compressed_path = os.path.join(self.temp_dir, 'compressed_audio.mp3')
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-codec:a', 'mp3',
                '-b:a', '64k', '-ac', '1', '-ar', '16000',
                compressed_path, '-y'
            ], check=True, capture_output=True)
            
            os.remove(audio_path)
            audio_path = compressed_path
            file_size = os.path.getsize(audio_path) / (1024 * 1024)

        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

            transcription = transcript.text.strip()
            st.success(f"✅ Transcription complete: {len(transcription)} characters, {len(transcription.split())} words")
            return transcription

        except Exception as e:
            raise Exception(f"Speech transcription failed: {str(e)}")

    def classify_english_accent(self, transcription: str) -> Dict:
        """Classify English accent using GPT-4o"""
        st.info("🧠 Analyzing English accent with GPT-4o...")
        
        if len(transcription.strip()) < 15:
            return {
                'classification': 'Insufficient Sample',
                'confidence': 0,
                'explanation': 'Transcription too short for reliable accent analysis'
            }

        prompt = f"""
You are analyzing English speech for hiring evaluation. Classify this speaker's English accent.

TRANSCRIPTION: "{transcription}"

Determine the accent from: American, British, Australian, Canadian, Irish, Scottish, South African, Indian, or Other
Provide confidence score (0-100%) and explain with specific examples.

Respond in JSON format:
{{
    "classification": "accent_category",
    "confidence": confidence_number,
    "explanation": "detailed explanation with evidence"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert English accent classifier. Be evidence-based."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()
            
            if not content.startswith('{'):
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    content = content[start_idx:end_idx]

            result = json.loads(content)
            
            classification = result.get('classification', 'Unknown')
            confidence = float(result.get('confidence', 50))
            explanation = result.get('explanation', 'Analysis completed')

            word_count = len(transcription.split())
            if word_count < 20:
                confidence = min(confidence, 50)
            elif word_count < 40:
                confidence = min(confidence, 75)

            st.success(f"✅ Classification: {classification} ({confidence}% confidence)")

            return {
                'classification': classification,
                'confidence': round(confidence, 1),
                'explanation': explanation
            }

        except Exception as e:
            st.error(f"⚠️ Analysis failed: {e}")
            return {
                'classification': 'Analysis Failed',
                'confidence': 0,
                'explanation': f'Processing error: {str(e)}'
            }

    def analyze_candidate_video(self, video_url: str) -> Dict:
        """Complete analysis pipeline"""
        try:
            audio_path = self.extract_audio_from_url(video_url)
            transcription = self.transcribe_speech(audio_path)
            accent_analysis = self.classify_english_accent(transcription)

            result = {
                'success': True,
                'candidate_url': video_url,
                'transcription': transcription,
                'accent_classification': accent_analysis['classification'],
                'confidence_score': accent_analysis['confidence'],
                'explanation': accent_analysis['explanation'],
                'word_count': len(transcription.split()),
                'processing_status': 'Analysis Complete',
                'suitable_for_hiring': accent_analysis['confidence'] >= 60,
                'recommendation': self._generate_recommendation(accent_analysis)
            }

            try:
                os.remove(audio_path)
            except:
                pass

            return result

        except Exception as e:
            return {
                'success': False,
                'candidate_url': video_url,
                'error': str(e),
                'accent_classification': 'Analysis Failed',
                'confidence_score': 0,
                'processing_status': 'Failed',
                'suitable_for_hiring': False,
                'recommendation': 'Manual review required'
            }

    def _generate_recommendation(self, analysis: Dict) -> str:
        confidence = analysis['confidence']
        classification = analysis['classification']
        
        if confidence >= 80:
            return f"High confidence {classification} English speaker - Recommend for English-speaking roles"
        elif confidence >= 60:
            return f"Good {classification} English proficiency - Suitable for most roles"
        elif confidence >= 40:
            return f"Moderate English proficiency detected - Consider role requirements"
        else:
            return "Low confidence in accent classification - Manual review recommended"

# Main app
def main():
    # Sidebar for API key
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")

    # Main interface
    st.header("📥 Candidate Video Analysis")
    st.write("Enter the candidate's video URL (Loom recording, Vimeo, YouTube, or direct video link)")

    # Input form
    with st.form("analysis_form"):
        video_url = st.text_input(
            "Video URL:",
            placeholder="https://www.loom.com/share/candidate-recording or https://vimeo.com/123456789"
        )
        
        submitted = st.form_submit_button("🚀 Analyze Candidate", type="primary")

    if submitted:
        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
            return
            
        if not video_url:
            st.error("❌ Please enter a candidate video URL")
            return
            
        if not video_url.startswith(('http://', 'https://')):
            st.error("❌ Please enter a valid URL starting with http:// or https://")
            return

        # Initialize detector
        detector = REMWasteAccentDetector(api_key)

        # Show progress
        with st.spinner("🔄 Processing candidate video... This may take 1-3 minutes..."):
            result = detector.analyze_candidate_video(video_url)

        # Display results
        if result['success']:
            st.success("🎯 CANDIDATE ACCENT ANALYSIS COMPLETED")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accent Classification", result['accent_classification'])
            
            with col2:
                st.metric("Confidence Score", f"{result['confidence_score']}%")
            
            with col3:
                suitable = "✅ SUITABLE" if result['suitable_for_hiring'] else "⚠️ REVIEW NEEDED"
                st.metric("Hiring Suitability", suitable)

            # Detailed results
            st.subheader("💼 Hiring Recommendation")
            st.info(result['recommendation'])

            st.subheader("🔍 Detailed Analysis")
            st.write(f"**Words Analyzed:** {result['word_count']}")
            st.write(f"**Processing Status:** {result['processing_status']}")
            st.write(f"**Explanation:** {result['explanation']}")

            # Transcription
            st.subheader("📜 Candidate Speech Sample")
            with st.expander("View Full Transcription"):
                st.text_area("Transcription", result['transcription'], height=200)

            # Export options
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download JSON Report"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"candidate_analysis_{result['accent_classification'].lower()}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📝 Download Text Summary"):
                    summary = f"""REM WASTE - CANDIDATE ACCENT ANALYSIS REPORT
{'='*60}

Candidate URL: {result['candidate_url']}
Accent Classification: {result['accent_classification']}
Confidence Score: {result['confidence_score']}%
Hiring Suitability: {'SUITABLE' if result['suitable_for_hiring'] else 'REVIEW NEEDED'}
Recommendation: {result['recommendation']}

DETAILED ANALYSIS:
{result['explanation']}

CANDIDATE SPEECH TRANSCRIPTION:
{result['transcription']}
"""
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"candidate_summary_{result['accent_classification'].lower()}.txt",
                        mime="text/plain"
                    )

        else:
            st.error("❌ ANALYSIS FAILED")
            st.error(f"Error: {result['error']}")
            
            st.info("💡 Troubleshooting:")
            st.write("• Ensure URL is publicly accessible")
            st.write("• Try with Loom, Vimeo, or YouTube links")
            st.write("• Check that video contains clear English speech")

    # Instructions
    with st.expander("📋 Usage Instructions"):
        st.markdown("""
        **For REM Waste Hiring Team:**
        
        1. 🎬 Ask candidates to submit Loom recordings or video links
        2. 📝 Ensure videos contain at least 30 seconds of clear English speech
        3. 🔗 Paste the public video URL in the field above
        4. ⚡ Get automated accent analysis in 1-3 minutes
        5. 📊 Use confidence scores to guide hiring decisions
        
        **Example Video Formats:**
        - **Loom:** https://www.loom.com/share/abc123...
        - **Vimeo:** https://vimeo.com/123456789
        - **YouTube:** https://youtu.be/abc123...
        - **Direct:** https://example.com/candidate-video.mp4
        
        *⚡ Pro tip: Videos with 1-3 minutes of speech provide the most accurate results*
        """)

if __name__ == "__main__":
    main()
