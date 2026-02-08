import streamlit as st
import requests
import base64
import os
import io
from dotenv import load_dotenv

# Load environment variables from .env file (parent directory)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configuration
API_URL = os.getenv("API_URL", "http://0.0.0.0:8000")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

st.set_page_config(
    page_title="Problem Solver AI",
    page_icon="💡",
    layout="wide"
)

# Custom CSS for cards
st.markdown("""
<style>
    .category-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .solvable { background-color: #d4edda; color: #155724; }
    .unsolvable { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

st.title("💡 AI Opportunity Compass")
st.caption("Transforming real-world pain points into actionable tech opportunities.")

# --- HELPER FUNCTION FOR SPEECH-TO-TEXT ---
def speech_to_text(audio_input) -> str:
    """
    Converts speech audio to text using ElevenLabs API.
    """
    if not ELEVENLABS_API_KEY:
        st.error("ElevenLabs API key not configured")
        return ""
    
    try:
        # Read the audio bytes from UploadedFile
        if hasattr(audio_input, 'read'):
            audio_bytes = audio_input.read()
        else:
            audio_bytes = audio_input
        
        # Create a BytesIO object for the request
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        # Call ElevenLabs speech-to-text API
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "model_id": "scribe_v2"
        }
        files = {
            "file": audio_file
        }
        
        response = requests.post(url, headers=headers, data=data, files=files)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("text", "")
        else:
            st.error(f"ElevenLabs error: {response.text}")
            return ""
            
    except Exception as e:
        st.error(f"Speech-to-text conversion failed: {e}")
        return ""

# Layout
tab1, tab2 = st.tabs(["🚀 Submit Problem", "🔍 Explore Opportunities"])

# --- TAB 1: SUBMIT ---
with tab1:
    st.header("What's broken?")
    st.write("Submit a pain point you face. Our AI will analyze if it's a solvable software problem.")
    
    with st.form("submission_form"):
        description = st.text_area(
            "Describe the problem in detail:",
            height=150,
            placeholder="e.g., Farmers in my village struggle to identify crop diseases early..."
        )
        
        # Speech-to-text option
        st.write("**Or** 🎤 Record your problem:")
        audio_bytes = st.audio_input("Click the microphone to record")
        
        if audio_bytes and not description:
            st.info("Processing audio...")
            transcribed = speech_to_text(audio_bytes)
            if transcribed:
                description = st.text_area(
                    "Transcribed text (you can edit):",
                    value=transcribed,
                    height=150
                )
        
        submitted = st.form_submit_button("Analyze & Submit")
        
        if submitted and description:
            # Custom spinner with more personality
            with st.status("🎅 Tech Santa is analyzing...", expanded=True) as status:
                try:
                    status.write("🧠 Understanding the problem context...")
                    # Simulating thinking time if backend is instantaneous (optional)
                    
                    status.write("🔍 Search for existing solutions...")
                    response = requests.post(f"{API_URL}/submit", json={"description": description})
                    
                    if response.status_code == 200:
                        status.write("✅ Analysis complete!")
                        status.update(label="Analysis Done", state="complete", expanded=False)
                        
                        data = response.json()
                        
                        # Handle duplicate/similar wishes gracefully
                        if data.get("status") == "upvoted_existing":
                            st.info("💡 **Great minds think alike!**")
                            st.write(data.get("message"))
                            st.metric("Similarity Score", f"{int(data.get('similarity_score', 0)*100)}%")
                            
                            # Display Resources for duplicate
                            resources = data.get("resources", [])
                            if resources:
                                st.divider()
                                st.subheader("📚 Relevant Resources")
                                if isinstance(resources, list):
                                    for resource in resources:
                                        st.write(f"• {resource}")
                                else:
                                    st.write(resources)
                            
                            st.caption("We've added your vote to the existing wish.")
                            status.update(label="Duplicate Found", state="complete", expanded=False)
                        else:
                            analysis = data.get("analysis", {})
                            
                            # Result Display
                            category = analysis.get('category', 'Unknown')
                            status_color = {
                                "Solvable": "solvable",
                                "Existing Solution": "unsolvable", # Reusing red style for now
                                "Unsolvable": "unsolvable", 
                                "Spam": "unsolvable"
                            }.get(category, "unsolvable")
                            
                            st.markdown(f"### Result: <span class='category-badge {status_color}'>{category}</span>", unsafe_allow_html=True)
                            
                            if category == "Solvable":
                                st.success(f"**Opportunity Found!**")
                                st.info(f"**Guidance:** {analysis.get('guidance')}")
                            elif category == "Existing Solution":
                                st.warning(f"**Solution Exists**")
                                st.info(f"**Check this out:** {analysis.get('guidance')}")
                            elif category == "Unsolvable":
                                st.error(f"**Not a Software Problem**")
                                st.write(f"**Reason:** {analysis.get('reasoning')}")
                            else:
                                st.info(f"**Status:** {category}")
                                if analysis.get('reasoning'):
                                    st.write(f"**AI Logic:** {analysis.get('reasoning')}")
                            
                            # Display Resources
                            resources = analysis.get("resources", [])
                            if resources:
                                st.divider()
                                st.subheader("📚 Relevant Resources")
                                if isinstance(resources, list):
                                    for resource in resources:
                                        st.write(f"• {resource}")
                                else:
                                    st.write(resources)
                        
                        # Show raw analysis in expander for debugging if needed
                            
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# --- TAB 2: EXPLORE ---
with tab2:
    st.header("Open Opportunities")
    st.write("Browse validated problems ready for a solution.")
    
    # Fetch Data
    try:
        response = requests.get(f"{API_URL}/opportunities")
        if response.status_code == 200:
            opportunities = response.json()
            
            # --- SIDEBAR CONTROLS ---
            with st.sidebar:
                st.header("🔍 Filter & Sort")
                
                # Sort Controls
                sort_option = st.selectbox(
                    "Sort By",
                    ["Newest", "Highest Impact"],
                    index=0
                )

                st.divider()
                st.subheader("Filters")

                # Filter: Industry
                all_industries = sorted(list(set(op['analysis'].get('industry', 'General') for op in opportunities)))
                selected_industry = st.multiselect("Industry", all_industries)

                # Filter: Difficulty
                all_difficulties = ["Low", "Medium", "High"]
                selected_diff = st.multiselect("Complexity", all_difficulties)

            # --- LOGIC ---
            
            # 1. Apply Filters
            if selected_industry:
                opportunities = [op for op in opportunities if op['analysis'].get('industry') in selected_industry]
            
            if selected_diff:
                opportunities = [op for op in opportunities if op.get('metrics', {}).get('difficulty', 'Medium') in selected_diff]

            # 2. Apply Sorting
            if sort_option == "Highest Impact":
                opportunities.sort(key=lambda x: x.get('metrics', {}).get('impact_score', 0), reverse=True)
            else: # Newest
                # Assuming the list comes sorted by date from backend, or strict parse
                pass 

            st.markdown(f"**Showing {len(opportunities)} opportunities**")
            st.markdown("---")
            
            # Grid Layout
            if not opportunities:
                st.info("No opportunities found matching your criteria.")
            
            for op in opportunities:
                analysis = op.get("analysis", {})
                metrics = op.get("metrics", {"difficulty": "Medium", "impact_score": 0})
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Header with Metrics badges
                        st.subheader(analysis.get("summary"))
                        
                        # Metrics Row
                        m1, m2 = st.columns(2)
                        m1.caption(f"⚡ Impact: {metrics.get('impact_score')}/100")
                        m2.caption(f"🏗️ Diff: {metrics.get('difficulty')}")


                        st.markdown(f"**Industry:** `{analysis.get('industry', 'General')}`")
                        st.markdown(f"> *\"{op.get('original_text')}\"*")
                        
                        # Display Resources prominently
                        resources = analysis.get("resources", [])
                        if resources:
                            st.markdown("**📚 Relevant Resources:**")
                            if isinstance(resources, list):
                                for resource in resources:
                                    st.write(f"• {resource}")
                            else:
                                st.write(resources)
                        
                        with st.expander("🛠 Technical Guidance"):
                            st.write(analysis.get("guidance"))
                            st.caption(f"Reasoning: {analysis.get('reasoning')}")
                            
                    with col2:
                        # Audio Player
                        audio_b64 = op.get("audio_b64")
                        if audio_b64:
                            st.markdown("**🎧 Audio Summary**")
                            audio_bytes = base64.b64decode(audio_b64)
                            st.audio(audio_bytes, format="audio/mp3")
                        else:
                            st.caption("No audio available")
                            
                    st.divider()
                    
        else:
            st.error("Failed to fetch opportunities.")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

