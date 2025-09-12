# Create a Streamlit app that takes an audio file, transcribes it with OpenAI Whisper,
# converts the transcript to an ELN-style HTML body using a strict template,
# and lets the user download a .eln (zip) output.
#
# Files created:
# - /mnt/data/streamlit_app.py
# - /mnt/data/requirements.txt

from pathlib import Path

app_code = r"""
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---- Optional: use python-dotenv if available (won't error if not installed) ----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---- OpenAI SDK (v1) ----
try:
    from openai import OpenAI
except Exception as e:
    st.error("OpenAI SDK not installed. Please `pip install openai` or use the provided requirements.txt.")
    raise

# =========================
# Configuration & Prompts
# =========================

SYSTEM_PROMPT = (
    "You are an expert Lab Notebook Compiler that converts experiment speech transcripts into a single HTML body "
    "matching an eLabFTW-export style. Do not invent data. Normalize number words to numerals. "
    "Standardize dates to YYYY-MM-DD and times as h:mm am/pm with timezone if mentioned. "
    "If a value is missing or unclear, leave the table cell blank. "
    "Output ONLY raw HTML (no Markdown fences, no extra comments), and follow the exact formatting rules provided."
)

HTML_INSTRUCTIONS = (
    "Build the following sections in order. For each, render a section header and a table (unless directed otherwise). "
    "Create a section only if the transcript contains at least one row/value for it.\n\n"
    "Formatting rules:\n"
    "• Section headers must be: <p><span style=\"font-size:18pt;\">{Title}</span></p>\n"
    "• Tables must be: <table style=\"min-width:25%;width:60%;border-width:1px;margin-left:0px;margin-right:auto;\" border=\"1\">\n"
    "• The first row of every table is a centered header row using: <td style=\"text-align:center;\">.\n"
    "• Each subsequent row is one record extracted from the transcript.\n"
    "• Use <p> for single-line notes and <ul><li><p>…</p></li>…</ul> for bullet lists when a section calls for free-text notes.\n\n"
    "Sections:\n"
    "1) Experiment Overview\n"
    "   Table columns: [Title, Date, Start Time, End Time, Timezone, Project ID, Operator]\n\n"
    "2) Materials & Reagents\n"
    "   Table columns: [Name, Lot/ID, Vendor, Purity/Grade, Amount, Units, Notes]\n\n"
    "3) Equipment & Settings\n"
    "   Table columns: [Instrument, Model, Setting/Program, Value, Units, Notes]\n\n"
    "4) Procedure (Chronological Steps)\n"
    "   Table columns: [Timestamp, Action, Quantity, Units, Vessel/Location, Notes]\n\n"
    "5) Solutions / Stocks (Composition)\n"
    "   Table columns: [Solution ID, Component, Amount, Units, Final Volume, Solvent, Temperature, Notes]\n\n"
    "6) Observations & Conditions\n"
    "   Use a bullet list (<ul><li><p>…</p></li></ul>) of short notes if present. No table for this section.\n\n"
    "7) Measurements — General\n"
    "   Table columns: [Type, Value, Units, Method/Instrument, Timestamp, Notes]\n\n"
    "8) Measurements — Volume Results\n"
    "   Table columns: [Sample/Vial ID, Volume, Units, Timestamp, Notes]\n\n"
    "9) Measurements — Mass Results\n"
    "   Table columns: [Sample/Vial ID, Mass, Units, Timestamp, Notes]\n\n"
    "10) Yields & Calculations\n"
    "   Table columns: [Sample/Vial ID, Theoretical Yield, Actual Yield, Percent Yield, Units, Notes]\n\n"
    "11) Attachments (filenames only if mentioned)\n"
    "   Table columns: [Filename, Type, Notes]\n\n"
    "12) Cleanup & Storage\n"
    "   Table columns: [Item, Location, Condition, Notes]\n\n"
    "13) Next Steps / TODO\n"
    "   Use a bullet list (<ul><li><p>…</p></li></ul>) if present. No table for this section.\n\n"
    "Validation:\n"
    "• Do not invent values or sections. Leave cells blank when unknown.\n"
    "• Keep original order of events where possible.\n"
)

USER_PREFIX = (
    "Parse the transcript below into the specified HTML sections. Do not add sections beyond the list. "
    "Leave cells blank when the transcript doesn't supply a value. Assume timezone '{tz}'.\n\n"
)

# =========================
# Helpers
# =========================

def openai_transcribe(audio_bytes: bytes, filename: str, openai_api_key: str) -> str:
    \"\"\"Transcribe audio bytes using OpenAI Whisper.\"\"\"
    client = OpenAI(api_key=openai_api_key)
    # Use a NamedTemporaryFile-like wrapper because OpenAI SDK expects a file-like object with .name
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True, suffix=Path(filename).suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=\"whisper-1\",
                file=f
            )
    # SDK returns object with .text in v1
    return transcript.text if hasattr(transcript, \"text\") else str(transcript)

def openai_html_from_transcript(transcript_text: str, tz: str, openai_api_key: str, model: str = \"gpt-4o-mini\") -> str:
    client = OpenAI(api_key=openai_api_key)
    user_msg = USER_PREFIX.format(tz=tz) + \"\\nTRANSCRIPT:\\n<<<\\n\" + transcript_text + \"\\n>>>\\n\"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {\"role\": \"system\", \"content\": SYSTEM_PROMPT},
            {\"role\": \"user\", \"content\": HTML_INSTRUCTIONS},
            {\"role\": \"user\", \"content\": user_msg},
        ],
    )
    html = resp.choices[0].message.content.strip()
    return html

def build_eln_zip_bytes(html_body: str, title: str, transcript_text: str | None, source_audio_name: str | None) -> bytes:
    \"\"\"Build a simple .eln (zip) in-memory.\n
    Structure (root of zip):\n
      - metadata.json\n
      - body.html\n
      - transcript.txt (optional)\n
    \"\"\"
    metadata = {
        \"title\": title,
        \"created\": datetime.now().astimezone().isoformat(),
        \"generator\": \"Streamlit Audio→ELN\"
    }
    if source_audio_name:
        metadata[\"source_audio_filename\"] = source_audio_name

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode=\"w\", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(\"metadata.json\", __import__(\"json\").dumps(metadata, ensure_ascii=False, indent=2))
        zf.writestr(\"body.html\", html_body)
        if transcript_text:
            zf.writestr(\"transcript.txt\", transcript_text)
    buf.seek(0)
    return buf.read()

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title=\"Audio → ELN\", page_icon=\"🧪\", layout=\"centered\")

st.title(\"🧪 Audio → ELN (HTML) Builder\")
st.write(\"Upload an audio file to transcribe, then convert the transcript into an ELN-style HTML body and download a .eln file.\")

with st.sidebar:
    st.header(\"Settings\")
    openai_api_key = st.text_input(\"OpenAI API Key\", type=\"password\", value=os.getenv(\"OPENAI_API_KEY\", \"\"))
    tz = st.text_input(\"Assumed Timezone (IANA)\", value=\"America/New_York\")
    llm_model = st.text_input(\"LLM for HTML generation\", value=\"gpt-4o-mini\")
    title_default = f\"Experiment {datetime.now().strftime('%Y-%m-%d %H:%M')}\"
    record_title = st.text_input(\"Record Title\", value=title_default)

tab1, tab2 = st.tabs([\"Transcribe from Audio\", \"Paste Transcript\"])

transcript_text: str | None = None
source_audio_name: str | None = None

with tab1:
    uploaded = st.file_uploader(\"Choose an audio file\", type=[\"wav\", \"mp3\", \"m4a\", \"ogg\", \"flac\", \"webm\"])
    if uploaded is not None:
        source_audio_name = uploaded.name
        st.audio(uploaded, format=None)
        if st.button(\"Transcribe Audio\", type=\"primary\", use_container_width=True):
            if not openai_api_key:
                st.error(\"Please provide an OpenAI API key.\")
            else:
                with st.spinner(\"Transcribing with Whisper...\"):
                    audio_bytes = uploaded.read()
                    try:
                        transcript_text = openai_transcribe(audio_bytes, uploaded.name, openai_api_key)
                        st.success(\"Transcription complete.\")
                        st.text_area(\"Transcript\", transcript_text, height=240)
                    except Exception as e:
                        st.error(f\"Transcription failed: {e}\")

with tab2:
    manual_text = st.text_area(\"Paste an existing transcript (bypasses audio transcription)\", height=240)
    if manual_text.strip():
        transcript_text = manual_text.strip()

st.divider()

if st.button(\"Generate ELN HTML\", type=\"secondary\", use_container_width=True):
    if not openai_api_key:
        st.error(\"Please provide an OpenAI API key.\")
    elif not transcript_text:
        st.error(\"Please provide a transcript (via audio or paste).\");
    else:
        with st.spinner(\"Generating ELN HTML with the LLM...\"):
            try:
                html_body = openai_html_from_transcript(transcript_text, tz, openai_api_key, model=llm_model)
            except Exception as e:
                st.error(f\"HTML generation failed: {e}\")
                st.stop()
        st.success(\"ELN HTML generated.\")
        st.write(\"Preview (first 1000 chars):\")
        st.code(html_body[:1000] + (\"...\" if len(html_body) > 1000 else \"\"), language=\"html\");

        # Build .eln (zip) and offer download
        eln_bytes = build_eln_zip_bytes(html_body, record_title, transcript_text, source_audio_name)
        eln_filename = f\"{record_title.replace('/', '-').replace('\\\\', '-')}.eln.zip\"
        st.download_button(
            label=\"⬇️ Download ELN (.eln.zip)\",
            data=eln_bytes,
            file_name=eln_filename,
            mime=\"application/zip\",
            use_container_width=True,
        )

st.caption(\"Tip: If eLabFTW import expects a different structure, you can unzip and adapt `metadata.json` or `body.html` as needed.\")
"""

reqs = """streamlit>=1.36
openai>=1.35
python-dotenv>=1.0
"""

# Write files
Path("/mnt/data/streamlit_app.py").write_text(app_code, encoding="utf-8")
Path("/mnt/data/requirements.txt").write_text(reqs, encoding="utf-8")

print("Created /mnt/data/streamlit_app.py and /mnt/data/requirements.txt")
