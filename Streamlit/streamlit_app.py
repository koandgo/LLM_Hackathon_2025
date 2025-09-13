# streamlit_app.py
# Streamlit wrapper to (1) transcribe audio with OpenAI (gpt-4o-transcribe only) and (2) optionally run the repo notebook.
# - Transcribes uploaded audio via OpenAI (gpt-4o-transcribe)
# - Saves transcript to OUTPUT_DIR/transcript.txt
# - Injects transcript path & text into the notebook as parameters
# - Clones/updates your repo and executes Audio_to_eln_wrapper.ipynb
#
# Deployment: Streamlit Cloud -> point at this file and add OPENAI_API_KEY to Secrets.

import os
import sys
import time
import tempfile
from pathlib import Path
import re
import streamlit as st

# Notebook execution utilities
import nbformat
from nbclient import NotebookClient
try:
    from nbclient.exceptions import CellExecutionError
except Exception:
    from nbclient import CellExecutionError

# Repo management utilities
from git import Repo, GitCommandError

# OpenAI client
from openai import OpenAI

# --------- App Config ---------
REPO_URL = "https://github.com/koandgo/LLM_Hackathon_2025.git"
REPO_BRANCH = "main"
NOTEBOOK_REL_PATH = "Audio_to_eln_wrapper.ipynb"  # located at repo root
CLONE_DIR = Path("./_repo_cache").resolve()
OUTPUT_DIR = Path("./_run_outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Speech → ELN (OpenAI Transcribe)", page_icon="🧪", layout="centered")

st.title("🧪 Speech → ELN")
st.caption("Transcribe audio via OpenAI **gpt-4o-transcribe**, then (optionally) run the repo notebook to build ELN artifacts.")
st.markdown(f"**Repo:** [{REPO_URL}]({REPO_URL})")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="If set, overrides the OPENAI_API_KEY env var (or Streamlit Secrets).")
    branch = st.text_input("Git branch", value=REPO_BRANCH)
    keep_repo = st.checkbox("Skip repo update if already cloned", value=False)
    st.subheader("Transcription")
    st.write("Model: **gpt-4o-transcribe**")
    prompt = st.text_area("Custom vocabulary / prompt (optional)", help="Useful for jargon, names, or style.")
    run_notebook_after = st.checkbox("Run the repo notebook after transcribing", value=True)
    st.divider()
    st.caption("Tip: If execution fails, check the **Execution Log** below for details.")

@st.cache_resource(show_spinner=True)
def clone_or_update_repo(url: str, branch: str, clone_dir: Path, keep: bool) -> Path:
    if clone_dir.exists() and any(clone_dir.iterdir()):
        repo = Repo(str(clone_dir))
        if not keep:
            repo.git.fetch("--all", "--prune")
            # ensure branch exists locally
            try:
                repo.git.checkout(branch)
            except GitCommandError:
                repo.git.checkout("-B", branch, f"origin/{branch}")
            repo.git.reset("--hard", f"origin/{branch}")
        return clone_dir
    else:
        clone_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(url, str(clone_dir), branch=branch, depth=1)
        return clone_dir


def inject_parameters(nb, params: dict):
    """Prepend code cells to define parameters and provide an ASR stub if NeMo isn't available and env vars for API keys."""
    import nbformat as _nbf
    import os as _os

    # 1) Parameters cell
    lines = ["# --- Parameters injected by Streamlit app ---"]
    for k, v in params.items():
        try:
            from pathlib import Path as _P
            if isinstance(v, (str, _P)):
                lines.append(f'{k} = r"""{str(v)}"""')
            else:
                lines.append(f"{k} = {repr(v)}")
        except Exception:
            lines.append(f"{k} = {repr(v)}")
    # Also ensure OPENAI_API_KEY is pulled from env if not set or has placeholder
    lines.append('try:
    OPENAI_API_KEY
    _has_key = bool(OPENAI_API_KEY and "YOUR_API" not in str(OPENAI_API_KEY))
except NameError:
    _has_key = False')
    lines.append('if not _has_key:
    OPENAI_API_KEY = _os.environ.get("OPENAI_API_KEY", "")')

    param_cell = _nbf.v4.new_code_cell("\n".join(lines), metadata={"tags": ["injected-parameters"]})

    # 2) ASR compatibility shim
    shim = r"""
# --- ASR stub injected by Streamlit app ---
try:
    asr_model  # if a real NeMo model exists, keep it
except NameError:
    class _Hypothesis:
        def __init__(self, text):
            self.text = text
            self.timestamp = {"word": [], "segment": [], "char": []}
    class _ASRStub:
        def transcribe(self, files, timestamps=True):
            # Return a list with one Hypothesis that uses TRANSCRIPT_TEXT if available
            txt = (globals().get('TRANSCRIPT_TEXT') or "")
            return [_Hypothesis(txt)]
    asr_model = _ASRStub()
"""
    shim_cell = _nbf.v4.new_code_cell(shim, metadata={"tags": ["injected-asr-stub"]})

    nb.cells.insert(0, shim_cell)
    nb.cells.insert(0, param_cell)
class cd:
    """Context manager to change current working dir safely."""
    def __init__(self, new_dir):
        self.new_dir = str(new_dir)
        self.old_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.new_dir)
    def __exit__(self, exc_type, exc, tb):
        os.chdir(self.old_dir)


def sanitize_notebook(nb):
    """
    Remove or neutralize cells that rely on NeMo/Colab or pip installs that break Streamlit Cloud.
    Heuristics: any cell containing these patterns will be replaced with a harmless 'pass' cell.
    """
    BAD_PATTERNS = [
        r"\bnemo\.collections\.asr\b",
        r"\bnemo_asr\b",
        r"parakeet-tdt",
        r"\bgoogle\.colab\b",
        r"\bfiles\.upload\(",
        r"^!\s*pip\s+install",
        r"^%\s*pip\s+install",
        r"^!\s*apt(-get)?\s+install",
        r"^%\s*bash",
    ]
    compiled = [re.compile(p, re.IGNORECASE|re.MULTILINE) for p in BAD_PATTERNS]

    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        # Skip if user already tagged it to skip
        tags = cell.get("metadata", {}).get("tags", [])
        if "skip_on_streamlit" in tags:
            nb.cells[i]["source"] = "pass  # skipped by Streamlit app (tag)"
            continue

        if any(c.search(src) for c in compiled):
            nb.cells[i]["source"] = "pass  # skipped by Streamlit app (NeMo/Colab/pip cell removed)"
    return nb


def execute_notebook(repo_dir: Path, notebook_rel_path: str, params: dict, timeout_s: int = 1800):
    nb_path = repo_dir / notebook_rel_path
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Load notebook & inject parameters
    nb = nbformat.read(nb_path, as_version=4)
    nb = sanitize_notebook(nb)
    inject_parameters(nb, params)

    executed_nb_path = OUTPUT_DIR / f"executed_{nb_path.name}"

    # Track file mtimes to detect new artifacts
    before = {}
    for p in repo_dir.rglob("*"):
        if p.is_file():
            try:
                before[p] = p.stat().st_mtime
            except Exception:
                pass

    start_time = time.time()

    # Execute in the repo root to respect relative paths inside the notebook
    with cd(repo_dir):
        client = NotebookClient(nb, timeout=timeout_s, kernel_name="python3", allow_errors=False)
        client.execute()

    # Save executed notebook
    nbformat.write(nb, executed_nb_path)

    # Collect new/changed artifacts
    created = []
    for p, mtime in before.items():
        try:
            new_mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if new_mtime >= start_time - 1:
            created.append(p)

    for p in repo_dir.rglob("*"):
        if p.is_file() and p not in before:
            try:
                if p.stat().st_mtime >= start_time - 1:
                    created.append(p)
            except Exception:
                pass

    # De-duplicate
    seen, uniq_created = set(), []
    for p in created:
        if p not in seen:
            seen.add(p)
            uniq_created.append(p)

    return executed_nb_path, uniq_created

def transcribe_with_openai(audio_path: Path, api_key: str, prompt: str = "") -> str:
    """Transcribe audio file with OpenAI Audio API (gpt-4o-transcribe) and return the text."""
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    kwargs = {"model": "gpt-4o-transcribe"}
    if prompt.strip():
        kwargs["prompt"] = prompt
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)
    return getattr(result, "text", None) or (hasattr(result, "data") and result.data.get("text")) or str(result)

# ------------- UI -------------
uploaded = st.file_uploader("Upload an audio file (mp3, m4a, wav, aac, flac, ogg)", type=["mp3", "m4a", "wav", "aac", "flac", "ogg"])
run_btn = st.button("▶️ Transcribe (and run notebook)")

log = st.empty()
status = st.empty()

if run_btn:
    if not uploaded:
        st.error("Please upload an audio file first.")
        st.stop()

    # Persist uploaded file
    tmp_audio = Path(tempfile.mkdtemp()) / uploaded.name
    with open(tmp_audio, "wb") as f:
        f.write(uploaded.getbuffer())

    # Make API key visible
    final_api_key = api_key or st.secrets.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if final_api_key:
        os.environ["OPENAI_API_KEY"] = final_api_key
    else:
        st.warning("No OPENAI_API_KEY provided. Set it in the sidebar or Streamlit Secrets to run transcription and ELN generation.")

    # 1) Transcribe with OpenAI
    status.info("Transcribing with OpenAI (gpt-4o-transcribe)…")
    try:
        transcript_text = transcribe_with_openai(tmp_audio, api_key=os.getenv("OPENAI_API_KEY"), prompt=prompt)
        status.success("Transcription complete.")
    except Exception as e:
        st.error("Transcription failed.")
        with st.expander("Transcription error details", expanded=True):
            st.exception(e)
        st.stop()

    # Save transcript for download & for the notebook
    transcript_path = OUTPUT_DIR / "transcript.txt"
    transcript_path.write_text(transcript_text or "", encoding="utf-8")
    st.subheader("📝 Transcript")
    st.download_button("Download transcript.txt", data=transcript_path.read_bytes(), file_name="transcript.txt")
    st.text_area("Preview", transcript_text, height=200)

    if not run_notebook_after:
        st.info("Notebook run skipped by user setting.")
        st.stop()

    # 2) Clone/update repo
    status.info("Cloning/updating repository…")
    repo_dir = clone_or_update_repo(REPO_URL, branch, CLONE_DIR, keep=keep_repo)

    # 3) Execute notebook with injected parameters
    params = {
        "AUDIO_FILE": tmp_audio,
        "ELN_OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "TRANSCRIPT_FILE": transcript_path,
        "TRANSCRIPT_TEXT": transcript_text,
        # Notebook wrapper config vars:
        "INPUT_TXT_PATH": transcript_path,                               # notebook expects a txt path
        "OUTPUT_ELN_PATH": (OUTPUT_DIR / "eln_export.zip"),              # target zip path
        "MODEL_NAME": "gpt-4o-mini",                                     # chat model for HTML generation
        "TIMEZONE": "America/New_York",
        "RECORD_TITLE": "ELN from transcript",
        "RUN_CONTEXT": "streamlit_openai_transcribe",
    }

    try:
        status.info("Executing notebook…")
        executed_nb_path, artifacts = execute_notebook(repo_dir, NOTEBOOK_REL_PATH, params)
        status.success("Notebook execution finished.")
        st.success("✅ Done! See artifacts below.")
        with st.expander("Execution log / details", expanded=False):
            st.write(f"Executed notebook saved to: `{executed_nb_path}`")
            st.write(f"Artifacts detected: {len(artifacts)} files")

        st.subheader("📦 Download artifacts")
        if artifacts:
            for p in artifacts:
                try:
                    data = p.read_bytes()
                    st.download_button(
                        label=f"Download {p.relative_to(repo_dir)}",
                        data=data,
                        file_name=p.name
                    )
                except Exception as e:
                    st.caption(f"Skipping {p} (read error: {e})")
        else:
            st.caption("No new files were detected in the repo after execution. The notebook may print results instead of writing files. Check the executed notebook below.")

        st.subheader("🗒️ Executed notebook")
        st.download_button("Download executed notebook (.ipynb)", data=Path(executed_nb_path).read_bytes(), file_name=Path(executed_nb_path).name)

    except CellExecutionError as e:
        status.error("Notebook raised an exception during execution.")
        with st.expander("Execution error (traceback)", expanded=True):
            st.exception(e)
    except Exception as e:
        status.error("Unexpected error.")
        with st.expander("Error details", expanded=True):
            st.exception(e)
