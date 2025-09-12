# streamlit_app.py
# Streamlit wrapper to run the Audio_to_eln_wrapper.ipynb notebook from the GitHub repo.
# - Clones/updates the repo
# - Lets you upload an audio file
# - Injects variables into the notebook
# - Executes it headlessly
# - Surfaces any files the notebook created so you can download them

import os
import sys
import time
import tempfile
from pathlib import Path
import streamlit as st

# Notebook execution utilities
import nbformat
from nbclient import NotebookClient
try:
    from nbclient.exceptions import CellExecutionError
except Exception:
    # Fallback for older nbclient versions
    from nbclient import CellExecutionError

# Repo management utilities
from git import Repo, GitCommandError

from jupyter_client.kernelspec import KernelSpecManager
import subprocess
import sys

def ensure_kernel_available(preferred_name: str = "python3") -> str:
    """Ensure a Jupyter kernelspec exists for the current Python.
    Returns the usable kernel name, or raises on failure."""
    ksm = KernelSpecManager()
    try:
        ksm.get_kernel_spec(preferred_name)
        return preferred_name
    except Exception:
        pass
    # Try to install the preferred name first
    try:
        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user",
             "--name", preferred_name, "--display-name", "Python (Streamlit)"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return preferred_name
    except Exception:
        # Fallback: install a versioned name
        versioned = f"python{sys.version_info.major}.{sys.version_info.minor}"
        try:
            subprocess.run(
                [sys.executable, "-m", "ipykernel", "install", "--user",
                 "--name", versioned, "--display-name", f"Python {sys.version_info.major}.{sys.version_info.minor}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return versioned
        except Exception as e:
            raise RuntimeError(f"Could not ensure a Jupyter kernel is installed: {e}")

# --------- App Config ---------
REPO_URL = "https://github.com/koandgo/LLM_Hackathon_2025.git"
REPO_BRANCH = "main"
NOTEBOOK_REL_PATH = "Audio_to_eln_wrapper.ipynb"  # located at repo root
CLONE_DIR = Path("./_repo_cache").resolve()
OUTPUT_DIR = Path("./_run_outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Speech → ELN Notebook Runner", page_icon="🧪", layout="centered")

st.title("🧪 Speech → ELN: Notebook Runner")
st.caption("Runs the repository’s **Audio_to_eln_wrapper.ipynb** headlessly and lets you download generated artifacts.")
st.markdown(f"**Repo:** [{REPO_URL}]({REPO_URL})")

with st.sidebar:
    st.header("Settings")
    force_openai_asr = st.checkbox("Force OpenAI ASR fallback (disable NeMo/Colab cells)", value=True)
    disable_nemo = st.checkbox("Disable NeMo cells even if present", value=True)
    api_key = st.text_input("OpenAI API Key (optional)", type="password", help="If the notebook uses OpenAI, put your key here. It will be exposed as the env var OPENAI_API_KEY during execution.")
    branch = st.text_input("Git branch", value=REPO_BRANCH)
    keep_repo = st.checkbox("Skip repo update if already cloned", value=False)
    st.divider()
    st.caption("Tip: If execution fails, check the **Execution Log** below for details.")

@st.cache_resource(show_spinner=True)
def clone_or_update_repo(url: str, branch: str, clone_dir: Path) -> Path:
    if clone_dir.exists() and any(clone_dir.iterdir()):
        repo = Repo(str(clone_dir))
        if not keep_repo:
            repo.git.fetch("--all", "--prune")
            # make sure branch exists locally
            try:
                repo.git.checkout(branch)
            except GitCommandError:
                repo.git.checkout("-B", branch, f"origin/{branch}")
            # hard reset to remote
            repo.git.reset("--hard", f"origin/{branch}")
        return clone_dir
    else:
        clone_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(url, str(clone_dir), branch=branch, depth=1)
        return clone_dir

def inject_parameters(nb, params: dict):
    """Prepend a code cell that defines variables expected by the notebook.
       This works even if the notebook is not papermill-parameterized."""
    lines = ["# --- Parameters injected by Streamlit app ---"]
    for k, v in params.items():
        if isinstance(v, (str, Path)):
            lines.append(f'{k} = r"""{str(v)}"""')
        else:
            lines.append(f"{k} = {repr(v)}")
    cell = nbformat.v4.new_code_cell("\n".join(lines), metadata={"tags": ["injected-parameters"]})
    nb.cells.insert(0, cell)


def rewrite_asr_cells_for_streamlit(nb, audio_var_name="AUDIO_FILE", require_openai=True, force=True, disable_nemo=True):
    """Detect cells that rely on Colab upload / NVIDIA NeMo ASR and replace them with
    an OpenAI transcription fallback that uses the uploaded AUDIO_FILE path."""
    asr_hit = False
    replaced_count = 0
    for i, cell in enumerate(list(nb.cells)):
        src = cell.get("source", "") or ""
        sl = src.lower()
        if (
            ("nemo" in sl) or
            ("google.colab" in sl) or
            ("asrmodel.from_pretrained" in sl) or
            ("files.upload(" in sl) or
            ("parakeet-tdt" in sl)
        ):
            asr_hit = True
            if force or disable_nemo:
            asr_hit = True
            replaced_count += 1
            nb.cells[i]["source"] = f"""
# --- Streamlit ASR fallback (auto-injected) ---
import os, json, sys, io
from pathlib import Path

_audio_path = str({audio_var_name})
print("ASR fallback using OpenAI; audio:", _audio_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set, but ASR fallback requires it. Set it in the Streamlit sidebar or as an environment secret.")

try:
    from openai import OpenAI
except Exception:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, "-m", "pip", "install", "--upgrade", "openai"], check=True)
    from openai import OpenAI

client = OpenAI()

# Prefer newer transcribe models if available in your account; otherwise try whisper-1
MODEL_CANDIDATES = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]

last_err = None
transcript_text = ""
word_timestamps = []
segment_timestamps = []
char_timestamps = []

for _m in MODEL_CANDIDATES:
    try:
        with open(_audio_path, "rb") as f:
            # verbose_json provides timestamps when supported
            tr = client.audio.transcriptions.create(
                model=_m, 
                file=f,
                response_format="verbose_json"
            )
        # Normalize common fields
        transcript_text = getattr(tr, "text", "") or (tr.get("text") if isinstance(tr, dict) else "")
        # Map segments/words if provided
        try:
            segs = getattr(tr, "segments", None) or (tr.get("segments") if isinstance(tr, dict) else None) or []
            segment_timestamps = [{"start": s.get("start"), "end": s.get("end"), "segment": s.get("text")} for s in segs]
        except Exception:
            pass
        try:
            words = getattr(tr, "words", None) or (tr.get("words") if isinstance(tr, dict) else None) or []
            word_timestamps = [{"start": w.get("start"), "end": w.get("end"), "word": w.get("word")} for w in words]
        except Exception:
            pass
        break
    except Exception as _e:
        last_err = _e
        continue

if not transcript_text and last_err:
    raise RuntimeError(f"OpenAI transcription failed across models: {last_err}")

# Compatibility prints (similar to the original example)
for stamp in segment_timestamps:
    try:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
    except Exception:
        pass
"""

    # If forced, ensure a top-level fallback cell exists (independent of detection)
    if force and (not any('ASR fallback (auto-injected)' in (c.get('source') or '') for c in nb.cells)):
        fb = nbformat.v4.new_code_cell(nbformat.v4.new_code_cell(f'''\n# --- Streamlit ASR fallback (auto-inserted at top) ---\nAUDIO_FILE = globals().get('AUDIO_FILE', '')\n''').source)
        nb.cells.insert(0, fb)
    return asr_hit or (replaced_count > 0)


class cd:
    """Context manager to change current working dir safely."""
    def __init__(self, new_dir):
        self.new_dir = str(new_dir)
        self.old_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.new_dir)
    def __exit__(self, exc_type, exc, tb):
        os.chdir(self.old_dir)

def execute_notebook(repo_dir: Path, notebook_rel_path: str, params: dict, timeout_s: int = 1800):
    nb_path = repo_dir / notebook_rel_path
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Load notebook & inject parameters
    nb = nbformat.read(nb_path, as_version=4)
    inject_parameters(nb, params)
    # Rewrite ASR/Colab cells to use OpenAI fallback if present
    try:
        hit = rewrite_asr_cells_for_streamlit(nb, audio_var_name='AUDIO_FILE', require_openai=True,
                                             force=params.get('FORCE_OPENAI_ASR', True),
                                             disable_nemo=params.get('DISABLE_NEMO', True))
        if hit and not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError('Notebook expects ASR (NeMo/Colab). Fallback enabled, but OPENAI_API_KEY is missing. Set it in the sidebar.')
    except Exception as _patch_e:
        # Non-fatal: continue; the notebook may still run if it does not need these cells
        pass

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
        kernel_name = ensure_kernel_available("python3")
        # Update notebook metadata kernelspec to match the available kernel
        nb.metadata.setdefault("kernelspec", {})
        nb.metadata.kernelspec["name"] = kernel_name
        nb.metadata.kernelspec["display_name"] = f"Python ({kernel_name})"
        client = NotebookClient(nb, timeout=timeout_s, kernel_name=kernel_name, allow_errors=False)
        client.execute()

    # Save executed notebook
    nbformat.write(nb, executed_nb_path)

    # Collect new/changed artifacts (created inside repo tree)
    created = []
    for p, mtime in before.items():
        try:
            new_mtime = p.stat().st_mtime
        except FileNotFoundError:
            # File was deleted; ignore
            continue
        # If file modified after we started, treat as new/updated artifact
        if new_mtime >= start_time - 1:
            created.append(p)

    # Also include any brand new files that weren't in "before"
    for p in repo_dir.rglob("*"):
        if p.is_file() and p not in before:
            try:
                if p.stat().st_mtime >= start_time - 1:
                    created.append(p)
            except Exception:
                pass

    # De-duplicate while preserving order
    seen = set()
    uniq_created = []
    for p in created:
        if p not in seen:
            seen.add(p)
            uniq_created.append(p)

    return executed_nb_path, uniq_created

# ------------- UI -------------
uploaded = st.file_uploader("Upload an audio file (mp3, m4a, wav)", type=["mp3", "m4a", "wav", "aac", "flac", "ogg"])
run_btn = st.button("▶️ Run notebook")

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

    status.info("Cloning/updating repository…")
    repo_dir = clone_or_update_repo(REPO_URL, branch, CLONE_DIR)

    # Make API key visible if provided (some notebooks expect this)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Common parameter names many wrappers use
    params = {
        "FORCE_OPENAI_ASR": force_openai_asr,
        "DISABLE_NEMO": disable_nemo,
        "AUDIO_FILE": tmp_audio,
        "ELN_OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "RUN_CONTEXT": "streamlit",  # so the notebook can branch on environment if it wants
    }

    try:
        status.info("Executing notebook… this can take a few minutes on first run.")
        executed_nb_path, artifacts = execute_notebook(repo_dir, NOTEBOOK_REL_PATH, params)
        status.success("Notebook execution finished.")
        st.success("✅ Done! See artifacts below.")

        with st.expander("Execution log / details", expanded=False):
            st.write(f"ASR rewrite forced: {params.get('FORCE_OPENAI_ASR', True)}, NeMo disabled: {params.get('DISABLE_NEMO', True)}")
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
