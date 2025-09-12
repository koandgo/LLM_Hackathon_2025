
# streamlit_app.py
# Clean Streamlit wrapper to run Audio_to_eln_wrapper.ipynb from this repo or by cloning.
# - Defaults to "local mode" (executes the notebook from the current repo tree)
# - Can optionally clone/update a remote repo
# - Upload an audio file; injects parameters; rewrites NeMo/Colab ASR cells to OpenAI fallback
# - Executes headlessly and exposes artifacts + executed notebook for download

import os
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
import nbformat
from nbclient import NotebookClient
try:
    from nbclient.exceptions import CellExecutionError
except Exception:
    # Older nbclient fallback
    from nbclient import CellExecutionError

# Repo management (only used when cloning)
try:
    from git import Repo, GitCommandError
except Exception:
    Repo = None
    GitCommandError = Exception

# Optional: kernelspec management
from jupyter_client.kernelspec import KernelSpecManager
import subprocess

# --------- App Config ---------
DEFAULT_REPO_URL = "https://github.com/koandgo/LLM_Hackathon_2025.git"
DEFAULT_BRANCH = "main"
NOTEBOOK_REL_PATH = "Audio_to_eln_wrapper.ipynb"  # expected at repo root
CACHE_DIR = Path("./_repo_cache").resolve()
OUTPUT_DIR = Path("./_run_outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Speech → ELN Notebook Runner", page_icon="🧪", layout="centered")
st.title("🧪 Speech → ELN: Notebook Runner")
st.caption("Runs **Audio_to_eln_wrapper.ipynb** headlessly and lets you download generated artifacts.")

# ------------- Sidebar -------------
with st.sidebar:
    st.header("Settings")
    exec_mode = st.radio("Execution mode", options=["Local (use this repo)", "Clone (fetch remote)"], index=0)
    repo_url = st.text_input("Repo URL (when cloning)", value=DEFAULT_REPO_URL)
    branch = st.text_input("Branch", value=DEFAULT_BRANCH)
    force_openai_asr = st.checkbox("Force OpenAI ASR fallback (disable NeMo/Colab cells)", value=True)
    disable_nemo = st.checkbox("Disable NeMo cells even if present", value=True)
    api_key = st.text_input("OpenAI API Key (optional)", type="password", help="If the notebook uses OpenAI, set this. Needed for the ASR fallback.")
    st.divider()
    keep_repo = st.checkbox("Skip repo update if already cloned (Clone mode only)", value=False)

st.markdown(f"**Notebook:** `{NOTEBOOK_REL_PATH}`")
st.markdown(f"**Default repo:** [{DEFAULT_REPO_URL}]({DEFAULT_REPO_URL})")

# --------- Helpers ---------
def ensure_kernel_available(preferred_name: str = "python3") -> str:
    """Ensure a Jupyter kernelspec exists for the current Python. Return kernel name."""
    ksm = KernelSpecManager()
    try:
        ksm.get_kernel_spec(preferred_name)
        return preferred_name
    except Exception:
        pass
    # install preferred
    try:
        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user",
             "--name", preferred_name, "--display-name", "Python (Streamlit)"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return preferred_name
    except Exception:
        # fallback to versioned
        versioned = f"python{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user",
             "--name", versioned, "--display-name", f"Python {sys.version_info.major}.{sys.version_info.minor}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return versioned

def clone_or_update_repo(url: str, branch: str, clone_dir: Path, keep: bool) -> Path:
    if Repo is None:
        raise RuntimeError("gitpython is not installed; cannot clone. Switch to Local mode or add gitpython to requirements.")
    if clone_dir.exists() and any(clone_dir.iterdir()):
        repo = Repo(str(clone_dir))
        if not keep:
            repo.git.fetch("--all", "--prune")
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
    """Prepend a code cell defining variables for the notebook."""
    lines = ["# --- Parameters injected by Streamlit app ---"]
    for k, v in params.items():
        if isinstance(v, (str, Path)):
            lines.append(f'{k} = r"""{str(v)}"""')
        else:
            lines.append(f"{k} = {repr(v)}")
    cell = nbformat.v4.new_code_cell("\n".join(lines), metadata={"tags": ["injected-parameters"]})
    nb.cells.insert(0, cell)

def rewrite_asr_cells_for_streamlit(nb, audio_var_name="AUDIO_FILE",
                                    require_openai=True, force=True, disable_nemo=True) -> bool:
    """Replace NeMo/Colab ASR cells with an OpenAI transcription fallback."""
    asr_hit = False
    replaced_count = 0
    for i, cell in enumerate(list(nb.cells)):
        src = (cell.get("source") or "")
        sl = src.lower()
        is_asr_cell = (
            ("nemo" in sl) or
            ("google.colab" in sl) or
            ("asrmodel.from_pretrained" in sl) or
            ("files.upload(" in sl) or
            ("parakeet-tdt" in sl)
        )
        if is_asr_cell:
            asr_hit = True
            if force or disable_nemo:
                nb.cells[i]["source"] = '''# --- Streamlit ASR fallback (auto-injected) ---
import os, sys, io, json
from pathlib import Path

_audio_path = str({audio_var_name})
print("ASR fallback using OpenAI; audio:", _audio_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set, but ASR fallback requires it. Set it in the Streamlit sidebar or as an environment secret.")

try:
    from openai import OpenAI
except Exception:
    import subprocess as _sp, sys as _sys
    _sp.run([_sys.executable, "-m", "pip", "install", "--upgrade", "openai"], check=True)
    from openai import OpenAI

client = OpenAI()

MODEL_CANDIDATES = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]

last_err = None
transcript_text = ""
word_timestamps = []
segment_timestamps = []
char_timestamps = []

for _m in MODEL_CANDIDATES:
    try:
        with open(_audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=_m,
                file=f,
                response_format="verbose_json"
            )
        transcript_text = getattr(tr, "text", "") or (tr.get("text") if isinstance(tr, dict) else "")
        try:
            segs = getattr(tr, "segments", None) or (tr.get("segments") if isinstance(tr, dict) else None) or []
            segment_timestamps = [{{"start": s.get("start"), "end": s.get("end"), "segment": s.get("text")}} for s in segs]
        except Exception:
            pass
        try:
            words = getattr(tr, "words", None) or (tr.get("words") if isinstance(tr, dict) else None) or []
            word_timestamps = [{{"start": w.get("start"), "end": w.get("end"), "word": w.get("word")}} for w in words]
        except Exception:
            pass
        break
    except Exception as _e:
        last_err = _e
        continue

if not transcript_text and last_err:
    raise RuntimeError(f"OpenAI transcription failed across models: {last_err}")

# Compatibility prints
for stamp in segment_timestamps:
    try:
        print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
    except Exception:
        pass
'''
                replaced_count += 1

    # If forced and no specific cell matched, ensure at least one fallback cell exists at top
    if force:
        already_has_fb = any("ASR fallback (auto-injected)" in (c.get("source") or "") for c in nb.cells)
        if not already_has_fb:
            fb = nbformat.v4.new_code_cell('''# --- Streamlit ASR fallback (auto-injected, top-level) ---
AUDIO_FILE = globals().get("AUDIO_FILE", "")
print("Fallback stub in place; set OPENAI_API_KEY to enable ASR if needed.")
''')
            nb.cells.insert(1, fb)  # after parameter cell

    return asr_hit or (replaced_count > 0)

class cd:
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

    nb = nbformat.read(nb_path, as_version=4)
    inject_parameters(nb, params)

    # Rewrite ASR/Colab cells if needed
    try:
        hit = rewrite_asr_cells_for_streamlit(
            nb,
            audio_var_name='AUDIO_FILE',
            require_openai=True,
            force=params.get('FORCE_OPENAI_ASR', True),
            disable_nemo=params.get('DISABLE_NEMO', True),
        )
        if hit and not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError('Notebook expects ASR (NeMo/Colab). Fallback enabled, but OPENAI_API_KEY is missing. Set it in the sidebar.')
    except Exception as _patch_e:
        # Non-fatal; continue. Notebook may not need ASR.
        pass

    executed_nb_path = OUTPUT_DIR / f"executed_{nb_path.name}"

    # Track files pre-execution
    before = {}
    for p in repo_dir.rglob("*"):
        if p.is_file():
            try:
                before[p] = p.stat().st_mtime
            except Exception:
                pass

    # Execute
    kernel_name = ensure_kernel_available("python3")
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata.kernelspec["name"] = kernel_name
    nb.metadata.kernelspec["display_name"] = f"Python ({kernel_name})"

    start_time = time.time()
    with cd(repo_dir):
        client = NotebookClient(nb, timeout=timeout_s, kernel_name=kernel_name, allow_errors=False)
        client.execute()

    nbformat.write(nb, executed_nb_path)

    # Collect new/modified files
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

status = st.empty()

if run_btn:
    if not uploaded:
        st.error("Please upload an audio file first.")
        st.stop()

    # Persist uploaded file
    tmp_audio = Path(tempfile.mkdtemp()) / uploaded.name
    with open(tmp_audio, "wb") as f:
        f.write(uploaded.getbuffer())

    # Determine repo directory
    if exec_mode.startswith("Local"):
        repo_dir = Path(".").resolve()
    else:
        status.info("Cloning/updating repository…")
        repo_dir = clone_or_update_repo(repo_url, branch, CACHE_DIR, keep_repo)

    # Make API key visible to notebook if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    params = {
        "AUDIO_FILE": tmp_audio,
        "ELN_OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "RUN_CONTEXT": "streamlit",
        "FORCE_OPENAI_ASR": force_openai_asr,
        "DISABLE_NEMO": disable_nemo,
    }

    try:
        status.info("Executing notebook…")
        executed_nb_path, artifacts = execute_notebook(repo_dir, NOTEBOOK_REL_PATH, params)
        status.success("Notebook execution finished.")
        st.success("✅ Done! See artifacts below.")

        with st.expander("Execution log / details", expanded=False):
            st.write(f"ASR rewrite forced: {params.get('FORCE_OPENAI_ASR', True)}, NeMo disabled: {params.get('DISABLE_NEMO', True)}")
            st.write(f"Executed notebook saved to: `{executed_nb_path}`")
            st.write(f"Artifacts detected: {len(artifacts)}")

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
        st.download_button("Download executed notebook (.ipynb)",
                           data=Path(executed_nb_path).read_bytes(),
                           file_name=Path(executed_nb_path).name)

    except CellExecutionError as e:
        status.error("Notebook raised an exception during execution.")
        with st.expander("Execution error (traceback)", expanded=True):
            st.exception(e)
    except Exception as e:
        status.error("Unexpected error.")
        with st.expander("Error details", expanded=True):
            st.exception(e)
