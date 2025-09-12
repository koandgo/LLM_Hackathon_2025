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
from nbclient import NotebookClient, CellExecutionError

# Repo management utilities
from git import Repo, GitCommandError

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
