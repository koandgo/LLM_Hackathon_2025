# LLM Hackathon 2025 — Speech→ELN Prototype

Short, punchy pitch: **Turn lab talk into lab records.** This prototype converts raw speech (or transcripts) into **structured electronic lab notebook (ELN)** entries using LLMs—accelerating documentation for materials & chemistry workflows.

> Built fast for exploration. Aiming for impact in materials/chemistry through better capture, structure, and reuse of experimental knowledge.

---

## Why this matters (Hackathon framing)

- **Potential for Impact (Mat/Chem):** Automates tedious record-keeping so researchers can iterate faster; unlocks downstream search/RAG over experiments.  
- **Innovativeness & Novelty:** Speech→LLM→ELN pipeline with normalized, template-aligned outputs suitable for eLab-style archives.  
- **Relevance:** Tailored to **materials & chemistry** notebooks (masses, volumes, conditions, observations, timelines).  
- **Exploration-first:** Lightweight notebooks + wrappers so teams can remix quickly.

---

## Repo structure

- `Notebooks/` — working notebooks for experimentation and demos.
- `txt_to_eln_wrapper.ipynb` — minimal wrapper to turn pasted text into an ELN-friendly record.

> Repo currently includes Jupyter notebooks (majority) plus a bit of Python.

---

## What the prototype does

- **Ingests speech or transcripts** (paste, file, or ASR output).
- **Standardizes** numbers, dates (YYYY-MM-DD), times (h:mm am/pm, with TZ if present).
- **Structures** content into ELN-style sections (overview, reagents, steps, results, notes).
- **Emits HTML (ELN-friendly)** and/or JSON stubs ready for export or eLab-style pages.
- **Leaves blanks** for uncertain fields—no hallucinated values.

> The wrapper notebook (`txt_to_eln_wrapper.ipynb`) is the quickest path from text → ELN-ready output.

---

## Quick start

### 1) Environment

- Python 3.10+  
- Jupyter (Lab or Notebook)

Recommended baseline packages (adjust as you evolve):
```bash
pip install jupyter pandas numpy python-dateutil
# Add your model/API stack here (e.g., openai/anthropic/whatever you use for LLM calls)
```

### 2) Run the wrapper

1. Open **`txt_to_eln_wrapper.ipynb`**.  
2. Paste your **transcript or lab text** into the input cell.  
3. Run the notebook to generate **structured HTML** (and optional JSON) you can save into your ELN.

> Tip: keep short example transcripts handy for quick demos.

---

## Suggested workflow (during hackathon)

1. **Collect** transcripts (ASR or manual notes).  
2. **Normalize & segment** (numbers, dates, times; split into sections).  
3. **Map to schema** (materials/chem sections & columns).  
4. **Render** HTML for ELN; optionally export JSON for downstream RAG/search.  
5. **Iterate** on prompts/templates as you see edge cases.

---

## Roadmap (fast wins)

- ✅ Wrapper notebook for text→ELN  
- ⏳ CLI script for batch processing  
- ⏳ eLabFTW export helpers (`.eln` packaging)  
- ⏳ Confidence/uncertainty flags per field  
- ⏳ Optional ASR step (e.g., Whisper) before structuring

---

## Demo data & examples

- Use a short example transcript (e.g., weighing reagents, solvent volumes, temperature ramps).  
- Show before/after: free-form text → clean ELN HTML table.  
- Save the HTML snippet into your ELN to prove integration.

---

## Contributing

PRs welcome—especially:
- small utilities (date/number normalizers),
- schema mappers for common materials/chem sections,
- exporters (eLabFTW, Benchling-style HTML, CSV).

---

## License

_No license file is present yet—add one when ready._

---

## Acknowledgments

Built for the **LLM Hackathon for Applications in Materials & Chemistry (Sept 11–12, 2025)** spirit of rapid exploration.

---
