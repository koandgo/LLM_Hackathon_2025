# LLM Hackathon 2025 — Speech→ELN Pipeline

 - **Turn audio descriptions of experimental work into lab records.** This prototype converts raw speech into **structured electronic lab notebook (ELN)** entries using automated speech recognition and LLMs—accelerating documentation for materials & chemistry workflows.

> Built fast for exploration. Aiming for impact in materials/chemistry through better capture, structure, and reuse of experimental knowledge.

---

## Why this matters

- **Potential for Impact (Mat/Chem):** Automates tedious record-keeping so researchers can document faster; unlocks downstream search/RAG over experiments.  
- **Innovativeness & Novelty:** Speech→LLM→ELN pipeline with normalized, template-aligned outputs suitable for eLab-style archives.  
- **Relevance:** Tailored to **materials & chemistry** notebooks (masses, volumes, conditions, observations, timelines).  

---

## Repo structure

- `Notebooks/` — working code for experimentation and demos.
- `txt_to_eln_wrapper.ipynb` — minimal wrapper to turn pasted text into an ELN-friendly record.

> Repo currently includes Jupyter notebooks (majority) plus a bit of Python.

---

## What the prototype does

- **Takes and transforms speech** (audio file to transcripts).
- **Standardizes** numbers, dates (YYYY-MM-DD), times (h:mm am/pm, with TZ if present).
- **Structures** content into ELN-style sections (overview, reagents, steps, results, notes).
- **Leaves blanks** for uncertain fields—no hallucinated values.

> The wrapper notebook (`txt_to_eln_wrapper.ipynb`) is the quickest path from text → ELN-ready output.

---

## Quick start

### 1) Environment

- Google Colaboratory
- Run first cell to install package requirements
```

### 2) Run the wrapper

1. Open **`txt_to_eln_wrapper.ipynb`**.  
2. Upload your **audio file** to your google drive, and select file when prompted by script. 
3. Run the notebook to generate **.eln** file to upload to web interface.

> Tip: keep short example transcripts handy for quick demos.
---

## Contributing

PRs welcome—especially:
- small utilities (date/number normalizers),
- schema mappers for common materials/chem sections,
- exporters (eLabFTW, Benchling-style HTML, CSV).

---

## Acknowledgments

Built for the **LLM Hackathon for Applications in Materials & Chemistry (Sept 11–12, 2025)**
---
