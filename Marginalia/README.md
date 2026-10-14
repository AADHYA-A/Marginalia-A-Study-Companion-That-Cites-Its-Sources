# Marginalia: a source-cited study companion

Submission for the **Multimodal AI Hackathon 2026, Track D: Personalized Tutoring & Adaptive Learning**.

Upload a lecture (transcript, audio or video), a textbook PDF and a slide deck. Marginalia merges them into one knowledge base, answers questions with citations to the exact minute, page or slide, and runs an adaptive quiz that follows your weakest concepts.

## Features

- **Real file ingestion.** PDF textbooks (cited by page), PPTX slides (cited by slide, including speaker notes), SRT/VTT/TXT transcripts (cited by timestamp), and MP3/MP4 audio/video lectures (transcribed with Whisper, cited by timestamp).
- **Slide image understanding.** With an API key, figures inside PPTX slides are described by Claude's vision and become searchable text.
- **Grounded answers.** With an API key, Claude writes the answer using only the retrieved passages and cites each claim as `[n]`, rendered as clickable chips (e.g. `Slide 12`, `Lecture 07:40`). If nothing matches, the app says so.
- **Generated adaptive quizzes.** Claude writes a question from a passage of your weakest concept (foundation or stretch, based on mastery). Without a key, a fill-in-the-blank question is built from the passage.
- **Mastery tracking and study plan.** Per-concept mastery, a mastery-over-time chart, and a plan listing the three weakest concepts with the passages to re-read. Progress is saved between sessions.
- **Works offline of any AI key.** Without `ANTHROPIC_API_KEY` the app uses quoted answers and fill-in-the-blank quizzes.

## How it works

1. **Parse and chunk.** Each file becomes chunks of about 650 characters (or 45-second transcript windows), each with `type`, `ref` (page, slide or timestamp), `source` and `text`.
2. **Index.** TF-IDF (unigrams and bigrams) over all chunks. KMeans groups chunks into up to 8 concepts, named by their top terms.
3. **Retrieve.** Cosine similarity between the question and every chunk returns the top 4.
4. **Answer.** Claude answers from those passages only and cites them, or the passages are quoted directly when no key is set.
5. **Adapt.** Mastery per concept starts at 0.3. Correct: `m ← m + α(1 − m)` with α = 0.35 (foundation) or 0.45 (stretch). Wrong: `m ← 0.7m`. The next question comes from the lowest-mastery concept, at foundation level below 0.5 and stretch level above.

## Prerequisites

- Python 3.10+
- Optional: an Anthropic API key (generated answers, generated quizzes, slide figure descriptions)
- Optional: `ffmpeg` and `pip install faster-whisper` for audio/video lectures

## Setup and run

```bash
git clone https://github.com/<your-username>/marginalia.git
cd marginalia/backend
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional, enables Claude-powered features
export ANTHROPIC_API_KEY=your-key-here                # Windows: set ANTHROPIC_API_KEY=your-key-here
# export CLAUDE_MODEL=claude-sonnet-5-5               # optional override

uvicorn app:app --reload
```

Open http://localhost:8000.

## Try it

1. Click **Load sample course**, or choose your own PDF, PPTX and transcript files and click **Add files**.
2. On **Ask**, enter a question and click the citation chips to see the source passage.
3. On **Adaptive quiz**, answer a few questions and watch the mastery bars change.
4. On **Progress**, view the mastery trend and the study plan.

## Project structure

```
marginalia/
├── backend/
│   ├── app.py              # FastAPI: parsing, retrieval, Q&A, quiz, mastery
│   ├── requirements.txt
│   ├── .env.example
│   └── static/index.html   # web UI (served by the backend)
├── static-demo/index.html  # original no-backend demo (hardcoded sample course)
├── README.md
└── LICENSE
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Add PDF, PPTX, SRT/VTT/TXT or media files |
| `POST /api/sample` | Load the sample course |
| `POST /api/ask` | `{q}` returns an answer and its sources |
| `POST /api/quiz/next` | Returns a question for the weakest (or a chosen) concept |
| `POST /api/quiz/answer` | Updates mastery |
| `GET /api/state` | Passages, mastery, history, study plan |
| `POST /api/reset` | Clear everything |

## Limitations

- Retrieval is keyword-based (TF-IDF), not embedding-based, so it can miss paraphrases.
- Concept names come from top terms of each cluster and can be rough (for example "set / pass") on small inputs.
- Scanned PDFs without a text layer are not read (no OCR).
- One shared learner profile, stored in `backend/data/store.json`. There are no user accounts.
- Slide figure descriptions need an API key; audio/video needs Whisper and ffmpeg.
- Tested with the sample course, SRT and PPTX uploads. PDF parsing, Whisper transcription and the Claude-powered paths were not exercised in testing.

## Roadmap

- Embedding-based retrieval and LLM concept naming.
- OCR for scanned textbooks.
- Spaced repetition and per-user accounts.

## License

MIT
