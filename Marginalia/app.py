"""Marginalia backend: ingest lecture/textbook/slides, cited Q&A, adaptive quizzes."""
import os, io, re, json, random, time, base64, tempfile
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
# Serverless hosts (e.g. Vercel) have a read-only filesystem except /tmp.
DATA = Path("/tmp/marginalia") if os.getenv("VERCEL") else ROOT / "data"
try: DATA.mkdir(parents=True, exist_ok=True)
except OSError: DATA = Path("/tmp")
STORE = DATA / "store.json"
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = None
if os.getenv("ANTHROPIC_API_KEY"):
    import anthropic
    client = anthropic.Anthropic()

S = {"chunks": [], "mastery": {}, "history": []}
try:
    if STORE.exists(): S = json.loads(STORE.read_text())
except Exception:
    pass
vec = mat = None
app = FastAPI(title="Marginalia")

def save():
    try: STORE.write_text(json.dumps(S))
    except OSError: pass  # keep working in memory if the disk is not writable
def fmt(s): return f"{s // 60:02d}:{s % 60:02d}"
def pub(c): return {k: c[k] for k in ("id", "type", "ref", "source", "text", "concept") if k in c}

def llm(prompt, system="", max_tokens=700):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

# ---------- indexing ----------
def reindex():
    global vec, mat
    ch = S["chunks"]
    if not ch:
        vec = mat = None; S["mastery"] = {}; return
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    mat = vec.fit_transform([c["text"] for c in ch])
    k = max(1, min(8, len(ch) // 3))
    labels = KMeans(k, n_init=5, random_state=0).fit_predict(mat) if k > 1 else [0] * len(ch)
    terms = np.array(vec.get_feature_names_out()); names = {}
    for g in set(labels):
        idx = [i for i, l in enumerate(labels) if l == g]
        top = np.asarray(mat[idx].mean(axis=0)).ravel().argsort()[::-1][:2]
        names[g] = " / ".join(terms[top])
    for c, l in zip(ch, labels): c["concept"] = names[l]
    cs = set(names.values())
    S["mastery"] = {c: S["mastery"].get(c, 0.3) for c in cs}

def retrieve(q, k=4):
    if vec is None: return []
    s = cosine_similarity(vec.transform([q]), mat).ravel()
    return [S["chunks"][i] for i in s.argsort()[::-1][:k] if s[i] > 0.02]

# ---------- parsers: each returns ([(ref, text)], type) ----------
def chunk_text(text, size=650):
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()); out, buf = [], ""
    for s in sents:
        if buf and len(buf) + len(s) > size: out.append(buf); buf = ""
        buf = (buf + " " + s).strip()
    if buf: out.append(buf)
    return out

def parse_pdf(data):
    from pypdf import PdfReader
    pages = PdfReader(io.BytesIO(data)).pages
    return [(f"p.{i+1}", t) for i, p in enumerate(pages) for t in chunk_text(p.extract_text() or "")], "Textbook"

def caption(blob, mt):
    """Slide image understanding via Claude vision (only when an API key is set)."""
    try:
        r = client.messages.create(model=MODEL, max_tokens=200, messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": base64.b64encode(blob).decode()}},
            {"type": "text", "text": "Describe this lecture slide figure in 1-2 factual sentences, including labels and numbers."}]}])
        return r.content[0].text
    except Exception:
        return ""

def parse_pptx(data):
    from pptx import Presentation
    out = []
    for i, sl in enumerate(Presentation(io.BytesIO(data)).slides, 1):
        txt = []
        for sh in sl.shapes:
            if sh.has_text_frame: txt.append(sh.text_frame.text)
            if client and sh.shape_type == 13:
                cap = caption(sh.image.blob, sh.image.content_type)
                if cap: txt.append("[Figure] " + cap)
        if sl.has_notes_slide: txt.append(sl.notes_slide.notes_text_frame.text)
        t = " ".join(x for x in txt if x.strip())
        if t.strip(): out.append((f"slide {i}", t))
    return out, "Slide"

TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,]?\d*\s*-->")
def group_segments(segs, window=45):
    out, start, buf = [], segs[0][0], ""
    for s, l in segs:
        if buf and s - start >= window: out.append((fmt(start), buf)); start, buf = s, ""
        buf = (buf + " " + l).strip()
    if buf: out.append((fmt(start), buf))
    return out

def parse_lecture(text):
    segs, cur = [], None
    for line in text.splitlines():
        m = TS.search(line)
        if m: cur = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3)); continue
        line = line.strip()
        if line and cur is not None and not line.isdigit() and line != "WEBVTT": segs.append((cur, line))
    if segs: return group_segments(segs), "Lecture"
    return [(f"para {i+1}", t) for i, t in enumerate(chunk_text(text))], "Notes"

def parse_media(data, name):
    try: from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(400, "Audio/video needs `pip install faster-whisper` and ffmpeg. Or upload an .srt/.vtt transcript.")
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as f:
        f.write(data); f.flush()
        segs, _ = WhisperModel("base", compute_type="int8").transcribe(f.name)
        segs = [(int(s.start), s.text.strip()) for s in segs if s.text.strip()]
    if not segs: raise HTTPException(400, "No speech found in that file.")
    return group_segments(segs), "Lecture"

MEDIA = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".mkv"}
def add(items, typ, source):
    nid = max([c["id"] for c in S["chunks"]], default=0) + 1
    for ref, t in items:
        S["chunks"].append({"id": nid, "type": typ, "ref": ref, "source": source, "text": t}); nid += 1

# ---------- API ----------
@app.get("/api/state")
def state():
    m = S["mastery"]; weak = sorted(m, key=m.get)[:3]
    plan = [{"concept": c, "mastery": m[c],
             "sources": [pub(x) for x in S["chunks"] if x.get("concept") == c][:2]} for c in weak]
    return {"chunks": [pub(c) for c in S["chunks"]], "mastery": m, "history": S["history"][-30:],
            "plan": plan, "llm": bool(client)}

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    for f in files:
        data, ext = await f.read(), Path(f.filename).suffix.lower()
        if ext == ".pdf": items, typ = parse_pdf(data)
        elif ext == ".pptx": items, typ = parse_pptx(data)
        elif ext in (".srt", ".vtt", ".txt", ".md"): items, typ = parse_lecture(data.decode("utf-8", "ignore"))
        elif ext in MEDIA: items, typ = parse_media(data, f.filename)
        else: raise HTTPException(400, f"Unsupported file type: {ext or f.filename}")
        if not items: raise HTTPException(400, f"No readable text found in {f.filename}.")
        add(items, typ, f.filename)
    reindex(); save(); return state()

SAMPLE = [
 ("Lecture", "04:12", "A neural network makes a prediction, then a loss function measures how wrong it was. Training means changing the weights to make that loss smaller."),
 ("Lecture", "07:40", "Gradient descent computes the gradient of the loss with respect to each weight, then steps each weight a small amount in the opposite direction. The step size is the learning rate."),
 ("Lecture", "15:05", "If the learning rate is too large the loss overshoots and may diverge. If it is too small, training crawls or stalls in a flat region."),
 ("Textbook", "p.141", "Backpropagation applies the chain rule layer by layer from the output back to the input, reusing intermediate results so all gradients are computed in one backward pass."),
 ("Textbook", "p.148", "Stochastic gradient descent estimates the gradient from a small random mini-batch rather than the full dataset, trading exact gradients for much cheaper steps."),
 ("Textbook", "p.153", "Overfitting occurs when a model memorizes training data and fails on new data. Regularization, dropout and early stopping on a validation set reduce it."),
 ("Slide", "slide 9", "Activation functions such as ReLU and sigmoid add non-linearity. Without them, stacked layers collapse into a single linear map."),
 ("Slide", "slide 12", "An epoch is one pass over the full training set. A batch is the set of examples used for one weight update."),
 ("Slide", "slide 17", "Validation loss rising while training loss keeps falling is the signal that the model is overfitting.")]

@app.post("/api/sample")
def sample():
    S.update(chunks=[], mastery={}, history=[])
    for t, r, x in SAMPLE: add([(r, x)], t, "sample-course")
    reindex(); save(); return state()

@app.post("/api/reset")
def reset():
    S.update(chunks=[], mastery={}, history=[]); reindex(); save(); return state()

class Ask(BaseModel): q: str
@app.post("/api/ask")
def ask(a: Ask):
    hits = retrieve(a.q)
    if not hits: return {"answer": "Nothing in your materials matches that question.", "sources": [], "mode": "none"}
    if client:
        ctx = "\n".join(f"[{i+1}] ({h['type']} {h['ref']}) {h['text']}" for i, h in enumerate(hits))
        ans = llm(f"Sources:\n{ctx}\n\nQuestion: {a.q}",
                  system="You are a study tutor. Answer using ONLY the numbered sources. Cite every claim as [n]. "
                         "If the sources do not answer the question, say so. Max 120 words.")
        mode = "llm"
    else:
        ans = " ".join(f"{h['text']} [{i+1}]" for i, h in enumerate(hits[:2])); mode = "extractive"
    return {"answer": ans, "sources": [pub(h) for h in hits], "mode": mode}

def cloze(src):
    terms = vec.get_feature_names_out(); i = S["chunks"].index(src)
    words = set(re.findall(r"[a-z]+", src["text"].lower()))
    row = mat[i].toarray().ravel()
    cands = [terms[j] for j in row.argsort()[::-1] if " " not in terms[j] and terms[j] in words]
    ans = cands[0]
    sent = next((s for s in re.split(r"(?<=[.!?])\s+", src["text"]) if re.search(rf"\b{ans}\b", s, re.I)), src["text"])
    other = [terms[j] for j in mat.sum(axis=0).A1.argsort()[::-1] if " " not in terms[j] and terms[j] not in words][:30]
    opts = [ans] + random.sample(other, min(3, len(other))); random.shuffle(opts)
    return {"q": "Fill in the blank: " + re.sub(rf"\b{ans}\b", "_____", sent, flags=re.I),
            "options": opts, "answer": opts.index(ans)}

def gen_question(src, hard):
    p = (f"Passage ({src['type']} {src['ref']}):\n{src['text']}\n\nWrite ONE {'harder, application-style' if hard else 'foundational'} "
         'multiple-choice question answerable only from this passage. Reply with JSON only: '
         '{"q": str, "options": [3 or 4 strings], "answer": index of correct option}')
    try:
        j = json.loads(re.search(r"\{.*\}", llm(p, max_tokens=400), re.S).group(0))
        if 2 <= len(j["options"]) <= 4 and 0 <= int(j["answer"]) < len(j["options"]):
            return {"q": j["q"], "options": j["options"], "answer": int(j["answer"])}
    except Exception:
        pass
    return None

class QReq(BaseModel): concept: str | None = None
@app.post("/api/quiz/next")
def quiz_next(r: QReq):
    if not S["mastery"]: raise HTTPException(400, "Upload materials or load the sample course first.")
    c = r.concept if r.concept in S["mastery"] else min(S["mastery"], key=S["mastery"].get)
    hard = S["mastery"][c] >= .5
    src = random.choice([x for x in S["chunks"] if x.get("concept") == c])
    q = (gen_question(src, hard) if client else None) or cloze(src)
    return {**q, "concept": c, "difficulty": 2 if hard else 1, "source": pub(src), "mode": "llm" if client else "cloze"}

class Ans(BaseModel): concept: str; correct: bool; difficulty: int
@app.post("/api/quiz/answer")
def quiz_answer(a: Ans):
    m = S["mastery"].get(a.concept, .3)
    S["mastery"][a.concept] = m + (1 - m) * (.45 if a.difficulty == 2 else .35) if a.correct else m * .7
    S["history"].append({"t": int(time.time()), "avg": sum(S["mastery"].values()) / len(S["mastery"])})
    save(); return state()

reindex()
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
