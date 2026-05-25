"""
MCP Prompt definitions for the Pixeltable MCP server.

Prompts guide LLMs through common multimodal workflows using Pixeltable.
Each prompt is registered as an MCP Prompt primitive that clients can request.
"""

# =============================================================================
# General Usage Prompt (existing)
# =============================================================================

PIXELTABLE_USAGE_PROMPT = """\
You are helping a user work with Pixeltable, a multimodal AI data infrastructure platform through its MCP server.

## Critical pitfalls (do these correctly the first time)

1. There is no `openai.vision` -- use `openai.chat_completions` with `image_url` content blocks:
   `{"type": "image_url", "image_url": {"url": t.image}}`
2. Cast to `pxt.String` before embedding -- `.text.astype(pxt.String)` on AI function outputs before `add_embedding_index`.
3. `if_exists='ignore'` will NOT fix a broken computed column. To change column logic: `drop_column()` then recreate.
4. Import `frame_iterator` as a function: `from pixeltable.functions.video import frame_iterator` (NOT `pixeltable.iterators.FrameIterator`).
5. Use `string=` keyword in similarity: `t.col.similarity(string=query)`, never positional.
6. Use `embedding=` (not `string_embed=`) in `add_embedding_index`.
7. Image content blocks use `image_url`, not `data`: `{"type": "image_url", "image_url": {"url": t.image}}`.
8. Don't write `for row in ...` loops calling AI models -- use computed columns.
9. Don't install a separate vector DB (Pinecone, Chroma, FAISS, Qdrant, pgvector). Use `add_embedding_index` + `.similarity()`.
10. Don't write `while not done:` agent loops -- use a table with a computed-column chain triggered by insert.

## Working with Pixeltable

1. Pixeltable is multimodal AI data infrastructure (images, videos, audio, documents, structured), not a generic database.
2. Pattern: create table -> insert data -> add computed columns (AI) -> query. Inserts cascade through computed columns automatically.
3. Common workflows:
   - Image analysis: table -> insert images -> YOLOX / DETR / chat_completions on `image_url` -> query
   - RAG: table -> insert documents -> view via `document_splitter` -> `add_embedding_index` -> `.similarity(string=...)`
   - Video: table -> insert videos -> view via `frame_iterator` -> per-frame analysis -> `extract_audio` -> `transcriptions`
   - Agent: table -> `messages(... tools=pxt.tools(...))` -> `invoke_tools` -> retrieval columns -> assemble -> final `messages` -> answer
4. Providers live under `pixeltable.functions.*` (openai, anthropic, gemini, ollama, fireworks, together, mistralai, groq, replicate, huggingface, whisper, yolox, ...).
5. MCP Resources expose read-only context cheaply: `pixeltable://tables`, `pixeltable://version`, `pixeltable://diagnostics`, `pixeltable://help`. Read these before calling mutating tools.
6. Always use `if_exists='ignore'` on `create_*` / `add_*` so workflows stay idempotent.
7. Use `execute_python` for one-off imperative logic (e.g. creating views with iterators) and `introspect_function` to look up signatures.
8. Use `log_bug` / `log_missing_feature` to capture issues during exploration.

Always aim to make Pixeltable accessible and useful for the user's specific multimodal AI data needs.
"""

# =============================================================================
# Getting Started Prompt
# =============================================================================

GETTING_STARTED_PROMPT = """\
Guide the user through their first Pixeltable workflow. Follow these steps in order:

## Step 1: Initialize
Check if Pixeltable is working:
- Read the `pixeltable://version` resource to confirm the version
- Read the `pixeltable://config/datastore` resource to see the current datastore

## Step 2: Create a Directory (optional)
If the user wants to organize tables:
```python
pixeltable_create_dir("my_project", if_exists="ignore")
```

## Step 3: Create a Table
Ask the user what kind of data they want to work with. Then create a table:
```python
# For images:
pixeltable_create_table("my_project.photos", schema={"image": "Image", "caption": "String"})

# For documents:
pixeltable_create_table("my_project.docs", schema={"document": "Document", "title": "String"})

# For structured data:
pixeltable_create_table("my_project.data", schema={"text": "String", "score": "Float"})
```

## Step 4: Insert Data
Help the user add data:
```python
pixeltable_insert_data("my_project.photos", [
    {"image": "https://example.com/photo1.jpg", "caption": "A sunset"},
    {"image": "/path/to/local/image.jpg", "caption": "My photo"}
])
```

## Step 5: Query
Show the data:
```python
pixeltable_query_table("my_project.photos", limit=5)
```

## Step 6: Add AI (the exciting part!)
Add a computed column with an AI model:
```python
pixeltable_add_computed_column("my_project.photos", "description",
    "openai.chat_completions(messages=[{'role':'user','content':[{'type':'text','text':'Describe this image'},{'type':'image_url','image_url':{'url':table.image}}]}], model='gpt-4o-mini')")
```

## Tips
- Read `pixeltable://types` to see available data types
- Read `pixeltable://diagnostics` to check installed dependencies
- Use `pixeltable_check_dependencies("openai")` before adding AI columns
- All computed columns update automatically when new data is inserted
"""

# =============================================================================
# Computer Vision Pipeline Prompt
# =============================================================================

COMPUTER_VISION_PROMPT = """\
Guide the user through building a computer vision pipeline with Pixeltable.

## Overview
Pixeltable makes it easy to run vision models on collections of images. Computed columns automatically process every image, including new ones added later.

## Step 1: Set Up the Image Table
```python
pixeltable_create_table("cv.images", schema={
    "image": "Image",
    "source": "String",
    "label": "String"
})
```

## Step 2: Check Dependencies
Before adding vision models, verify dependencies:
```python
pixeltable_check_dependencies("yolox openai")
```
Install what's missing:
```python
pixeltable_install_dependency("yolox")   # For object detection
pixeltable_install_dependency("openai")  # For GPT-4 / chat_completions
```

## Step 3: Add Images
```python
pixeltable_insert_data("cv.images", [
    {"image": "https://example.com/street.jpg", "source": "web", "label": "street"},
    {"image": "/path/to/photo.jpg", "source": "local", "label": "nature"}
])
```

## Step 4: Add Computer Vision Computed Columns

### Object Detection (YOLOX)
```python
pixeltable_add_computed_column("cv.images", "detections",
    "yolox.yolox(table.image, model_id='yolox_m', threshold=0.5)")
```

### Image Description (GPT-4 Vision)
```python
pixeltable_add_computed_column("cv.images", "description",
    "openai.chat_completions(messages=[{'role':'user','content':[{'type':'text','text':'Describe this image in detail'},{'type':'image_url','image_url':{'url':table.image}}]}], model='gpt-4o-mini')")
```

### Image Metadata
```python
pixeltable_add_computed_column("cv.images", "width", "image.width(table.image)")
pixeltable_add_computed_column("cv.images", "height", "image.height(table.image)")
```

## Step 5: Query Results
```python
pixeltable_query_table("cv.images", limit=10)
```

## Key Concepts
- **Incremental**: Adding new images triggers automatic processing
- **Model Comparison**: Add multiple detection/description columns with different models
- **Views**: Create filtered views (e.g., only images with detected people)
- **Snapshots**: Save a point-in-time copy before changing models
"""

# =============================================================================
# RAG Pipeline Prompt
# =============================================================================

RAG_PIPELINE_PROMPT = """\
Guide the user through building a Retrieval-Augmented Generation (RAG) pipeline with Pixeltable.

## Overview
Pixeltable handles the entire RAG pipeline: document ingestion, text extraction, chunking, embedding generation, and similarity search -- all as computed columns that update automatically.

## Step 1: Create a Documents Table
```python
pixeltable_create_table("rag.documents", schema={
    "document": "Document",
    "title": "String",
    "category": "String"
})
```

## Step 2: Check Dependencies
```python
pixeltable_check_dependencies("openai sentence-transformers")
```
Install what's needed:
```python
pixeltable_install_dependency("openai")
pixeltable_install_dependency("sentence-transformers")
```

## Step 3: Ingest Documents
```python
pixeltable_insert_data("rag.documents", [
    {"document": "/path/to/paper.pdf", "title": "Research Paper", "category": "academic"},
    {"document": "/path/to/manual.pdf", "title": "User Manual", "category": "technical"}
])
```

## Step 4: Create a Chunks View
Use Pixeltable's document splitter to chunk documents into searchable rows (see pixeltable-skill: `document_splitter`):
```python
# Use the REPL for more complex operations
execute_python('''
import pixeltable as pxt
from pixeltable.functions.document import document_splitter

docs = pxt.get_table("rag.documents")
chunks = pxt.create_view(
    "rag.chunks",
    docs,
    iterator=document_splitter(docs.document, separators="token_limit", limit=300),
    if_exists="ignore",
)
''')
```

## Step 5: Add Embeddings
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

chunks = pxt.get_table("rag.chunks")
embed_fn = sentence_transformer.using(model_id="all-MiniLM-L6-v2")
chunks.add_embedding_index("text", embedding=embed_fn, if_exists="ignore")
''')
```

## Step 6: Search
Use `similarity(string=...)` on the embedded column (keyword argument required):
```python
execute_python('''
import pixeltable as pxt
chunks = pxt.get_table("rag.chunks")
sim = chunks.text.similarity(string="How does X work?")
results = chunks.order_by(sim, asc=False).limit(5).select(chunks.text, sim).collect()
print(list(results))
''')
```

## Step 7: Add LLM Response Column
Generate answers using retrieved context:
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions import openai

chunks = pxt.get_table("rag.chunks")
# Use similarity search results as context for an LLM
# This can be done as a computed column for automatic RAG
''')
```

## Key Concepts
- **Automatic Chunking**: `document_splitter` iterator expands each document into chunk rows
- **Embedding Indexes**: Built-in vector search with multiple embedding models
- **Incremental Updates**: New documents are automatically chunked and embedded
- **Hybrid Search**: Combine vector similarity with structured filters
"""

# =============================================================================
# Video Analysis Prompt
# =============================================================================

VIDEO_ANALYSIS_PROMPT = """\
Guide the user through building a video analysis pipeline with Pixeltable.

## Overview
Pixeltable can extract frames from videos, run AI models on each frame, transcribe audio, and aggregate results -- all using computed columns and views.

## Step 1: Create a Videos Table
```python
pixeltable_create_table("video.clips", schema={
    "video": "Video",
    "title": "String",
    "source": "String"
})
```

## Step 2: Insert Videos
```python
pixeltable_insert_data("video.clips", [
    {"video": "/path/to/video.mp4", "title": "Meeting Recording", "source": "local"},
    {"video": "https://example.com/clip.mp4", "title": "Tutorial", "source": "web"}
])
```

## Step 3: Extract Frames
Create a view that extracts frames using `frame_iterator` from `pixeltable.functions.video` (not `pixeltable.iterators`):
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator

clips = pxt.get_table("video.clips")
frames = pxt.create_view(
    "video.frames",
    clips,
    iterator=frame_iterator(clips.video, fps=1.0),
    if_exists="ignore",
)
''')
```

## Step 4: Analyze Frames
Add AI models to process each extracted frame:

### Object Detection
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.yolox import yolox

frames = pxt.get_table("video.frames")
frames.add_computed_column(detections=yolox(frames.frame, model_id="yolox_m", threshold=0.5), if_exists="ignore")
''')
```

### Scene Description
Use `chat_completions` with image_url blocks (there is no `openai.vision`):
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions

frames = pxt.get_table("video.frames")
frames.add_computed_column(
    scene_description=chat_completions(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Briefly describe this video frame"},
            {"type": "image_url", "image_url": {"url": frames.frame}},
        ]}],
        model="gpt-4o-mini",
    ).choices[0].message.content,
    if_exists="ignore",
)
''')
```

## Step 5: Audio Transcription
```python
execute_python('''
import pixeltable as pxt
clips = pxt.get_table("video.clips")
# Extract audio and transcribe with Whisper
clips.add_computed_column(
    transcription=openai.transcriptions(audio=clips.video, model="whisper-1")
)
''')
```

## Step 6: Query Results
```python
pixeltable_query_table("video.frames", limit=20)
```

## Key Concepts
- **Frame Extraction**: `frame_iterator` yields one row per sampled frame
- **Per-Frame Analysis**: Computed columns run on every extracted frame
- **Audio + Video**: Process both tracks independently, then join
- **Temporal Queries**: Filter frames by timestamp or detected objects
"""

# =============================================================================
# Audio Processing Prompt
# =============================================================================

AUDIO_PROCESSING_PROMPT = """\
Guide the user through building an audio processing pipeline with Pixeltable.

## Overview
Pixeltable can transcribe audio, extract features, and index spoken content for search -- all automatically via computed columns.

## Step 1: Create an Audio Table
```python
pixeltable_create_table("audio.recordings", schema={
    "audio": "Audio",
    "title": "String",
    "speaker": "String"
})
```

## Step 2: Check Dependencies
```python
pixeltable_check_dependencies("whisper openai")
```
Install what's needed:
```python
pixeltable_install_dependency("whisper")
pixeltable_install_dependency("openai")
```

## Step 3: Insert Audio Files
```python
pixeltable_insert_data("audio.recordings", [
    {"audio": "/path/to/recording.mp3", "title": "Interview", "speaker": "Alice"},
    {"audio": "/path/to/podcast.wav", "title": "Episode 1", "speaker": "Bob"}
])
```

## Step 4: Add Transcription
Using OpenAI Whisper API:
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions import openai

recordings = pxt.get_table("audio.recordings")
recordings.add_computed_column(
    transcription=openai.transcriptions(audio=recordings.audio, model="whisper-1")
)
''')
```

## Step 5: Add Analysis
Analyze transcriptions with an LLM:
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions import openai

recordings = pxt.get_table("audio.recordings")
recordings.add_computed_column(
    summary=openai.chat_completions(
        messages=[{"role": "user", "content": "Summarize this transcript:\\n" + recordings.transcription}],
        model="gpt-4o-mini"
    )
)
''')
```

## Step 6: Search Transcriptions
Create an embedding index for semantic search over transcriptions:
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

recordings = pxt.get_table("audio.recordings")
recordings.add_embedding_index(
    "transcription",
    embedding=sentence_transformer.using(model_id="all-MiniLM-L6-v2"),
    if_exists="ignore",
)

# Search (use .similarity(string=...) on the embedded column; order_by + limit)
sim = recordings.transcription.similarity(string="pricing discussion")
results = recordings.order_by(sim, asc=False).limit(5).select(
    recordings.title, recordings.transcription, sim,
).collect()
print(list(results))
''')
```

## Key Concepts
- **Automatic Transcription**: Computed columns transcribe on insert
- **Multi-Model**: Use Whisper for transcription, GPT for analysis, embeddings for search
- **Incremental**: New audio files are automatically processed
- **Searchable**: Embedding indexes enable semantic search over spoken content
"""


# =============================================================================
# Tool-Calling Agent Pipeline (from pixeltable-skill workflows.md)
# =============================================================================

TOOL_CALLING_AGENT_PROMPT = """\
Guide the user through building a production-grade tool-calling agent with Pixeltable.

The agent lives in a table; each insert triggers the full DAG of computed columns:
LLM picks tools -> tools run -> RAG retrieval -> assemble context -> final LLM -> answer.
There are NO while-loops or async orchestration code.

## Step 1: Define tools as @pxt.udf / @pxt.query

```python
execute_python('''
import pixeltable as pxt
from ddgs import DDGS

@pxt.udf
def web_search(query: str) -> list:
    return list(DDGS().text(query, max_results=5))

# Document-search query bound to an existing knowledge-base table.
docs = pxt.get_table("kb.docs")

@pxt.query
def search_documents(query_text: str, limit: int = 5):
    sim = docs.content.similarity(string=query_text)
    return docs.order_by(sim, asc=False).limit(limit).select(docs.title, docs.content, sim)
''')
```

## Step 2: Build pxt.tools() and bind it in the REPL

Use the MCP helper so the resulting collection is reusable later:
```python
pixeltable_create_tools(
    function_names=["web_search", "search_documents"],
    register_as="tools",
)
```

## Step 3: Create the agent table

```python
pixeltable_create_table("agents.tool_agent", schema={
    "prompt": "String",
    "timestamp": "Timestamp",
    "system_prompt": "String",
    "max_tokens": "Int",
}, if_exists="ignore")
```

## Step 4: Compose the agent chain

```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.anthropic import messages, invoke_tools

@pxt.udf
def assemble_context(question: str, tool_outputs: list | None, doc_context: list | None) -> str:
    tools_str = str(tool_outputs) if tool_outputs else "N/A"
    docs_str = "\\n".join(
        f"- {item.get('text', '')}" for item in (doc_context or []) if isinstance(item, dict)
    ) or "N/A"
    return f"QUESTION: {question}\\n\\n[TOOL RESULTS]\\n{tools_str}\\n\\n[DOCUMENTS]\\n{docs_str}"

agent = pxt.get_table("agents.tool_agent")

# 1. LLM selects tools (tool_choice forces a tool call so invoke_tools can run).
agent.add_computed_column(initial_response=messages(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": [{"type": "text", "text": agent.prompt}]}],
    tools=tools, tool_choice=tools.choice(required=True),
    max_tokens=agent.max_tokens,
    model_kwargs={"system": agent.system_prompt},
), if_exists="ignore")

# 2. Execute tools.
agent.add_computed_column(tool_output=invoke_tools(tools, agent.initial_response), if_exists="ignore")

# 3. Run RAG retrieval in parallel (just declare another column).
agent.add_computed_column(doc_context=search_documents(agent.prompt), if_exists="ignore")

# 4. Assemble final context.
agent.add_computed_column(context=assemble_context(agent.prompt, agent.tool_output, agent.doc_context), if_exists="ignore")

# 5. Final LLM pass.
agent.add_computed_column(final_response=messages(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": [{"type": "text", "text": agent.context}]}],
    max_tokens=agent.max_tokens,
    model_kwargs={"system": agent.system_prompt},
), if_exists="ignore")

agent.add_computed_column(answer=agent.final_response.content[0].text, if_exists="ignore")
''')
```

## Step 5: Invoke the agent (one row in -> entire chain runs)

```python
pixeltable_insert_data("agents.tool_agent", [{
    "prompt": "What is quantum computing? Cite the docs.",
    "timestamp": "2026-05-24T00:00:00",
    "system_prompt": "You are a helpful research assistant.",
    "max_tokens": 1024,
}])
pixeltable_query_table("agents.tool_agent", limit=1)
```

## Anti-patterns to avoid (from pixeltable-skill)
- NO `while not done:` loops -- the table + chain IS the agent.
- NO LangChain / LangGraph / LlamaIndex -- pxt.tools + invoke_tools is the whole runtime.
- NO separate vector DB -- use `add_embedding_index` + `.similarity(string=...)`.
- Match the `invoke_tools` import to the provider you call (anthropic / openai / groq / gemini / bedrock).
"""


# =============================================================================
# Agent With Memory Pipeline (from pixeltable-skill agents-memory-mcp.md)
# =============================================================================

AGENT_WITH_MEMORY_PROMPT = """\
Guide the user through an agent that has persistent chat history and a memory bank.
All state lives in Pixeltable tables; recall is done with embedding indexes and `user_id`
scoping so the same agent works for many users.

## Step 1: Chat history + memory bank tables (one of each per agent, NOT per user)

```python
pixeltable_create_table("memory.chat_history", schema={
    "user_id":   "Required[String]",
    "role":      "String",
    "content":   "String",
    "timestamp": "Timestamp",
}, if_exists="ignore")

pixeltable_create_table("memory.memory_bank", schema={
    "user_id":   "Required[String]",
    "content":   "String",
    "category":  "String",
    "timestamp": "Timestamp",
}, if_exists="ignore")
```

## Step 2: Embedding indexes for recall

```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

embed_fn = sentence_transformer.using(model_id="all-MiniLM-L6-v2")
pxt.get_table("memory.chat_history").add_embedding_index("content", embedding=embed_fn, if_exists="ignore")
pxt.get_table("memory.memory_bank").add_embedding_index("content", embedding=embed_fn, if_exists="ignore")
''')
```

## Step 3: Recall queries (always scoped by user_id, always `string=` keyword)

```python
execute_python('''
import pixeltable as pxt

chat = pxt.get_table("memory.chat_history")
mem  = pxt.get_table("memory.memory_bank")

@pxt.query
def recall_chat(user_id: str, query: str, limit: int = 5):
    sim = chat.content.similarity(string=query)
    return chat.where(chat.user_id == user_id).order_by(sim, asc=False).limit(limit).select(chat.role, chat.content, sim)

@pxt.query
def recall_memories(user_id: str, query: str, limit: int = 5):
    sim = mem.content.similarity(string=query)
    return mem.where(mem.user_id == user_id).order_by(sim, asc=False).limit(limit).select(mem.category, mem.content, sim)
''')
```

## Step 4: save_memory tool (clear docstring matters for tool selection)

```python
execute_python('''
import pixeltable as pxt
from datetime import datetime

@pxt.udf
def save_memory(user_id: str, content: str, category: str = "general") -> str:
    # Docstring matters: LLMs select tools based on it.
    "Save a durable fact about the user (preferences, profile, decisions)."
    mem = pxt.get_table("memory.memory_bank")
    mem.insert([{"user_id": user_id, "content": content, "category": category, "timestamp": datetime.now()}])
    return "Saved."
''')
```

## Step 5: Bundle local + MCP tools

```python
# Optional: pull tools from another MCP server (in-process binding lives in the REPL).
pixeltable_connect_mcp(url="http://localhost:8080/mcp", register_as="mcp_tools")

pixeltable_create_tools(
    function_names=["save_memory", "recall_chat", "recall_memories", "*mcp_tools"],
    register_as="tools",
)
```

## Step 6: Agent table chain (user_id flows through every column)

Build a chain exactly like the tool-calling agent prompt, but include
`chat_context = recall_chat(agent.user_id, agent.prompt)` and
`memory_context = recall_memories(agent.user_id, agent.prompt)` columns that
get merged into `assemble_context`.

## Step 7: Persist the new turn AFTER the answer is materialized

```python
execute_python('''
import pixeltable as pxt
from datetime import datetime

agent = pxt.get_table("agents.memory_agent")
chat  = pxt.get_table("memory.chat_history")

result = agent.where(agent.prompt == "<the prompt>").select(agent.user_id, agent.answer).collect()
for row in result:
    chat.insert([
        {"user_id": row["user_id"], "role": "user",      "content": "<the prompt>", "timestamp": datetime.now()},
        {"user_id": row["user_id"], "role": "assistant", "content": row["answer"],  "timestamp": datetime.now()},
    ])
''')
```

## Critical
- All recall queries MUST filter by `user_id` -- otherwise users see each other's data.
- Use `pxt.Required[pxt.String]` (or `"Required[String]"`) for `user_id`.
- Match `invoke_tools` import to the LLM provider.
- Chat-history insertion is NOT automatic via computed columns; do it explicitly after each turn.
"""


# =============================================================================
# Video RAG Agent Pipeline (from pixeltable-skill video-rag-agents.md)
# =============================================================================

VIDEO_RAG_AGENT_PROMPT = """\
Guide the user through a video RAG agent: ingest videos, index frames + transcripts, expose
search as `@pxt.query` tools, and let an LLM choose between them.

## Step 1: Videos table and frame view

```python
pixeltable_create_table("media.videos", schema={
    "video": "Video",
    "title": "String",
}, if_exists="ignore")

# frame_iterator is from pixeltable.functions.video (NOT pixeltable.iterators).
pixeltable_create_view(
    path="media.frames",
    base_table_path="media.videos",
    iterator="frame_iterator",
    iterator_kwargs={"video": "table.video", "fps": 1.0},
    if_exists="ignore",
)
```

## Step 2: Visual index + frame descriptions

```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.huggingface import clip
from pixeltable.functions.openai import chat_completions

frames = pxt.get_table("media.frames")
clip_fn = clip.using(model_id="openai/clip-vit-base-patch32")
frames.add_embedding_index("frame", embedding=clip_fn, if_exists="ignore")

# Use chat_completions with image_url content blocks -- there is no openai.vision.
frames.add_computed_column(
    description=chat_completions(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this frame in one sentence."},
            {"type": "image_url", "image_url": {"url": frames.frame}},
        ]}],
        model="gpt-4o-mini",
    ).choices[0].message.content,
    if_exists="ignore",
)
''')
```

## Step 3: Audio extraction + transcript view + sentence index

```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.video import extract_audio
from pixeltable.functions.openai import transcriptions
from pixeltable.functions.huggingface import sentence_transformer

videos = pxt.get_table("media.videos")
videos.add_computed_column(audio=extract_audio(videos.video, format="mp3"), if_exists="ignore")
videos.add_computed_column(
    transcript=transcriptions(audio=videos.audio, model="whisper-1").text,
    if_exists="ignore",
)
''')
```

Then chunk transcripts into sentences with a view that uses
`iterator="string_splitter"` and `iterator_kwargs={"text": "table.transcript", "separators": "sentence"}`,
and index the resulting `text` column with the sentence-transformer model.

## Step 4: @pxt.query tools

```python
execute_python('''
import pixeltable as pxt

frames = pxt.get_table("media.frames")
sentences = pxt.get_table("media.transcript_chunks")

@pxt.query
def search_video_frames(query: str, limit: int = 5):
    sim = frames.frame.similarity(string=query)
    return frames.order_by(sim, asc=False).limit(limit).select(frames.title, frames.description, sim)

@pxt.query
def search_transcripts(query: str, limit: int = 5):
    sim = sentences.text.similarity(string=query)
    return sentences.order_by(sim, asc=False).limit(limit).select(sentences.title, sentences.text, sim)
''')
```

## Step 5: Bind tools and build the agent chain

```python
pixeltable_create_tools(
    function_names=["search_video_frames", "search_transcripts"],
    register_as="tools",
)
```

Then follow the `tool_calling_agent_pipeline` prompt to attach an LLM that picks between
the visual and transcript searches, runs them via `invoke_tools`, and assembles a final answer.

## Anti-patterns
- NO `cv2.VideoCapture` frame loops -- use the `frame_iterator` view.
- NO openai.vision -- use `openai.chat_completions` with `image_url` content blocks.
- Always pass `string=` to `.similarity(...)` and chain `order_by` + `limit`.
"""


# =============================================================================
# Agentic Patterns Guide (from pixeltable-skill agentic-patterns.md)
# =============================================================================

AGENTIC_PATTERNS_PROMPT = """\
Reference card for the six agentic patterns from pixeltable-skill. Each pattern is one
Pixeltable table whose computed columns express the orchestration. No while-loops,
no async glue, no separate orchestration framework.

## 1. Prompt Chaining
Sequential computed columns where each step depends on the previous one.
table -> draft (LLM) -> critique (LLM) -> revision (LLM) -> answer.

## 2. Routing
A first LLM column emits a `route` string; downstream columns conditional on the route are
built either as multiple computed columns or as one `@pxt.udf` that switches on `route`.

## 3. Parallelization
Multiple computed columns derived from the same input run in parallel (Pixeltable executes
them concurrently). Final column merges their outputs via an `@pxt.udf`.

## 4. Tool Use (production agent)
See the `tool_calling_agent_pipeline` prompt: `messages(... tools=pxt.tools(...))` +
`invoke_tools(tools, response)` columns plus RAG retrieval columns.

## 5. Evaluator-Optimizer
Two LLM columns: `candidate = generator(prompt)` and `score = evaluator(prompt, candidate)`.
Loop is replaced by re-inserting rows with `attempt += 1` until `score` clears a threshold;
keep the iteration count outside the table.

## 6. Orchestrator-Worker
Wrap a "worker" table as a sub-agent via `pxt.udf(table, return_value=table.answer)`.
The orchestrator table calls that UDF in its computed columns; each worker call materializes
a new row in the worker table.

## Reasoning strategies
- ReAct: alternate `thought` (LLM) and `action` (UDF / tool) columns; terminate when
  `action == "final_answer"`.
- Planning: one column generates a JSON plan, another iterates the plan via a UDF that
  appends results.

## Cross-cutting rules
- Anti-patterns the patterns above explicitly avoid: while-loops, framework state machines,
  pandas as the working store, separate vector DBs, hardcoded API keys.
- Use `@pxt.query` for retrieval logic, `@pxt.udf` for transformations / business logic,
  computed columns to wire them together.
- Make outputs deterministic by snapshotting input tables before evaluation runs.

For a concrete production example combining patterns 4 + 5, see the
`tool_calling_agent_pipeline` and `video_rag_agent_pipeline` prompts.
"""


# =============================================================================
# ML Data Pipeline (from pixeltable-skill ml-data-pipeline.md)
# =============================================================================

ML_DATA_PIPELINE_PROMPT = """\
Guide the user through ingest -> enrich -> curate -> snapshot -> export for ML training.
Pixeltable replaces ad-hoc pandas pipelines: every step is a column or view, snapshots
freeze a dataset version, and exports go straight to PyTorch / Parquet / pandas.

## Step 1: Ingest

```python
pixeltable_create_table("ml.raw", schema={
    "image": "Image",
    "source": "String",
    "label": "String",
    "timestamp": "Timestamp",
}, if_exists="ignore")

# Or import directly from a file:
# pixeltable_create_table("ml.raw", source="s3://bucket/data.parquet")
```

## Step 2: Enrich with computed columns (run on insert)

```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions import image as pxt_image
from pixeltable.functions.huggingface import detr_for_object_detection, clip
from pixeltable.functions.openai import chat_completions

t = pxt.get_table("ml.raw")

t.add_computed_column(thumbnail=pxt_image.thumbnail(t.image, size=(224, 224)), if_exists="ignore")

t.add_computed_column(
    detections=detr_for_object_detection(t.image, model_id="facebook/detr-resnet-50"),
    if_exists="ignore",
)

t.add_computed_column(
    caption=chat_completions(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "One-sentence caption."},
            {"type": "image_url", "image_url": {"url": t.image}},
        ]}],
        model="gpt-4o-mini",
    ).choices[0].message.content,
    if_exists="ignore",
)

clip_fn = clip.using(model_id="openai/clip-vit-base-patch32")
t.add_embedding_index("image", embedding=clip_fn, if_exists="ignore")
''')
```

## Step 3: Curate (filter, dedup, fix errors)

```python
execute_python('''
import pixeltable as pxt

t = pxt.get_table("ml.raw")

# Inspect failed computed columns.
errs = t.where(t.caption.errortype != None).select(t.source, t.caption.errormsg).collect()
print(list(errs))

# Retry just the failures.
t.recompute_columns(columns=["caption"], where=t.caption.errortype != None)
''')
```

## Step 4: Snapshot a dataset version

```python
pixeltable_create_snapshot(
    path="ml.train_v1",
    base_table_path="ml.raw",
    if_exists="ignore",
)
```

## Step 5: Export for training

```python
execute_python('''
import pixeltable as pxt

snap = pxt.get_table("ml.train_v1")

# PyTorch Dataset
torch_ds = snap.to_pytorch_dataset(
    image_format="np",
)

# Parquet
from pixeltable.io import export_parquet
export_parquet(snap, "out/train_v1/")

# pandas (analysis only -- not a working store)
df = snap.select(snap.label, snap.caption).collect().to_pandas()
print(df.head())
''')
```

## Step 6: Retrieval UDFs for structured lookup

For lookup-style joins (e.g. label -> taxonomy), define a `@pxt.udf` that calls
`other_table.where(...).select(...).collect()` and use it in a computed column. The skill
calls these "retrieval UDFs".

## Anti-patterns
- NO pandas as a working store. Materialize tables, snapshot, then `.to_pandas()` at the end.
- NO loops calling AI models per row -- use a computed column.
- NO ad-hoc JSON files for dataset versions -- snapshots are the version control story.
"""
