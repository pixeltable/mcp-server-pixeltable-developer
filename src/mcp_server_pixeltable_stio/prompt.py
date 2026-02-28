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

Key points to remember:

1. **Pixeltable is NOT just a database** - it's multimodal AI data infrastructure designed for AI workflows with images, videos, audio, documents, and structured data.

2. **Start with the basics**: Always help users create tables first, then add data, then apply AI functions.

3. **Think in workflows**: Pixeltable excels at data pipelines where you:
   - Store multimodal data
   - Apply AI models (computer vision, NLP, speech recognition)
   - Query and analyze results

4. **Common patterns**:
   - Image analysis: Create table -> Add images -> Apply object detection/classification -> Query results
   - Document processing: Create table -> Add PDFs -> Extract/chunk text -> Generate embeddings -> Search
   - Video workflows: Create table -> Add videos -> Extract frames -> Apply vision models -> Transcribe audio

5. **Built-in AI capabilities**:
   - YOLOX object detection
   - Whisper speech recognition
   - CLIP embeddings
   - Ollama integration
   - HuggingFace models
   - Custom model integration

6. **MCP Resources**: Read-only data is exposed via Resources (pixeltable://tables, pixeltable://version, etc.) -- use these instead of tool calls for listing/inspecting.

7. **Be helpful with complex concepts**: Pixeltable can be complex. Break down workflows into simple steps and explain what each step accomplishes.

8. **Suggest practical examples**: When users seem unsure, suggest concrete examples like "Let's start by creating a table for your photos and running object detection."

9. **Use available tools**: Make use of the introspection, REPL, and bug logging tools when users need help exploring or debugging.

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
pixeltable_smart_install("yolox")   # For object detection
pixeltable_smart_install("openai")  # For GPT-4 Vision
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
pixeltable_smart_install("openai")
pixeltable_smart_install("sentence-transformers")
```

## Step 3: Ingest Documents
```python
pixeltable_insert_data("rag.documents", [
    {"document": "/path/to/paper.pdf", "title": "Research Paper", "category": "academic"},
    {"document": "/path/to/manual.pdf", "title": "User Manual", "category": "technical"}
])
```

## Step 4: Create a Chunks View
Use Pixeltable's document chunking to split documents into searchable pieces:
```python
# Use the REPL for more complex operations
execute_python('''
import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter

docs = pxt.get_table("rag.documents")
chunks = pxt.create_view(
    "rag.chunks",
    docs,
    iterator=DocumentSplitter.create(
        document=docs.document,
        separators="sentence",
        metadata="title,page"
    )
)
''')
```

## Step 5: Add Embeddings
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

chunks = pxt.get_table("rag.chunks")
chunks.add_embedding_index("text", string_embed=sentence_transformer.using(model_id="all-MiniLM-L6-v2"))
''')
```

## Step 6: Search
```python
execute_python('''
import pixeltable as pxt
chunks = pxt.get_table("rag.chunks")
results = chunks.select(chunks.text, chunks.title).similarity("text", "How does X work?").limit(5).collect()
print(results)
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
- **Automatic Chunking**: DocumentSplitter handles PDF/text splitting
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
Create a view that extracts frames at regular intervals:
```python
execute_python('''
import pixeltable as pxt
from pixeltable.iterators import FrameIterator

clips = pxt.get_table("video.clips")
frames = pxt.create_view(
    "video.frames",
    clips,
    iterator=FrameIterator.create(
        video=clips.video,
        fps=1  # 1 frame per second
    )
)
''')
```

## Step 4: Analyze Frames
Add AI models to process each extracted frame:

### Object Detection
```python
execute_python('''
import pixeltable as pxt
frames = pxt.get_table("video.frames")
frames.add_computed_column(detections=yolox.yolox(frames.frame, model_id="yolox_m"))
''')
```

### Scene Description
```python
execute_python('''
import pixeltable as pxt
from pixeltable.functions import openai

frames = pxt.get_table("video.frames")
frames.add_computed_column(scene_description=openai.chat_completions(
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Briefly describe this video frame"},
        {"type": "image_url", "image_url": {"url": frames.frame}}
    ]}],
    model="gpt-4o-mini"
))
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
- **Frame Extraction**: FrameIterator creates a row per frame automatically
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
pixeltable_smart_install("whisper")
pixeltable_smart_install("openai")
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
    string_embed=sentence_transformer.using(model_id="all-MiniLM-L6-v2")
)

# Search
results = recordings.select(
    recordings.title, recordings.transcription
).similarity("transcription", "pricing discussion").limit(5).collect()
print(results)
''')
```

## Key Concepts
- **Automatic Transcription**: Computed columns transcribe on insert
- **Multi-Model**: Use Whisper for transcription, GPT for analysis, embeddings for search
- **Incremental**: New audio files are automatically processed
- **Searchable**: Embedding indexes enable semantic search over spoken content
"""
