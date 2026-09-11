Build a deterministic audio processing application in `app.py` using `fixtures/audio.wav`.

Declare a `TableModel` table named `recordings` with integer primary key `id` and a `pxt.Audio` column `audio`. Declare an `audio_segments` iterator view with the current `pixeltable.functions.audio.audio_splitter` API and a segment duration of 0.25 seconds. Do not use a deprecated iterator import.

Initialize the project, check and apply the schema to `eval_audio`, insert the one-second fixture, and prove that it produces four segments whose last `segment_end` is 1.0. Report executable evidence.
