Build a deterministic image-and-video processing application in `app.py` using `fixtures/image.png` and `fixtures/video.mp4`.

Declare a `TableModel` table named `media` with integer primary key `id`, `pxt.Image` column `image`, and `pxt.Video` column `video`. Add a stored computed image column `rotated` that rotates the image 90 degrees. Declare a `frames` iterator view using the current `pixeltable.functions.video.frame_iterator` API at 2 frames per second. Do not use a deprecated iterator import.

Initialize the project, check and apply the schema to `eval_media`, insert both fixtures, and prove that the rotated image and at least three frame rows are available. Report executable evidence.
