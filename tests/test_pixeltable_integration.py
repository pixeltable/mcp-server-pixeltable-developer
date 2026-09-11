"""End-to-end Pixeltable 0.7.6 application, service, and recovery tests."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import textwrap
import urllib.request
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from mcp import Client
from mcp.types import TextContent
from PIL import Image

from mcp_server_pixeltable_developer.runtime import ServerConfig
from mcp_server_pixeltable_developer.server import create_server

pytestmark = pytest.mark.slow


async def _call(client: Client, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await client.call_tool(name, arguments or {})
    assert result.is_error is False, result.content
    assert isinstance(result.structured_content, dict)
    return result.structured_content


def _first_text(result: Any) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 200
        value = json.loads(response.read())
    assert isinstance(value, dict)
    return value


def _stop_service(config: ServerConfig, name: str) -> None:
    subprocess.run(
        [config.pxt_executable, "service", "stop", name, "--json"],
        cwd=config.project_root,
        env=config.command_environment(),
        capture_output=True,
        check=False,
        timeout=15,
    )


@pytest.mark.asyncio
async def test_scaffold_schema_data_and_http_service_lifecycle(server_config: ServerConfig) -> None:
    service_started = False
    try:
        async with Client(create_server(server_config), raise_exceptions=False) as client:
            scaffold = await _call(client, "pixeltable_scaffold_app")
            assert scaffold["kind"] == "service"
            assert (server_config.project_root / "app.py").is_file()

            duplicate = await client.call_tool("pixeltable_scaffold_app")
            assert duplicate.is_error is True
            assert "overwrite" in _first_text(duplicate)

            schema_check = await _call(client, "pixeltable_schema_check", {"schema_file": "app.py"})
            assert schema_check["valid"] is True
            schema_diff = await _call(
                client,
                "pixeltable_schema_diff",
                {"schema_file": "app.py", "target": "trial"},
            )
            assert schema_diff["pending"] is True
            assert schema_diff["details"]["summary"]["create"] == 1
            await _call(
                client,
                "pixeltable_schema_update",
                {"schema_file": "app.py", "target": "trial"},
            )

            catalog = await _call(client, "pixeltable_list_catalog")
            assert catalog["entries"][0]["path"] == "trial"
            description = await _call(client, "pixeltable_describe", {"path": "trial/docs"})
            assert description["has_default_idxs"] is False
            assert description["columns"]["title_upper"]["is_computed"] is True

            insertion = await _call(
                client,
                "pixeltable_insert_rows",
                {
                    "path": "trial/docs",
                    "rows": [{"doc_id": 1, "title": "hello", "body": None}],
                },
            )
            assert insertion["num_rows"] == 1
            rows = await _call(client, "pixeltable_rows", {"path": "trial/docs", "limit": 5})
            assert rows["rows"] == [
                {
                    "doc_id": 1,
                    "title": "hello",
                    "body": None,
                    "title_upper": "HELLO",
                    "summary": "hello",
                }
            ]

            service_check = await _call(client, "pixeltable_service_check", {"app_file": "app.py"})
            assert service_check["valid"] is True
            service_diff = await _call(
                client,
                "pixeltable_service_diff",
                {"app_file": "app.py", "target": "trial"},
            )
            assert service_diff["pending"] is True
            dry_update = await _call(
                client,
                "pixeltable_service_update",
                {"app_file": "app.py", "target": "trial", "dry_run": True},
            )
            assert dry_update["pending"] is True
            await _call(
                client,
                "pixeltable_service_update",
                {"app_file": "app.py", "target": "trial"},
            )
            service_started = True
            services = await _call(client, "pixeltable_service_list")
            assert len(services["services"]) == 1
            endpoint = services["services"][0]["endpoint"]
            port = services["services"][0]["port"]
            response = await asyncio.to_thread(
                _post_json,
                f"{endpoint}/docs",
                {"doc_id": 2, "title": "served", "body": "body"},
            )
            assert response == {"title_upper": "SERVED", "summary": "served"}

            await _call(
                client,
                "pixeltable_service_update",
                {"app_file": "app.py", "target": "trial"},
            )
            services_after = await _call(client, "pixeltable_service_list")
            assert services_after["services"][0]["port"] == port

            stopped = await _call(client, "pixeltable_service_stop", {"names": ["trial/ingest"]})
            service_started = False
            assert stopped["names"] == ["trial/ingest"]
            assert (await _call(client, "pixeltable_service_list"))["services"] == []
            pruned = await _call(
                client,
                "pixeltable_service_prune",
                {"app_file": "app.py", "target": "trial"},
            )
            assert pruned["pending"] is False
    finally:
        if service_started:
            _stop_service(server_config, "trial/ingest")


RECOVERY_APP = """
from pathlib import Path

import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


@pxt.udf
def recover(value: str) -> str:
    if not Path(__file__).with_name('recovery.ready').exists():
        raise ValueError('fixture not ready')
    return value.upper()


class Items(TableModel, name='items'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    value: pxt.String
    recovered = recover(value)
"""


@pytest.mark.asyncio
async def test_errors_only_recovery_and_schema_evolution(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=False) as client:
        await _call(
            client,
            "pixeltable_scaffold_app",
            {"kind": "schema", "output_path": "app.py"},
        )
        app_file = server_config.project_root / "app.py"
        app_file.write_text(textwrap.dedent(RECOVERY_APP).lstrip())
        assert (await _call(client, "pixeltable_schema_check", {"schema_file": "app.py"}))["valid"] is True
        await _call(
            client,
            "pixeltable_schema_update",
            {"schema_file": "app.py", "target": "recovery"},
        )

        insertion = await _call(
            client,
            "pixeltable_insert_rows",
            {
                "path": "recovery/items",
                "rows": [{"id": 1, "value": "one"}, {"id": 2, "value": "two"}],
                "on_error": "ignore",
            },
        )
        assert insertion["num_exceptions"] == 2
        errors = await _call(
            client,
            "pixeltable_errors",
            {"path": "recovery/items", "column": "recovered"},
        )
        assert errors["count"] == 2
        assert {entry["pk"]["id"] for entry in errors["errors"]} == {1, 2}
        row = await _call(
            client,
            "pixeltable_get_row",
            {"path": "recovery/items", "primary_key": ["2"]},
        )
        assert row["row"] == {"id": 2, "value": "two", "recovered": None}

        invalid_recompute = await client.call_tool(
            "pixeltable_recompute",
            {"path": "recovery/items", "columns": ["recovered", "value"]},
        )
        assert invalid_recompute.is_error is True
        assert "exactly one" in _first_text(invalid_recompute)
        preview = await _call(
            client,
            "pixeltable_recompute",
            {"path": "recovery/items", "columns": ["recovered"]},
        )
        assert preview["dry_run"] is True
        assert preview["errors_only"] is True
        assert preview["pending"] is True

        (server_config.project_root / "recovery.ready").touch()
        applied = await _call(
            client,
            "pixeltable_recompute",
            {
                "path": "recovery/items",
                "columns": ["recovered"],
                "dry_run": False,
            },
        )
        assert applied["pending"] is False
        assert (await _call(client, "pixeltable_errors", {"path": "recovery/items"}))["count"] == 0
        recovered = await _call(
            client,
            "pixeltable_get_row",
            {"path": "recovery/items", "primary_key": ["2"]},
        )
        assert recovered["row"]["recovered"] == "TWO"

        original = app_file.read_text()
        app_file.write_text(
            original.replace("    value: pxt.String\n", "    value: pxt.String\n    note: pxt.String | None\n")
        )
        additive = await _call(
            client,
            "pixeltable_schema_diff",
            {"schema_file": "app.py", "target": "recovery"},
        )
        assert additive["details"]["summary"]["update_additive"] == 1
        await _call(
            client,
            "pixeltable_schema_update",
            {"schema_file": "app.py", "target": "recovery"},
        )

        app_file.write_text(original)
        destructive = await _call(
            client,
            "pixeltable_schema_diff",
            {"schema_file": "app.py", "target": "recovery"},
        )
        assert destructive["details"]["summary"]["update_destructive"] == 1
        refused = await client.call_tool(
            "pixeltable_schema_update",
            {"schema_file": "app.py", "target": "recovery"},
        )
        assert refused.is_error is True
        await _call(
            client,
            "pixeltable_schema_update",
            {
                "schema_file": "app.py",
                "target": "recovery",
                "allow_destructive": True,
            },
        )

        changed_expression = original.replace("recovered = recover(value)", "recovered = pxtf.string.lower(value)")
        app_file.write_text(changed_expression)
        unsupported = await _call(
            client,
            "pixeltable_schema_diff",
            {"schema_file": "app.py", "target": "recovery"},
        )
        assert unsupported["details"]["summary"]["unsupported"] == 1
        unsupported_update = await client.call_tool(
            "pixeltable_schema_update",
            {
                "schema_file": "app.py",
                "target": "recovery",
                "allow_destructive": True,
            },
        )
        assert unsupported_update.is_error is True
        assert "cannot be updated" in _first_text(unsupported_update).lower()


MULTIMODAL_APP = """
import numpy as np
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.openai import invoke_tools
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf
def add_numbers(a: int, b: int) -> int:
    return a + b


@pxt.udf
def mock_model(question: str) -> dict:
    return {
        'choices': [
            {
                'message': {
                    'tool_calls': [
                        {'function': {'name': 'add_numbers', 'arguments': '{"a": 2, "b": 3}'}}
                    ]
                }
            }
        ]
    }


@pxt.udf
def deterministic_embed(text: str) -> pxt.Array[(3,), pxt.Float]:
    lowered = text.lower()
    return np.array(
        [float(lowered.count('alpha')), float(lowered.count('beta')), 1.0],
        dtype=np.float32,
    )


agent_tools = pxt.tools(add_numbers)


class Media(TableModel, name='media'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    document: pxt.Document
    image: pxt.Image
    video: pxt.Video
    audio: pxt.Audio
    rotated = image.rotate(90)


class Chunks(
    TableModel,
    name='chunks',
    base=Media,
    iterator=pxtf.document.document_splitter(Media.document, separators='paragraph'),
):
    __indexes__ = [
        pxt.EmbeddingIndex(text, embedding=deterministic_embed.using(), name='chunks_embed')
    ]


@pxt.query
def search_chunks(query_text: str):
    similarity = Chunks.text.similarity(string=query_text)
    return Chunks.order_by(similarity, asc=False).select(
        text=Chunks.text,
        score=similarity,
    ).limit(2)


search = FastAPIRouter(name='search', prefix='/api')
search.add_query_route(path='/search', query=search_chunks, method='post')


class Frames(
    TableModel,
    name='frames',
    base=Media,
    iterator=pxtf.video.frame_iterator(Media.video, fps=2.0),
):
    pass


class AudioSegments(
    TableModel,
    name='audio_segments',
    base=Media,
    iterator=pxtf.audio.audio_splitter(Media.audio, duration=0.25),
):
    pass


class AgentRuns(TableModel, name='agent_runs'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    question: pxt.String
    response = mock_model(question)
    tool_results = invoke_tools(agent_tools, response)
"""


def _write_multimodal_fixtures(project_root: Path) -> dict[str, str]:
    document = project_root / "document.html"
    document.write_text("<html><body><p>alpha retrieval paragraph.</p><p>beta second paragraph.</p></body></html>")
    image = project_root / "image.png"
    Image.new("RGB", (32, 20), (255, 0, 0)).save(image)

    audio = project_root / "audio.wav"
    samples = [int(12_000 * math.sin(2 * math.pi * 440 * index / 8_000)) for index in range(8_000)]
    with wave.open(str(audio), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(np.array(samples, dtype="<i2").tobytes())

    video = project_root / "video.mp4"
    container = av.open(str(video), "w")
    stream = container.add_stream("mpeg4", rate=2)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for index in range(4):
        pixels = np.full((24, 32, 3), (index * 60, 0, 255 - index * 60), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return {
        "document": str(document),
        "image": str(image),
        "video": str(video),
        "audio": str(audio),
    }


@pytest.mark.asyncio
async def test_deterministic_multimodal_views_and_mocked_tool_calling(server_config: ServerConfig) -> None:
    service_started = False
    try:
        async with Client(create_server(server_config), raise_exceptions=False) as client:
            await _call(
                client,
                "pixeltable_scaffold_app",
                {"kind": "schema", "output_path": "app.py"},
            )
            (server_config.project_root / "app.py").write_text(textwrap.dedent(MULTIMODAL_APP).lstrip())
            check = await _call(client, "pixeltable_schema_check", {"schema_file": "app.py"})
            assert check["valid"] is True
            update = await _call(
                client,
                "pixeltable_schema_update",
                {"schema_file": "app.py", "target": "multi"},
            )
            assert update["details"]["summary"]["create"] == 5

            fixtures = _write_multimodal_fixtures(server_config.project_root)
            inserted = await _call(
                client,
                "pixeltable_insert_rows",
                {"path": "multi/media", "rows": [{"id": 1, **fixtures}]},
            )
            assert inserted["num_exceptions"] == 0
            media = await _call(client, "pixeltable_rows", {"path": "multi/media"})
            assert media["count"] == 1
            assert media["rows"][0]["rotated"].startswith("<Image ")

            chunks = await _call(
                client,
                "pixeltable_rows",
                {"path": "multi/chunks", "columns": ["text"], "limit": 10},
            )
            assert [row["text"] for row in chunks["rows"]] == [
                "alpha retrieval paragraph.",
                "beta second paragraph.",
            ]

            await _call(
                client,
                "pixeltable_service_update",
                {"app_file": "app.py", "target": "multi"},
            )
            service_started = True
            services = await _call(client, "pixeltable_service_list")
            endpoint = services["services"][0]["endpoint"]
            retrieval = await asyncio.to_thread(
                _post_json,
                f"{endpoint}/api/search",
                {"query_text": "alpha"},
            )
            assert retrieval["rows"][0]["text"] == "alpha retrieval paragraph."
            assert retrieval["rows"][0]["score"] > retrieval["rows"][1]["score"]

            frames = await _call(
                client,
                "pixeltable_rows",
                {"path": "multi/frames", "columns": ["pos", "frame_attrs"], "limit": 10},
            )
            assert frames["count"] >= 3
            assert frames["rows"][0]["frame_attrs"]["time"] == 0.0
            audio_segments = await _call(
                client,
                "pixeltable_rows",
                {
                    "path": "multi/audio_segments",
                    "columns": ["pos", "segment_start", "segment_end"],
                    "limit": 10,
                },
            )
            assert audio_segments["count"] == 4
            assert audio_segments["rows"][-1]["segment_end"] == 1.0

            agent_insert = await _call(
                client,
                "pixeltable_insert_rows",
                {"path": "multi/agent_runs", "rows": [{"id": 1, "question": "add two and three"}]},
            )
            assert agent_insert["num_exceptions"] == 0
            agent_rows = await _call(client, "pixeltable_rows", {"path": "multi/agent_runs"})
            assert agent_rows["rows"][0]["tool_results"] == {"add_numbers": [5]}
    finally:
        if service_started:
            _stop_service(server_config, "multi/search")
