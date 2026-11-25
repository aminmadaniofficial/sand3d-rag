"""End-to-end stdio test for the Sand3D RAG MCP server & MCP App extension."""

import asyncio
import json
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.apps import APP_MIME_TYPE

from config import PROJECT_DIR
from mcp_server import EXPLORER_URI, _render_thumbnail_data_uri


class TestMCPServer(unittest.TestCase):
    def test_thumbnail_data_uri_generation(self) -> None:
        uri = _render_thumbnail_data_uri(0)
        self.assertIsNotNone(uri)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_stdio_handshake_apps_resource_and_search(self) -> None:
        async def _run() -> None:
            params = StdioServerParameters(
                command=str(PROJECT_DIR / ".venv" / "bin" / "python"),
                args=[str(PROJECT_DIR / "mcp_server.py")],
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools_res = await session.list_tools()
                    by_name = {t.name: t for t in tools_res.tools}
                    self.assertTrue(
                        {
                            "open_sand3d_explorer",
                            "search_sand3d_assets",
                            "get_sand3d_api_reference",
                            "get_index_stats",
                            "generate_sand3d_program",
                            "app_get_asset_details",
                        }.issubset(by_name.keys())
                    )
                    search_meta = by_name["search_sand3d_assets"].meta or {}
                    self.assertEqual(
                        search_meta.get("ui", {}).get("resourceUri"), EXPLORER_URI
                    )
                    self.assertEqual(
                        search_meta.get("ui", {}).get("visibility"), ["model", "app"]
                    )
                    self.assertEqual(
                        search_meta.get("com.google.antigravity/ui", {})
                        .get("entrypoints", [{}])[0]
                        .get("type"),
                        "aux_pane",
                    )

                    open_meta = by_name["open_sand3d_explorer"].meta or {}
                    ep_types = {
                        ep.get("type")
                        for ep in open_meta.get("com.google.antigravity/ui", {}).get(
                            "entrypoints", []
                        )
                    }
                    self.assertEqual(ep_types, {"aux_pane", "global"})

                    app_only_meta = by_name["app_get_asset_details"].meta or {}
                    self.assertEqual(
                        app_only_meta.get("ui", {}).get("visibility"), ["app"]
                    )

                    resources_res = await session.list_resources()
                    uris = {str(r.uri) for r in resources_res.resources}
                    self.assertIn(EXPLORER_URI, uris)

                    ui_res = await session.read_resource(EXPLORER_URI)
                    self.assertEqual(ui_res.contents[0].mime_type, APP_MIME_TYPE)
                    self.assertIn("SAND3D-RAG", ui_res.contents[0].text)

                    # Entrypoint launch with empty '{}' must return default assets quickly
                    open_res = await session.call_tool("open_sand3d_explorer", {})
                    self.assertFalse(open_res.is_error)
                    self.assertGreaterEqual(
                        len(open_res.structured_content["results"]), 1
                    )

                    stats_res = await session.call_tool("get_index_stats", {})
                    stats = json.loads(stats_res.content[0].text)
                    self.assertEqual(stats["points_count"], 10057)
                    self.assertEqual(stats["mcp_app_uri"], EXPLORER_URI)

                    search_res = await session.call_tool(
                        "search_sand3d_assets",
                        {
                            "query": "قارچ جنگلی روی صخره",
                            "top_k": 2,
                            "include_program": False,
                        },
                    )
                    self.assertFalse(search_res.is_error)
                    hits = json.loads(search_res.content[0].text)
                    self.assertEqual(len(hits), 2)
                    self.assertIn("prompt", hits[0])
                    self.assertIsNotNone(search_res.structured_content)
                    self.assertEqual(len(search_res.structured_content["results"]), 2)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
