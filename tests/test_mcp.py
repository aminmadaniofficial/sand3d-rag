"""End-to-end stdio test for the Sand3D RAG MCP server."""

import asyncio
import json
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import PROJECT_DIR


class TestMCPServer(unittest.TestCase):
    def test_stdio_handshake_list_tools_and_search(self) -> None:
        async def _run() -> None:
            params = StdioServerParameters(
                command=str(PROJECT_DIR / ".venv" / "bin" / "python"),
                args=[str(PROJECT_DIR / "mcp_server.py")],
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_res = await session.list_tools()
                    names = {t.name for t in tools_res.tools}
                    self.assertTrue(
                        {
                            "search_sand3d_assets",
                            "get_sand3d_api_reference",
                            "get_index_stats",
                            "generate_sand3d_program",
                        }.issubset(names)
                    )

                    stats_res = await session.call_tool("get_index_stats", {})
                    stats_text = stats_res.content[0].text
                    stats = json.loads(stats_text)
                    self.assertEqual(stats["points_count"], 10057)

                    search_res = await session.call_tool(
                        "search_sand3d_assets",
                        {
                            "query": "قارچ جنگلی روی صخره",
                            "top_k": 2,
                            "include_program": False,
                        },
                    )
                    self.assertFalse(search_res.is_error)
                    self.assertGreater(len(search_res.content), 0)
                    hits = json.loads(search_res.content[0].text)
                    if isinstance(hits, dict):
                        hits = [hits]
                    self.assertGreaterEqual(len(hits), 1)
                    self.assertIn("prompt", hits[0])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
