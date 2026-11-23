import sys
from typing import Any, Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.config import settings


class MCPToolManager:
    """
    Connects to the MCP server over stdio, dynamically discovers tools,
    and executes tool calls cleanly per request or session.
    """

    def __init__(self):
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[settings.mcp_server_script],
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_resp.tools
                ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                texts = [
                    block.text for block in result.content if hasattr(block, "text")
                ]
                return "\n".join(texts)


mcp_manager = MCPToolManager()
