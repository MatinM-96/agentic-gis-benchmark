import logging
import sys

RESET = "\033[0m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"



class CleanFormatter(logging.Formatter):
    def format(self, record):
        name = record.name
        msg = record.getMessage()

        if name == "__main__":
            if "Starting MCP server" in msg:
                return f"      [SYSTEM] starting MCP server"
            if "Server ready, yielding app context" in msg:
                return f"     [SYSTEM] server ready"
            return f"{msg}"

        if name == "backend.database.database":
            if "Pool created for conn_id=" in msg:
                conn_id = msg.split("conn_id=")[-1]
                return f"{GREEN}      [DB]{RESET} pool created : {conn_id}"
            return f"{GREEN}      [DB]{RESET} {msg}"

        if name == "backend.mcp.client.client":
            return f"{CYAN}      [MCP CLIENT]{RESET} {msg}"

        if name.startswith("backend.mcp.server"):
            return f"{BLUE}      [MCP SERVER]{RESET} {msg}"

        if name == "mcp.server.lowlevel.server":
            if "Processing request of type CallToolRequest" in msg:
                return f"{BLUE}      [MCP]{RESET} executing tool request"
            if "Processing request of type ListToolsRequest" in msg:
                return f"{BLUE}      [MCP]{RESET} listing tools"
            return f"{BLUE}      [MCP]{RESET} {msg}"

        if name == "httpx":
            if "HTTP Request:" in msg:
                return f"{YELLOW}      [HTTPX]{RESET} request sent ok"
            return f"{YELLOW}      [HTTPX]{RESET} {msg}"

        if name == "openai._base_client":
            if "Request options:" in msg:
                return f"{MAGENTA}      [OPENAI]{RESET} request options prepared"
            if "Sending HTTP Request:" in msg:
                return f"{MAGENTA}      [OPENAI]{RESET} sending request"
            if "HTTP Response:" in msg:
                return f"{MAGENTA}      [OPENAI]{RESET} response received"
            if "request_id:" in msg:
                return f"{MAGENTA}      [OPENAI]{RESET} {msg}"
            return f"{MAGENTA}      [OPENAI]{RESET} {msg}"

        if name == "httpcore.connection":
            if "connect_tcp.started" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} tcp connect started"
            if "connect_tcp.complete" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} tcp connect complete"
            if "start_tls.started" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} tls start"
            if "start_tls.complete" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} tls ready"
            if "close.started" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} close started"
            if "close.complete" in msg:
                return f"{GRAY}      [HTTPCORE]{RESET} close complete"
            return f"{GRAY}      [HTTPCORE]{RESET} {msg}"

        if name == "httpcore.http11":
            if "send_request_headers.started" in msg:
                return f"{DIM}      [HTTP11]{RESET} sending headers"
            if "send_request_headers.complete" in msg:
                return f"{DIM}      [HTTP11]{RESET} headers sent"
            if "send_request_body.started" in msg:
                return f"{DIM}      [HTTP11]{RESET} sending body"
            if "send_request_body.complete" in msg:
                return f"{DIM}      [HTTP11]{RESET} body sent"
            if "receive_response_headers.started" in msg:
                return f"{DIM}      [HTTP11]{RESET} waiting for headers"
            if "receive_response_headers.complete" in msg:
                return f"{DIM}      [HTTP11]{RESET} headers received"
            if "receive_response_body.started" in msg:
                return f"{DIM}      [HTTP11]{RESET} receiving body"
            if "receive_response_body.complete" in msg:
                return f"{DIM}      [HTTP11]{RESET} body received"
            if "response_closed.started" in msg:
                return f"{DIM}      [HTTP11]{RESET} closing response"
            if "response_closed.complete" in msg:
                return f"{DIM}      [HTTP11]{RESET} response closed"
            return f"{DIM}      [HTTP11]{RESET} {msg}"
        
        
        
        if name == "backend.assistant_runtime":
            if "✓" in msg:
                return f"{GREEN}  [AGENT]{RESET} {msg}"
            if "✗" in msg or "error" in msg.lower() or "failed" in msg.lower():
                return f"{RED}  [AGENT]{RESET} {msg}"
            return f"{CYAN}  [AGENT]{RESET} {msg}"

        if name == "reporting":
            return f"      {msg}"

        level_color = {
            logging.DEBUG: GRAY,
            logging.INFO: RESET,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: RED,
        }.get(record.levelno, RESET)

        return f"{level_color}      [{name}]{RESET} {msg}"


def setup_logging() -> None:
    logging.getLogger("azure").setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(CleanFormatter())



    root = logging.getLogger()
    root.handlers.clear()
    root.filters.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)





    for name in [
        "__main__",
        "backend.assistant_runtime",
        "backend.database.database",
        "backend.mcp.client.client",
        "backend.mcp.server.run",
        "backend.mcp.server.step_tools",
        "backend.mcp.server.discovery_tools",
        "mcp.server.lowlevel.server",
        "httpx",
        "openai._base_client",
        "httpcore.connection",
        "httpcore.http11",
    ]:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.DEBUG)