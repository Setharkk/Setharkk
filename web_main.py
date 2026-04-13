"""Setharkk Web — Point d'entree."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.server:app",
        host="127.0.0.1",
        port=8090,
        reload=False,
    )
