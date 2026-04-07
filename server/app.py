import os

import uvicorn
from backend.main import app


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
