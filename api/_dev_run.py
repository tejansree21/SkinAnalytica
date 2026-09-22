import os
os.environ.setdefault("SKINANALYTICA_API_KEY", "bdfbecb2333bb2f07d593c4a3f7582b3")
import uvicorn
uvicorn.run("SA05_api:app", host="0.0.0.0", port=8001)
