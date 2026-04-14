# InterVA Analysis API

A web application for analyzing Verbal Autopsy (VA) data using the InterVA5 and InterVA6 algorithms to determine probable causes of death. Built with a FastAPI backend and a browser-based frontend.

---

## Prerequisites

- Python 3.11+
- pip
- (Optional) Docker

---

## Local Development Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd interva_jnja
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the development server

```bash
uvicorn main:app --reload
```

The app will be available at **http://127.0.0.1:8000**

- Web interface: `http://127.0.0.1:8000/`
- API docs (Swagger): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

---

## Running with Docker

### Build the image

```bash
docker build -t vman3_ccva .
```

### Multi-platform build and push to repo
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ilyatuu/vman3_ccva:latest --push .
```

### Run the container

```bash
docker run -p 8000:8000 vman3_ccva
```

The app will be available at **http://localhost:8000**

---

## Project Structure

```
interva_jnja/
├── main.py              # FastAPI app — entry point
├── interva6.py          # InterVA6 algorithm implementation
├── index.html           # Web interface (served by FastAPI)
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── probbase2022.csv     # Probability database for the InterVA algorithm
├── causetext2022.csv    # Cause of death text mappings
├── VA_output/           # Generated output files
├── interva6/            # InterVA module
└── vman3/               # VMan3 data processing toolkit
    ├── vman3/           # Core module (cleaning, validation)
    ├── pycrossva/       # WHO VA format transformation
    ├── interva/         # InterVA implementations
    └── vacheck/         # Data quality checks
```

---

## Supported Input Formats

| Format | Description |
|--------|-------------|
| `2016WHOv151` | WHO VA 2016 v1.5.1 |
| `2022WHOv0101` | WHO VA 2022 v0.1.0.1 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `POST` | `/upload-chunk` | Upload data (chunked) |
| `GET` | `/stream/{session_id}` | Real-time progress (SSE) |
| `GET` | `/download-result/{session_id}` | Download analysis results |
| `GET` | `/get-csmf/{session_id}` | Get cause-specific mortality fractions |
| `GET` | `/get-error-log/{session_id}` | Get validation errors |
| `POST` | `/cancel/{session_id}` | Cancel a running session |
| `POST` | `/cleanup-session/{session_id}` | Clean up session data |

---

## VMan3 Module (Development)

If you need to work on the `vman3` data processing toolkit directly:

```bash
cd vman3
pip install -r requirements.txt

# Set the Python path so imports resolve correctly
export PYTHONPATH=.          # Linux/macOS
$env:PYTHONPATH = "."        # Windows PowerShell

# Run tests
python tests/test.py
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to Render.com via Docker.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `PYTHONUNBUFFERED` | `1` | Unbuffered output |
| `ENVIRONMENT` | `production` | Environment flag |
