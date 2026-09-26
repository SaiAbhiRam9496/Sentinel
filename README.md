# Sentinal

**Sentinal** is an intelligent data quality, profiling, and analytics platform. When users upload structured CSV datasets, Sentinal automatically detects whether the dataset matches an EV charging telemetry schema or is a generic dataset, and runs the corresponding analysis pipeline:

- **Generic Mode**: Schema detection, data profiling, missing-value handling, duplicate detection, transparent data quality scoring, and automated visualizations.
- **EV Mode**: EV-specific domain validation, feature engineering, ML-based anomaly detection (Isolation Forest), anomaly explanations, and EV telemetry visualizations.
- **Grounded AI Assistant**: Domain-aware RAG chat scoped to the active dataset.

---

## Prerequisites

- **Python 3.12** (`python3.12` specifically required)
- **Node.js** (v18+ / v20+) & **npm**
- **Docker** (for local PostgreSQL)

---

## Quick Start: Running the Project

### 1. Start the PostgreSQL Database (Docker)

From the project root:

```bash
docker compose up -d
```

*(Alternatively, run the container directly)*:
```bash
docker run -d --name sentinel-postgres -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=sentinel_db postgres:16-alpine
```

---

### 2. Run the Backend (FastAPI)

From the project root:

```bash
# 1. Create Python 3.12 virtual environment (do NOT use default python3)
python3.12 -m venv backend/venv

# 2. Activate the virtual environment
source backend/venv/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Copy environment configuration
cp backend/.env.example backend/.env

# 5. Start the FastAPI server (runs on http://localhost:8000)
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 3. Run the Frontend (React + Vite + Tailwind CSS)

In a new terminal:

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Vite dev server (runs on http://localhost:5173)
npm run dev
```

---

## Verifying Local Setup

- **Frontend UI**: Open [http://localhost:5173](http://localhost:5173) in your browser.
- **Backend Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Run Backend Tests**:
  ```bash
  PYTHONPATH=. backend/venv/bin/pytest backend/tests/
  ```
