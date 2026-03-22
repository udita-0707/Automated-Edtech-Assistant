# Node.js Backend Gateway

The backend serves as the API orchestrator and persistence layer for the Automated EdTech Assistant.

## Key Responsibilities
- **API Orchestration**: Routes grading requests between the frontend and the Python ML service.
- **Persistence**: Manages student submission history using **SQLite3**.
- **Historical Tracking**: Allows users to view and audit past grading results.

## Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start server:
   ```bash
   node index.js
   ```

## API Endpoints
- `GET /api/history`: Retrieve full grading history.
- `POST /api/grade`: Submit a new answer for analysis (proxied to ML service).
