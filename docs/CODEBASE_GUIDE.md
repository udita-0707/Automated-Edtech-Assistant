File: backend/index.js
Function: (module startup)
Purpose:
Starts the Node.js "gateway" API that the frontend talks to. It validates requests, proxies them to the Python ML service, and stores results in a local SQLite database for history/analytics.
Input:
- Environment variables:
  - PORT (optional, default 3000)
  - ML_SERVICE_URL (optional, default http://localhost:8000)
Output:
- A running HTTP server (Express) with REST endpoints under /api/*
- A local SQLite file created/used at backend/database.sqlite
Step-by-step logic:
1. Imports libraries (Express, CORS, SQLite client, axios for HTTP, multer for uploads).
2. Reads PORT and ML_SERVICE_URL from environment (or uses defaults).
3. Installs middleware (CORS and JSON body parsing).
4. Opens/creates SQLite database file and ensures the submissions table exists.
5. Creates an in-memory multer uploader for OCR images.
6. Registers API routes (grade, ocr, history, analytics, health).
7. Calls app.listen() to start the server.
Why this function exists:
To separate "web app concerns" (HTTP, persistence, request validation) from "ML concerns" (OCR + grading). This makes it easier to swap ML implementations without rewriting the UI.
How it connects to other components:
- Called by: developer running `node backend/index.js`
- Called by: frontend via HTTP (hardcoded base URL in frontend/src/App.jsx)
- Calls into: ML service via HTTP at ML_SERVICE_URL (/predict and /ocr)
- Writes/reads: SQLite (submissions table) for /api/history and /api/analytics

File: backend/index.js
Function: POST /api/grade handler
Purpose:
Receives a grading request from the frontend, forwards it to the ML service, stores the result, and returns it back to the frontend.
Input:
- HTTP request JSON body:
  - question (string)
  - reference_answer (string)
  - student_answer (string)
Output:
- HTTP JSON response from ML service (the "grade result"):
  - predicted_label (string)
  - similarity_score (number)
  - confidence (number)
  - feedback (string)
Step-by-step logic:
1. Reads question/student_answer/reference_answer from req.body.
2. Validates required fields; returns 400 if any are missing.
3. Sends axios POST to `${ML_SERVICE_URL}/predict` with the same fields.
4. Receives ML result JSON.
5. Inserts a new row into SQLite submissions table with inputs + ML outputs.
6. Responds with res.json(result) to the frontend.
Why this function exists:
The frontend should not talk to the ML service directly if you want persistence, consistent errors, and a single stable API for the UI.
How it connects to other components:
- Called by: frontend/src/App.jsx (GradePanel.handleGrade)
- Calls into: ml-service/main.py:predict (or phase1/api/main.py:predict, if using that service)
- Writes to: backend/database.sqlite (submissions table)
- Feeds: HistoryPanel and AnalyticsPanel via stored rows

File: backend/index.js
Function: POST /api/ocr handler
Purpose:
Receives an uploaded handwriting image from the frontend and forwards it to the ML service OCR endpoint, returning the transcribed text.
Input:
- HTTP multipart/form-data:
  - file (image binary)
Output:
- HTTP JSON response:
  - transcribed_text (string)
Step-by-step logic:
1. Validates that multer parsed an uploaded file; returns 400 if missing.
2. Builds a FormData object containing the uploaded file buffer.
3. Sends axios POST to `${ML_SERVICE_URL}/ocr` with multipart headers.
4. Returns the ML OCR response back to the frontend.
Why this function exists:
To keep the browser UI simple (upload once to backend) and avoid CORS/credentials issues with the ML service, while allowing the ML service to evolve independently.
How it connects to other components:
- Called by: frontend/src/App.jsx (GradePanel.handleOCRUpload)
- Calls into: ml-service/main.py:ocr (Phase 1 Tesseract OCR) by default
- Supplies: student_answer text that is then graded by POST /api/grade

File: backend/index.js
Function: GET /api/history handler
Purpose:
Lets the frontend show "recent submissions" by reading the latest rows from SQLite.
Input:
- No request body (simple GET)
Output:
- JSON array of up to 50 submission rows (each row includes question, answers, predicted label, scores, feedback, created_at).
Step-by-step logic:
1. Runs a SQL query: select submissions ordered by created_at DESC limit 50.
2. Returns the rows as JSON.
Why this function exists:
Users want to see previously graded answers. The ML service does not store history; SQLite in the backend does.
How it connects to other components:
- Called by: frontend/src/App.jsx (HistoryPanel useEffect)
- Reads from: backend/database.sqlite

File: backend/index.js
Function: GET /api/analytics handler
Purpose:
Computes quick platform stats (count, average similarity, and label distribution) from the stored SQLite submissions.
Input:
- No request body (simple GET)
Output:
- JSON object:
  - total_submissions (number)
  - average_similarity (number)
  - distribution (object: correct/partially correct/incorrect -> count)
Step-by-step logic:
1. Queries total submission count.
2. Queries average similarity_score.
3. Queries counts grouped by predicted_label.
4. Fills a fixed distribution map so missing labels appear as 0.
5. Returns the combined analytics object.
Why this function exists:
Analytics should be computed from stored results (history), not recomputed by ML, and this endpoint keeps the frontend simple.
How it connects to other components:
- Called by: frontend/src/App.jsx (AnalyticsPanel useEffect)
- Reads from: backend/database.sqlite
- Visualized by: DistributionBar in the frontend

File: backend/index.js
Function: GET / handler
Purpose:
Simple health check endpoint to confirm the backend server is up.
Input:
- No input
Output:
- Plain text status string
Step-by-step logic:
1. Responds with a static string.
Why this function exists:
It is a quick sanity check while developing (browser or curl).
How it connects to other components:
- Indirectly helps debugging when the frontend shows "backend not running".

File: frontend/src/main.jsx
Function: (React DOM bootstrap)
Purpose:
Mounts the React app into the HTML page so the UI appears in the browser.
Input:
- DOM element with id="root" from frontend/index.html
Output:
- A running React application rendered into #root
Step-by-step logic:
1. Imports StrictMode and ReactDOM createRoot.
2. Imports global CSS (index.css) and the main App component.
3. Creates a root on the #root element.
4. Renders <App /> inside <StrictMode>.
Why this function exists:
Every React app needs an entry point that attaches React to the DOM.
How it connects to other components:
- Calls into: frontend/src/App.jsx (the full UI)
- Depends on: frontend/index.html for the root element

File: frontend/src/App.jsx
Function: cn
Purpose:
Combines conditional CSS class strings in a Tailwind-friendly way (deduplicates with tailwind-merge).
Input:
- Any number of class name strings/values (often conditional)
Output:
- A single merged class string
Step-by-step logic:
1. Uses clsx() to conditionally build a class string.
2. Uses twMerge() to remove conflicting Tailwind classes (for example "p-2" vs "p-4").
Why this function exists:
Tailwind UIs often build className strings dynamically; this helper reduces bugs and messy JSX.
How it connects to other components:
- Used by: App, HistoryPanel, and DistributionBar to style elements based on state/data.

File: frontend/src/App.jsx
Function: App
Purpose:
Top-level UI component: provides the header and tab navigation between grading, history, and analytics.
Input:
- No props (local state only)
Output:
- Rendered UI containing:
  - GradePanel, HistoryPanel, or AnalyticsPanel based on activeTab
Step-by-step logic:
1. Creates state activeTab ("grade" by default).
2. Renders header text and three tab buttons.
3. On button click, updates activeTab.
4. Conditionally renders the selected panel component.
Why this function exists:
Keeps the UI organized: a single parent that controls which "screen" is visible.
How it connects to other components:
- Renders: GradePanel (calls backend /api/grade and /api/ocr)
- Renders: HistoryPanel (calls backend /api/history)
- Renders: AnalyticsPanel (calls backend /api/analytics)

File: frontend/src/App.jsx
Function: GradePanel
Purpose:
Collects the grading inputs (question/reference/student), optionally runs OCR to fill the student answer, calls the backend to grade, and displays the returned result.
Input:
- No props (local state only)
Output:
- Rendered grading form and a result dashboard
Step-by-step logic:
1. Initializes text fields with example question/answers.
2. Tracks UI state: loading, ocrLoading, result, error.
3. Provides a "Scan Handwriting" file input that triggers OCR upload.
4. Provides an "Evaluate Response" button that triggers grading.
5. Displays a placeholder, a loading spinner, or the result depending on state.
Why this function exists:
This is the main user workflow: turn a student response (typed or handwritten) into a grade and feedback.
How it connects to other components:
- Calls backend: POST /api/ocr -> proxies to ML OCR -> fills student answer
- Calls backend: POST /api/grade -> proxies to ML predict -> stores to SQLite
- The stored results later appear in HistoryPanel and AnalyticsPanel

File: frontend/src/App.jsx
Function: GradePanel.handleGrade
Purpose:
Sends the user-entered question/reference/student text to the backend for grading.
Input:
- Current React state:
  - question (string)
  - reference (string) -> sent as reference_answer
  - student (string) -> sent as student_answer
Output:
- Updates React state:
  - result (object) on success
  - error (string) on failure
Step-by-step logic:
1. Validates that question/reference/student are not empty; sets error if missing.
2. Sets loading=true and clears previous result.
3. POSTs JSON to http://localhost:3000/api/grade.
4. Stores response JSON in result.
5. On error, shows a beginner-friendly message.
6. Always sets loading=false at the end.
Why this function exists:
You want one predictable place that turns the current form state into an API call and handles loading/errors.
How it connects to other components:
- Calls into: backend/index.js POST /api/grade
- Backend calls into: ml-service/main.py POST /predict (default)
- Backend writes: SQLite submission row that powers history/analytics

File: frontend/src/App.jsx
Function: GradePanel.handleOCRUpload
Purpose:
Uploads a handwriting image to the backend OCR endpoint and puts the transcribed text into the student answer box.
Input:
- Browser change event from <input type="file">
Output:
- Updates React state:
  - student (string) from OCR result
  - error (string) if OCR fails or returns empty
Step-by-step logic:
1. Reads the first selected file from e.target.files.
2. Sets ocrLoading=true and clears any previous error.
3. Builds a FormData payload with field name "file".
4. POSTs multipart data to http://localhost:3000/api/ocr.
5. If response contains transcribed_text, sets student to that value.
6. Otherwise shows an error telling the user to try a clearer image.
7. Resets loading state and clears the file input value to allow re-uploading the same file.
Why this function exists:
Handwriting entry is a key part of the "Document AI" pipeline; OCR is how images become text that the grader can understand.
How it connects to other components:
- Calls into: backend/index.js POST /api/ocr
- Backend calls into: ml-service/main.py POST /ocr (default)
- Feeds into: handleGrade, because grading uses the student text produced by OCR

File: frontend/src/App.jsx
Function: HistoryPanel
Purpose:
Fetches and displays recent grading submissions stored by the backend.
Input:
- No props; fetches from backend
Output:
- A list of recent submissions or an empty-state message
Step-by-step logic:
1. On mount, GETs http://localhost:3000/api/history.
2. Stores the returned array in history state.
3. While loading, shows a spinner.
4. Renders each row with label badge, timestamp, question, student answer snippet, and score chips.
Why this function exists:
It turns the backend database (raw rows) into a human-friendly UI so users can review past work.
How it connects to other components:
- Calls into: backend/index.js GET /api/history
- Backend reads: SQLite submissions table populated by POST /api/grade

File: frontend/src/App.jsx
Function: AnalyticsPanel
Purpose:
Fetches and displays aggregate analytics computed from stored submissions.
Input:
- No props; fetches from backend
Output:
- A stats dashboard (total graded, average similarity, distribution bars)
Step-by-step logic:
1. On mount, GETs http://localhost:3000/api/analytics.
2. Stores the returned stats object.
3. Shows a spinner while loading.
4. Renders numeric stats.
5. Renders a DistributionBar for each label category.
Why this function exists:
Users need high-level insights (how often answers are correct, average similarity, etc.) without reading every submission.
How it connects to other components:
- Calls into: backend/index.js GET /api/analytics
- Uses: DistributionBar to visualize backend-provided distribution values

File: frontend/src/App.jsx
Function: DistributionBar
Purpose:
Reusable UI component that visualizes a category count as a percentage bar.
Input:
- label (string)
- count (number)
- total (number)
- color (string Tailwind class, e.g., "bg-green-500")
Output:
- A labeled progress bar showing count and percentage
Step-by-step logic:
1. Computes percentage = (count / total) * 100 (handles total=0 safely).
2. Renders label text and percentage text.
3. Renders an outer bar container and an inner bar with width set to percentage%.
Why this function exists:
It keeps the analytics UI simple by centralizing the "percentage bar" logic in one place.
How it connects to other components:
- Called by: AnalyticsPanel
- Visualizes: stats produced by backend/index.js GET /api/analytics

File: frontend/vite.config.js
Function: (Vite configuration export)
Purpose:
Configures the frontend development/build tool (Vite) to use React and Tailwind.
Input:
- Vite runtime reads this file automatically during `npm run dev` and `npm run build`
Output:
- Vite build behavior (plugins enabled)
Step-by-step logic:
1. Imports defineConfig, React plugin, and Tailwind Vite plugin.
2. Exports a config object with plugins enabled.
Why this function exists:
Without it, Vite would not automatically handle React JSX and Tailwind compilation the way this project expects.
How it connects to other components:
- Enables: frontend/src/App.jsx to compile and style correctly

File: frontend/eslint.config.js
Function: (ESLint configuration export)
Purpose:
Defines linting rules for the frontend codebase to catch common JS/React issues early.
Input:
- ESLint reads this file during `npm run lint`
Output:
- Lint results (errors/warnings)
Step-by-step logic:
1. Exports a config describing language globals, plugins, and rules.
Why this function exists:
Linting reduces bugs (especially in React hooks usage) by catching mistakes before runtime.
How it connects to other components:
- Applies to: frontend/src/*.jsx during local development and CI (if you add CI later)

File: ml-service/config.py
Function: (configuration constants)
Purpose:
Central place for ML-service hyperparameters (dataset name, label scheme, TF-IDF size) and artifact paths.
Input:
- Imported by ml-service/utils.py, ml-service/model.py, and ml-service/run_train_eval.py
Output:
- Constant values used by the training and inference code
Step-by-step logic:
1. Defines DATASET_NAME and LABEL_SCHEME for consistent dataset loading.
2. Defines MAX_FEATURES and RANDOM_STATE for reproducible training.
3. Defines relative paths for model/scaler/vectorizer artifacts.
Why this function exists:
Separating constants from logic keeps the code readable and makes it easier to tune settings.
How it connects to other components:
- Controls: which dataset split/label mapping the pipelines use
- Controls: where artifacts are saved/loaded by TextClassifier.save/load

File: ml-service/utils.py
Function: load_scientsbank
Purpose:
Downloads/loads the SciEntsBank dataset from Hugging Face so training/evaluation can run.
Input:
- None (uses config.DATASET_NAME)
Output:
- A Hugging Face DatasetDict with splits like train, test_ua, test_uq, test_ud
Step-by-step logic:
1. Calls datasets.load_dataset(config.DATASET_NAME).
2. Returns the dataset object.
Why this function exists:
All training and evaluation needs a consistent data source; this hides the dataset library details from the rest of the code.
How it connects to other components:
- Called by: ml-service/run_train_eval.py, notebooks/eda.py (via import), phase1/phase2 training scripts (similar code)
- Feeds: prepare_dataframe which supplies model training data

File: ml-service/utils.py
Function: convert_labels
Purpose:
Converts the original SciEntsBank 5-way labels into the project’s 3-way label scheme.
Input:
- dataset (DatasetDict)
- scheme (string, default config.LABEL_SCHEME)
Output:
- A dataset with labels aligned to a 3-way ClassLabel:
  - 0: correct
  - 1: contradictory
  - 2: incorrect (includes partially_correct_incomplete, irrelevant, non_domain)
Step-by-step logic:
1. If scheme is "3way", maps original label names into three bins.
2. Casts the label column into a ClassLabel with the 3-way names.
3. Returns the modified dataset.
Why this function exists:
The ML models in this repo are trained as a 3-class classifier. Label conversion must be consistent across training, evaluation, and analysis.
How it connects to other components:
- Called by: ml-service/run_train_eval.py and evaluation notebooks/scripts
- Affects: how TextClassifier.train interprets y labels and what the UI label_map returns

File: ml-service/utils.py
Function: get_jaccard_similarity
Purpose:
Computes a simple token-overlap similarity between student and reference answers, after stopword removal.
Input:
- str1 (student answer string)
- str2 (reference answer string)
Output:
- similarity score (float in [0, 1])
Step-by-step logic:
1. Lowercases and tokenizes both strings using NLTK.
2. Removes stopwords and non-alphanumeric tokens.
3. Computes Jaccard = |intersection| / |union|.
4. Returns 1.0 if both sets are empty (edge case).
Why this function exists:
Jaccard overlap is an inexpensive feature that captures keyword coverage, which helps classical ML graders.
How it connects to other components:
- Used by: TextClassifier.extract_features -> becomes part of the feature vector for Logistic Regression
- Used by: notebooks/eda.py to build the "manifold" feature set

File: ml-service/utils.py
Function: get_token_density
Purpose:
Measures how much of the reference answer’s important vocabulary appears in the student answer.
Input:
- student (string)
- reference (string)
Output:
- density score (float in [0, 1])
Step-by-step logic:
1. Tokenizes and stopword-filters reference and student texts.
2. Treats reference tokens as the "required set".
3. Computes density = |intersection| / |reference tokens|.
4. Returns 1.0 if reference tokens are empty (edge case).
Why this function exists:
Token density encourages answers that include key domain words, which is useful for scientific short answers.
How it connects to other components:
- Used by: TextClassifier.extract_features (Phase 1 engineered feature)
- Complements: Jaccard similarity (overlap/union) by focusing on coverage of reference tokens

File: ml-service/utils.py
Function: prepare_dataframe
Purpose:
Creates aligned "student answer", "reference answer", and "label" series for training and evaluation.
Input:
- dataset (DatasetDict)
- split (string, e.g., "train", "test_ua", "test_uq", "test_ud")
Output:
- 6 values:
  - X_train (Series of student answers from train split)
  - X_test (Series of student answers from requested split)
  - y_train (Series of labels from train split)
  - y_test (Series of labels from requested split)
  - train_ref (Series of reference answers from train split)
  - test_ref (Series of reference answers from requested split)
Step-by-step logic:
1. Converts dataset["train"] and dataset[split] to pandas DataFrames.
2. Extracts the columns needed for grading tasks.
3. Returns them as a fixed tuple so scripts can unpack consistently.
Why this function exists:
Most scripts need the same columns. Returning them in a consistent order prevents "off-by-one unpacking" mistakes.
How it connects to other components:
- Called by: training/evaluation scripts to feed TextClassifier.train and TextClassifier.evaluate
- Supplies: reference answers used by engineered similarity features

File: ml-service/model.py
Function: TextClassifier.__init__
Purpose:
Initializes the Phase 1 grading model components (TF-IDF vectorizer, numeric feature scaler, and Logistic Regression classifier).
Input:
- max_features (int, defaults to config.MAX_FEATURES)
Output:
- A TextClassifier instance ready to train or load artifacts
Step-by-step logic:
1. Builds a TfidfVectorizer with stopwords, 1-2 grams, and max_features.
2. Creates a StandardScaler for numeric engineered features.
3. Creates a LogisticRegression model with class balancing.
Why this function exists:
It centralizes all ML component initialization so training and inference share the same configuration.
How it connects to other components:
- Called by: ml-service/main.py on server startup and ml-service/run_train_eval.py during training
- Depends on: config.py constants

File: ml-service/model.py
Function: TextClassifier.extract_features
Purpose:
Creates the engineered numeric feature matrix for student answers (and optionally uses reference answers to compute semantic overlap features).
Input:
- texts (iterable of student answer strings)
- references (optional iterable/Series of reference answers aligned to texts)
Output:
- numpy array shape (n_samples, 5) containing:
  - length, word_count, avg_word_len, jaccard, token_density
Step-by-step logic:
1. For each student text, computes length/word_count/avg_word_len.
2. If references are provided, computes Jaccard and token density vs the reference.
3. Returns a stacked numpy array for all samples.
Why this function exists:
Pure TF-IDF is sometimes not enough; adding structural and overlap features improves grading accuracy with classical models.
How it connects to other components:
- Called by: TextClassifier.train and TextClassifier.evaluate to build training/evaluation matrices
- Calls into: utils.get_jaccard_similarity and utils.get_token_density

File: ml-service/model.py
Function: TextClassifier.train
Purpose:
Trains the Logistic Regression model on TF-IDF + engineered features.
Input:
- X_train (student answers)
- y_train (integer labels)
- references (reference answers aligned to X_train, needed for Jaccard/density)
Output:
- Trained model stored inside the TextClassifier instance
Step-by-step logic:
1. Fits TF-IDF vectorizer on X_train and transforms it to a sparse matrix.
2. Extracts engineered features and fits the scaler.
3. Converts TF-IDF to dense array and horizontally stacks scaled engineered features.
4. Fits Logistic Regression on the combined feature matrix and labels.
Why this function exists:
This is the core of the Phase 1 "classical ML" grading pipeline.
How it connects to other components:
- Called by: ml-service/run_train_eval.py and phase1/run_train_eval.py
- Produces artifacts saved by: TextClassifier.save (used later by inference endpoints)

File: ml-service/model.py
Function: TextClassifier.evaluate
Purpose:
Evaluates the trained classifier on a test split and produces standard classification metrics.
Input:
- X_test (student answers)
- y_test (labels)
- references (reference answers aligned to X_test)
Output:
- acc (float)
- f1 (float, macro)
- report (string, classification_report)
Step-by-step logic:
1. Transforms X_test using the existing TF-IDF vectorizer.
2. Extracts engineered features and scales them using the fitted scaler.
3. Combines TF-IDF and engineered features.
4. Predicts labels and computes accuracy/F1/report.
Why this function exists:
You need a repeatable way to measure model quality across the SciEntsBank evaluation splits.
How it connects to other components:
- Called by: ml-service/run_train_eval.py and phase1/run_train_eval.py
- Results are printed (ml-service) or saved as files/plots (phase1)

File: ml-service/model.py
Function: TextClassifier.predict_detailed
Purpose:
Runs inference for one student/reference pair and returns UI-friendly fields (label, confidence, similarity, and feedback).
Input:
- student_answer (string)
- reference_answer (string)
Output:
- dict with:
  - label_idx (int)
  - confidence (float)
  - similarity_score (float, Jaccard)
  - length_ratio (float)
  - feedback (string)
Step-by-step logic:
1. Transforms student_answer with TF-IDF vectorizer.
2. Extracts engineered features against the reference answer.
3. Scales numeric features and combines with TF-IDF.
4. Uses Logistic Regression to predict the class and class probabilities.
5. Picks max probability as confidence.
6. Builds a human-readable feedback string based on predicted class.
Why this function exists:
The UI and API need more than a raw class id; they need a label, confidence, and an explanation for the student/teacher.
How it connects to other components:
- Called by: ml-service/main.py POST /predict (and phase1/api/main.py POST /predict)
- Drives UI fields: predicted_label, similarity_score, confidence, feedback

File: ml-service/model.py
Function: TextClassifier.save
Purpose:
Writes trained artifacts to disk so the API server can load them later without retraining.
Input:
- None (uses config paths)
Output:
- Files written (joblib):
  - data/model.pkl
  - data/scaler.pkl
  - data/vectorizer.pkl
Step-by-step logic:
1. Ensures the artifact directory exists.
2. joblib.dump() model, scaler, and vectorizer to the configured paths.
Why this function exists:
Training is slow (dataset download + fitting). Saving lets the API start quickly and be reproducible.
How it connects to other components:
- Called by: ml-service/run_train_eval.py after training
- Read by: TextClassifier.load during API startup

File: ml-service/model.py
Function: TextClassifier.load
Purpose:
Loads previously trained model artifacts so the API can do predictions without training.
Input:
- None (uses config paths)
Output:
- True if artifacts were found and loaded; False otherwise
Step-by-step logic:
1. Checks if config.MODEL_PATH exists.
2. joblib.load() model, scaler, and vectorizer from disk.
3. Returns True if successful, else False.
Why this function exists:
API servers should not retrain on every startup; they should load known-good artifacts.
How it connects to other components:
- Called by: ml-service/main.py at startup
- Required for: POST /predict to work correctly

File: ml-service/main.py
Function: (module startup)
Purpose:
Initializes the FastAPI app, sets up CORS, attempts to enable Tesseract OCR dependencies, and loads the saved TextClassifier artifacts (if present).
Input:
- Python process working directory (affects relative artifact paths in ml-service/config.py)
- Optional local installation of:
  - pytesseract
  - Pillow
  - tesseract binary in PATH
Output:
- A FastAPI app object (`app`) with routes registered
- A global predictor (`predictor`) that is either loaded from artifacts or left uninitialized (warning printed)
Step-by-step logic:
1. Tries to import pytesseract/Pillow and locate the `tesseract` binary; sets a flag `_TESSERACT_AVAILABLE`.
2. Creates FastAPI app and enables permissive CORS (allow_origins ["*"]).
3. Instantiates TextClassifier() as `predictor`.
4. Calls predictor.load(); prints a warning if artifacts are missing.
5. Defines request/response schemas and route handlers.
Why this function exists:
FastAPI needs an app object with routes and initialized model state; this block ensures the service is ready as soon as uvicorn imports `main:app`.
How it connects to other components:
- Called by: `uvicorn main:app` (imports this module)
- Serves: backend/index.js proxy calls to /predict and /ocr
- Depends on: artifacts created by ml-service/run_train_eval.py

File: ml-service/main.py
Function: GradeRequest (Pydantic model)
Purpose:
Defines the expected JSON schema for the /predict endpoint so FastAPI can validate inputs.
Input:
- JSON body with:
  - question (string)
  - student_answer (string)
  - reference_answer (string)
Output:
- A validated GradeRequest object passed into the predict() handler
Step-by-step logic:
1. Pydantic parses and validates incoming JSON.
2. If invalid/missing, FastAPI returns a 422 error automatically.
Why this function exists:
Beginner-friendly APIs should fail early with clear errors; schema validation prevents confusing model failures.
How it connects to other components:
- Used by: ml-service/main.py predict() to access typed fields
- Mirrors: backend/index.js expected JSON shape

File: phase1/api/main.py
Function: (module startup)
Purpose:
Initializes the Phase 1 FastAPI app, sets up CORS, optionally enables Tesseract OCR imports, and loads Phase 1 TextClassifier artifacts if present.
Input:
- Python process working directory (affects relative artifact paths in phase1/config.py)
- Optional pytesseract/Pillow installation for OCR
Output:
- FastAPI app object and global predictor instance
Step-by-step logic:
1. Tries to import pytesseract/Pillow and sets `_TESSERACT_AVAILABLE`.
2. Creates FastAPI app and enables permissive CORS.
3. Instantiates TextClassifier() and calls load().
4. Registers /, /predict, and /ocr routes.
Why this function exists:
This makes Phase 1 runnable as a standalone service (separate from ml-service/) for submission/evaluation.
How it connects to other components:
- Can be used by: backend/index.js by pointing ML_SERVICE_URL to this service
- Uses: phase1/grading/model.py and phase1/config.py for inference

File: phase2/api/main.py
Function: (module startup)
Purpose:
Initializes the Phase 2 FastAPI app and attempts to load the hybrid grader (Model C) and neural OCR (TrOCR) at startup.
Input:
- phase2/config.py constants (HYBRID_ALPHA, MODEL_A_PATH, TROCR_MODEL)
- Availability of:
  - phase2/data/artifacts/model_a.pkl (trained SVM artifact)
  - torch/transformers/sentence-transformers installed
Output:
- FastAPI app object plus global `grader` and `ocr_engine` instances (or a warning if they fail to load)
Step-by-step logic:
1. Creates FastAPI app.
2. In a try/except:
   - Instantiates HybridGrader(alpha=HYBRID_ALPHA) and loads Model A from MODEL_A_PATH.
   - Instantiates HandwritingOCR(model_name=TROCR_MODEL).
3. Defines schema and route handlers.
Why this function exists:
Phase 2 models are heavier; loading at startup ensures requests fail fast if artifacts/deps are missing rather than failing mid-request.
How it connects to other components:
- Can be wired to: backend/index.js by setting ML_SERVICE_URL=http://localhost:8001
- Depends on: phase2/run_train_eval.py producing model_a.pkl

File: phase1/backend/index.js
Function: POST /api/grade handler
Purpose:
Phase 1 snapshot of the backend grading proxy endpoint (same behavior as backend/index.js).
Input:
- JSON body: question, student_answer, reference_answer
Output:
- Grade result JSON from the ML service
Step-by-step logic:
1. Validates body fields.
2. Proxies to `${ML_SERVICE_URL}/predict`.
3. Persists to SQLite and returns result.
Why this function exists:
Keeps Phase 1 submission runnable as a self-contained system.
How it connects to other components:
- Called by: phase1/frontend/src/App.jsx handleGrade
- Calls into: whichever ML service ML_SERVICE_URL points to (Phase 1 or ml-service)

File: phase1/backend/index.js
Function: POST /api/ocr handler
Purpose:
Phase 1 snapshot of the backend OCR proxy endpoint (same behavior as backend/index.js).
Input:
- multipart file upload
Output:
- { transcribed_text }
Step-by-step logic:
1. Receives upload via multer and proxies to `${ML_SERVICE_URL}/ocr`.
Why this function exists:
Supports the handwriting -> text step in the Phase 1 end-to-end demo.
How it connects to other components:
- Called by: phase1/frontend/src/App.jsx handleOCRUpload
- Calls into: ML service OCR endpoint

File: phase1/backend/index.js
Function: GET /api/history handler
Purpose:
Phase 1 snapshot of history endpoint.
Input:
- None
Output:
- Recent submissions array
Step-by-step logic:
1. Queries SQLite and returns rows.
Why this function exists:
Lets the Phase 1 UI show recent submissions.
How it connects to other components:
- Called by: phase1/frontend/src/App.jsx HistoryPanel

File: phase1/backend/index.js
Function: GET /api/analytics handler
Purpose:
Phase 1 snapshot of analytics endpoint.
Input:
- None
Output:
- Analytics JSON (counts/averages/distribution)
Step-by-step logic:
1. Aggregates over SQLite submissions.
Why this function exists:
Lets the Phase 1 UI show summary insights.
How it connects to other components:
- Called by: phase1/frontend/src/App.jsx AnalyticsPanel

File: phase1/frontend/src/App.jsx
Function: cn
Purpose:
Phase 1 snapshot of the Tailwind class merge helper (same behavior as frontend/src/App.jsx).
Input:
- class name values
Output:
- merged class string
Step-by-step logic:
1. clsx() then twMerge().
Why this function exists:
Keeps UI styling logic consistent in the Phase 1 snapshot.
How it connects to other components:
- Used throughout the Phase 1 UI components.

File: phase1/frontend/src/App.jsx
Function: GradePanel.handleGrade
Purpose:
Phase 1 snapshot: posts grading request to the backend.
Input:
- question/reference/student React state
Output:
- result state updated with grade result
Step-by-step logic:
1. Validates fields, posts to /api/grade, stores response.
Why this function exists:
Implements the main grading workflow in the Phase 1 UI.
How it connects to other components:
- Calls into: phase1/backend/index.js POST /api/grade (or backend/index.js if you run that)

File: phase1/frontend/src/App.jsx
Function: GradePanel.handleOCRUpload
Purpose:
Phase 1 snapshot: uploads handwriting image to the backend OCR endpoint.
Input:
- file input change event
Output:
- student state updated with transcribed_text
Step-by-step logic:
1. Builds FormData and posts to /api/ocr.
Why this function exists:
Connects handwriting OCR into the same grading form.
How it connects to other components:
- Calls into: phase1/backend/index.js POST /api/ocr (or backend/index.js)

File: phase1/frontend/src/App.jsx
Function: App
Purpose:
Phase 1 snapshot of the top-level React component that hosts the tab navigation.
Input:
- No props
Output:
- Renders GradePanel, HistoryPanel, or AnalyticsPanel
Step-by-step logic:
1. Tracks activeTab in state.
2. Renders tab buttons and conditionally renders the selected panel.
Why this function exists:
It is the entry component for the Phase 1 UI snapshot.
How it connects to other components:
- Renders: GradePanel (calls /api/grade and /api/ocr)
- Renders: HistoryPanel (calls /api/history)
- Renders: AnalyticsPanel (calls /api/analytics)

File: phase1/frontend/src/App.jsx
Function: HistoryPanel
Purpose:
Phase 1 snapshot: fetches and displays recent submissions.
Input:
- None (fetches from backend)
Output:
- Rendered list of submission cards
Step-by-step logic:
1. useEffect GETs /api/history on mount.
2. Stores response array and renders list.
Why this function exists:
Shows persisted grading history in the Phase 1 UI.
How it connects to other components:
- Calls into: phase1/backend/index.js GET /api/history
- Data is generated by: POST /api/grade persistence

File: phase1/frontend/src/App.jsx
Function: AnalyticsPanel
Purpose:
Phase 1 snapshot: fetches and displays analytics from stored submissions.
Input:
- None (fetches from backend)
Output:
- Rendered totals, averages, and distribution bars
Step-by-step logic:
1. useEffect GETs /api/analytics on mount.
2. Stores stats and renders DistributionBar components.
Why this function exists:
Gives a quick overview of system behavior without scanning all submissions.
How it connects to other components:
- Calls into: phase1/backend/index.js GET /api/analytics
- Uses: DistributionBar for visualization

File: phase1/frontend/src/App.jsx
Function: DistributionBar
Purpose:
Phase 1 snapshot: renders a percentage bar for a label category.
Input:
- label, count, total, color
Output:
- A progress bar row with computed percentage
Step-by-step logic:
1. Computes percentage from count/total.
2. Renders a bar with width set to that percentage.
Why this function exists:
Keeps the analytics UI clean and reusable.
How it connects to other components:
- Called by: AnalyticsPanel

File: ml-service/main.py
Function: root
Purpose:
Basic ML service health endpoint.
Input:
- None
Output:
- JSON describing service status, model name, and version string
Step-by-step logic:
1. Returns a static JSON dict.
Why this function exists:
It is a quick check that FastAPI is running and reachable by the backend.
How it connects to other components:
- Useful when debugging backend proxy issues to ML_SERVICE_URL

File: ml-service/main.py
Function: predict
Purpose:
API endpoint that grades a student answer vs a reference answer using the Phase 1 TextClassifier.
Input:
- GradeRequest (question, student_answer, reference_answer)
Output:
- JSON:
  - predicted_label (string)
  - similarity_score (float)
  - length_ratio (float)
  - confidence (float)
  - feedback (string)
Step-by-step logic:
1. Calls predictor.predict_detailed(student_answer, reference_answer).
2. Maps label_idx (0/1/2) to UI labels ("correct"/"incorrect"/"partially correct").
3. Returns the mapped fields.
Why this function exists:
This is the ML service contract the Node backend expects at `${ML_SERVICE_URL}/predict`.
How it connects to other components:
- Called by: backend/index.js POST /api/grade
- Feeds: frontend result dashboard and SQLite persistence

File: ml-service/main.py
Function: ocr
Purpose:
API endpoint that transcribes handwriting images using Tesseract OCR (Phase 1 OCR).
Input:
- UploadFile named "file" (image binary)
Output:
- JSON:
  - transcribed_text (string)
Step-by-step logic:
1. Verifies pytesseract/Pillow imports are available; otherwise returns 503.
2. Reads file bytes and opens with PIL Image.
3. Preprocesses image (grayscale, autocontrast, sharpen, binarize).
4. Runs pytesseract.image_to_string with `--oem 0 --psm 6`.
5. Cleans whitespace and returns the text.
Why this function exists:
The grading model expects text. OCR is the bridge from image -> text in the end-to-end pipeline.
How it connects to other components:
- Called by: backend/index.js POST /api/ocr
- Used by: frontend OCR upload to fill the student answer textbox

File: ml-service/run_train_eval.py
Function: main
Purpose:
Offline script to train the Phase 1 model artifacts and print evaluation metrics across SciEntsBank test splits.
Input:
- No CLI args; uses config constants and downloads dataset
Output:
- Saved artifacts (model/scaler/vectorizer)
- Printed evaluation metrics to stdout
Step-by-step logic:
1. Loads and converts labels for the dataset.
2. Builds train dataframe outputs and trains TextClassifier.
3. Saves artifacts.
4. Loops over test splits (test_ua, test_uq, test_ud) and prints accuracy/F1/report.
Why this function exists:
The API requires trained artifacts; this script is the simplest reproducible way to generate them.
How it connects to other components:
- Produces: artifacts that ml-service/main.py loads at startup
- Uses: ml-service/utils.py and ml-service/model.py pipeline pieces

File: phase1/config.py
Function: (configuration constants)
Purpose:
Same role as ml-service/config.py but stored under the "phase1" submission folder for academic packaging.
Input:
- Imported by phase1/* modules
Output:
- Constants used for dataset name, label scheme, and artifact paths
Step-by-step logic:
1. Defines dataset + label scheme.
2. Defines TF-IDF and reproducibility constants.
3. Defines artifact paths (relative).
Why this function exists:
Phase 1 submission code is kept self-contained; it should not depend on ml-service/ at runtime.
How it connects to other components:
- Mirrors: ml-service/config.py so training/inference behavior stays consistent across folders

File: phase1/utils.py
Function: load_scientsbank
Purpose:
Phase 1 copy of ml-service/utils.py: loads the dataset for training/evaluation.
Input:
- None
Output:
- DatasetDict
Step-by-step logic:
1. Calls load_dataset(config.DATASET_NAME).
Why this function exists:
Phase 1 code is organized as a self-contained package for grading and evaluation.
How it connects to other components:
- Used by: phase1/run_train_eval.py and phase1/evaluation/ablation.py

File: phase1/utils.py
Function: convert_labels
Purpose:
Phase 1 copy of ml-service/utils.py: converts 5-way labels to 3-way labels.
Input:
- dataset, scheme
Output:
- DatasetDict with 3-way labels
Step-by-step logic:
1. Applies align_labels_with_mapping and casts ClassLabel.
Why this function exists:
Keeps label conversion consistent across Phase 1 training/evaluation scripts.
How it connects to other components:
- Feeds: phase1/grading/model.py (TextClassifier) label expectations

File: phase1/utils.py
Function: get_jaccard_similarity
Purpose:
Phase 1 copy: computes token-overlap similarity for engineered features.
Input:
- student and reference strings
Output:
- float in [0, 1]
Step-by-step logic:
1. Tokenize, remove stopwords, compute Jaccard overlap.
Why this function exists:
Provides a cheap semantic-overlap feature for classical grading.
How it connects to other components:
- Used by: phase1/grading/model.py TextClassifier.extract_features

File: phase1/utils.py
Function: get_token_density
Purpose:
Phase 1 copy: computes reference vocabulary coverage for engineered features.
Input:
- student and reference strings
Output:
- float in [0, 1]
Step-by-step logic:
1. Tokenize, stopword-filter, compute intersection/reference ratio.
Why this function exists:
Adds "coverage" signal that TF-IDF alone can miss.
How it connects to other components:
- Used by: phase1/grading/model.py TextClassifier.extract_features

File: phase1/utils.py
Function: prepare_dataframe
Purpose:
Phase 1 copy: returns aligned train/test Series for model training and evaluation.
Input:
- dataset, split
Output:
- (X_train, X_test, y_train, y_test, train_ref, test_ref)
Step-by-step logic:
1. Converts HF dataset splits to pandas and selects needed columns.
Why this function exists:
Avoids repeated boilerplate in training/evaluation scripts.
How it connects to other components:
- Used by: phase1/run_train_eval.py and phase1/evaluation/ablation.py

File: phase1/grading/model.py
Function: TextClassifier.__init__
Purpose:
Phase 1 copy of the Phase 1/ML-service TextClassifier initialization.
Input:
- max_features
Output:
- A classifier instance
Step-by-step logic:
1. Creates TF-IDF, scaler, and Logistic Regression objects.
Why this function exists:
Defines the Phase 1 grading model in a self-contained submission folder.
How it connects to other components:
- Used by: phase1/run_train_eval.py and phase1/api/main.py

File: phase1/grading/model.py
Function: TextClassifier.extract_features
Purpose:
Phase 1 copy: builds engineered numeric features.
Input:
- texts, references
Output:
- numpy feature matrix
Step-by-step logic:
1. Computes structural features and overlap features against references.
Why this function exists:
Improves classical model performance using simple, interpretable signals.
How it connects to other components:
- Called by: TextClassifier.train/evaluate/predict_detailed in Phase 1
- Calls into: phase1/utils.py similarity helpers

File: phase1/grading/model.py
Function: TextClassifier.train
Purpose:
Phase 1 copy: trains LogReg on TF-IDF + engineered features.
Input:
- X_train, y_train, references
Output:
- Fitted model stored in the instance
Step-by-step logic:
1. Fit TF-IDF, fit scaler on engineered features, stack, fit LogReg.
Why this function exists:
Core Phase 1 training pipeline.
How it connects to other components:
- Produces artifacts saved by: TextClassifier.save (used by Phase 1 API)

File: phase1/grading/model.py
Function: TextClassifier.evaluate
Purpose:
Phase 1 copy: evaluates trained model and returns metrics.
Input:
- X_test, y_test, references
Output:
- accuracy, f1 macro, classification report string
Step-by-step logic:
1. Transform TF-IDF, scale engineered features, stack, predict, compute metrics.
Why this function exists:
Supports Phase 1 evaluation scripts and report generation.
How it connects to other components:
- Called by: phase1/run_train_eval.py

File: phase1/grading/model.py
Function: TextClassifier.predict_detailed
Purpose:
Phase 1 copy: returns label/confidence/similarity/feedback for a single pair, for UI/API usage.
Input:
- student_answer, reference_answer
Output:
- dict with label_idx, confidence, similarity_score, length_ratio, feedback
Step-by-step logic:
1. Vectorize + engineer features, predict probabilities, pick confidence, return details.
Why this function exists:
APIs and UIs need interpretable outputs, not just a raw label.
How it connects to other components:
- Called by: phase1/api/main.py POST /predict
- Matches: backend/index.js expectations when proxying to Phase 1 service

File: phase1/grading/model.py
Function: TextClassifier.save
Purpose:
Phase 1 copy: saves model artifacts for later loading.
Input:
- None
Output:
- Artifact files written under the configured paths
Step-by-step logic:
1. Makes directories and dumps artifacts with joblib.
Why this function exists:
Avoids retraining for every server start; ensures reproducible evaluation.
How it connects to other components:
- Read by: TextClassifier.load during Phase 1 API startup

File: phase1/grading/model.py
Function: TextClassifier.load
Purpose:
Phase 1 copy: loads saved artifacts for inference.
Input:
- None
Output:
- True/False indicating whether artifacts were found and loaded
Step-by-step logic:
1. Checks artifact existence and loads with joblib.
Why this function exists:
Makes the Phase 1 API usable as a real service without retraining.
How it connects to other components:
- Used by: phase1/api/main.py at startup

File: phase1/ocr/tesseract_engine.py
Function: TesseractOCR.__init__
Purpose:
Initializes the Phase 1 OCR engine and optionally sets the Tesseract binary path.
Input:
- tesseract_cmd (optional string path to tesseract executable)
Output:
- A TesseractOCR instance
Step-by-step logic:
1. If a path is provided, assigns it to pytesseract.pytesseract.tesseract_cmd.
Why this function exists:
Tesseract installation paths differ across machines; allowing an override makes OCR more reliable.
How it connects to other components:
- Can be used by: Phase 1 OCR flows before calling transcribe()

File: phase1/ocr/tesseract_engine.py
Function: TesseractOCR.preprocess
Purpose:
Applies image processing steps that improve Tesseract accuracy on handwriting.
Input:
- image (PIL.Image)
Output:
- processed_image (PIL.Image) in binarized form
Step-by-step logic:
1. Converts to grayscale.
2. Applies autocontrast.
3. Sharpens edges.
4. Binarizes using a threshold.
Why this function exists:
OCR works best with clean, high-contrast input; preprocessing reduces background noise and boosts stroke edges.
How it connects to other components:
- Called by: TesseractOCR.transcribe
- Mirrors: preprocessing inside ml-service/main.py /ocr endpoint

File: phase1/ocr/tesseract_engine.py
Function: TesseractOCR.transcribe
Purpose:
Runs Tesseract OCR on a preprocessed image and returns cleaned text.
Input:
- image (PIL.Image)
Output:
- transcribed text (string)
Step-by-step logic:
1. Calls preprocess(image).
2. Runs pytesseract.image_to_string with `--oem 0 --psm 6`.
3. Collapses whitespace/newlines into a single spaced string.
Why this function exists:
This is the "image -> text" step for Phase 1, which the grading model requires.
How it connects to other components:
- Used conceptually by: phase1/api/main.py /ocr endpoint (Phase 1 service)
- Supplies: student_answer text that is then graded by /predict

File: phase1/api/main.py
Function: GradeRequest (Pydantic model)
Purpose:
Validates the request schema for Phase 1 /predict.
Input:
- JSON body (question, student_answer, reference_answer)
Output:
- Validated GradeRequest instance
Step-by-step logic:
1. Pydantic validates required fields and types.
Why this function exists:
Prevents confusing model errors by rejecting invalid requests early.
How it connects to other components:
- Called by: Phase 1 predict() route
- Matches: backend/index.js POST /api/grade payload

File: phase1/api/main.py
Function: root
Purpose:
Health endpoint for the Phase 1 API.
Input:
- None
Output:
- JSON status fields
Step-by-step logic:
1. Returns a static JSON dict.
Why this function exists:
Helps confirm the Phase 1 API is running before wiring it to the backend gateway.
How it connects to other components:
- Useful when ML_SERVICE_URL points at the Phase 1 service

File: phase1/api/main.py
Function: predict
Purpose:
Grades a student answer via the Phase 1 TextClassifier and returns UI-ready output.
Input:
- GradeRequest
Output:
- JSON grade result (label, similarity, confidence, feedback)
Step-by-step logic:
1. Calls predictor.predict_detailed(student_answer, reference_answer).
2. Maps label index to string label.
3. Returns the response object.
Why this function exists:
Defines the Phase 1 service contract for grading.
How it connects to other components:
- Can be called by: backend/index.js if ML_SERVICE_URL points to this service
- Powers: frontend result UI via the backend proxy

File: phase1/api/main.py
Function: ocr
Purpose:
Transcribes handwriting images using Tesseract and returns a cleaned string.
Input:
- UploadFile "file"
Output:
- JSON { transcribed_text }
Step-by-step logic:
1. Validates pytesseract/Pillow availability.
2. Reads file bytes and preprocesses the image.
3. Runs Tesseract OCR and cleans the output.
Why this function exists:
Phase 1 includes an OCR pathway so handwriting can enter the same grading pipeline as typed text.
How it connects to other components:
- Can be called by: backend/index.js POST /api/ocr via ML_SERVICE_URL
- Used by: frontend Scan Handwriting button

File: phase1/evaluation/ablation.py
Function: run_phase1_ablation
Purpose:
Runs a simplified Phase 1 ablation study: train once, then evaluate across multiple test splits.
Input:
- None (downloads dataset and uses config.LABEL_SCHEME)
Output:
- Printed results
- CSV written to phase1/evaluation/ablation_results.csv
Step-by-step logic:
1. Loads dataset and converts labels.
2. Trains TextClassifier on the train split.
3. Evaluates on test_ua, test_uq, test_ud using TextClassifier.evaluate.
4. Writes a summary CSV.
Why this function exists:
It provides a reproducible evaluation baseline for the report without generating many plots.
How it connects to other components:
- Uses: phase1/utils.py and phase1/grading/model.py pipeline pieces
- Produces: report-friendly metrics for Phase 1

File: phase1/run_train_eval.py
Function: plot_label_distribution
Purpose:
Creates a bar chart showing the class distribution across dataset splits (train/test_ua/test_uq/test_ud).
Input:
- dataset (DatasetDict)
- out_dir (string path)
Output:
- PNG file saved in out_dir (label_distribution.png)
- Counts dict returned to the caller
Step-by-step logic:
1. For each split, counts labels.
2. Builds a grouped bar chart.
3. Saves the figure and returns the counts.
Why this function exists:
Class imbalance is important context for why class_weight="balanced" is used and why results may skew.
How it connects to other components:
- Called by: phase1/run_train_eval.py main()
- Outputs are used in: Phase 1 report and metrics_summary.json

File: phase1/run_train_eval.py
Function: plot_feature_importance
Purpose:
Visualizes the most influential TF-IDF features learned by the Logistic Regression model.
Input:
- clf (TextClassifier with a trained model)
- out_dir (string)
- top_n (int)
Output:
- PNG file saved in out_dir (feature_importance.png)
Step-by-step logic:
1. Extracts TF-IDF feature names from vectorizer.
2. Computes mean absolute coefficients over classes.
3. Plots the top-N coefficients as a horizontal bar chart.
Why this function exists:
Explainability matters in education tools; this plot shows which words influence predictions.
How it connects to other components:
- Called by: phase1/run_train_eval.py main()
- Depends on: phase1/grading/model.py having a trained LogReg model and vectorizer

File: phase1/run_train_eval.py
Function: plot_accuracy_comparison
Purpose:
Compares accuracy and macro-F1 across test splits in one chart.
Input:
- results (list of dicts with split/accuracy/f1_macro)
- out_dir (string)
Output:
- PNG file saved in out_dir (accuracy_comparison.png)
Step-by-step logic:
1. Extracts split names and scores.
2. Draws grouped bars for accuracy and F1.
3. Saves the plot.
Why this function exists:
It makes it easy to see generalization differences between UA, UQ, and UD splits.
How it connects to other components:
- Called by: phase1/run_train_eval.py main()
- Uses metrics computed from predictions produced by phase1/grading/model.py

File: phase1/run_train_eval.py
Function: main
Purpose:
Full Phase 1 training + evaluation pipeline that generates both model artifacts and report-ready evaluation artifacts.
Input:
- None (uses SciEntsBank download and config constants)
Output:
- Saved model artifacts
- Saved evaluation plots and CSVs under phase1/evaluation/
- metrics_summary.json
Step-by-step logic:
1. Loads dataset and converts labels.
2. Prepares train data and trains TextClassifier.
3. Saves artifacts.
4. Generates label distribution plot.
5. For each test split, predicts, computes metrics, saves CSVs and confusion matrices.
6. Generates feature importance plot and accuracy comparison plot.
7. Writes a JSON metrics summary for the report.
Why this function exists:
Academic deliverables (Phase 1 report) need plots/CSVs/JSON, not just console output.
How it connects to other components:
- Uses: phase1/utils.py and phase1/grading/model.py
- Produces: artifacts used by phase1/api/main.py if you run it as a service

File: phase1/backend/index.js
Function: (module startup)
Purpose:
Phase 1 snapshot of the backend gateway (same job as backend/index.js).
Input:
- PORT and ML_SERVICE_URL env vars
Output:
- Express server with the same /api endpoints
Step-by-step logic:
1. Same as backend/index.js: middleware, SQLite table, route handlers, listen().
Why this function exists:
Phase folders often include self-contained submission snapshots; this copy ensures Phase 1 can be evaluated in isolation.
How it connects to other components:
- Duplicates: backend/index.js behavior for Phase 1 packaging

File: phase1/frontend/src/App.jsx
Function: App
Purpose:
Phase 1 snapshot of the React UI (same job as frontend/src/App.jsx).
Input:
- No props
Output:
- UI tabs and panels for grading/history/analytics
Step-by-step logic:
1. Same as frontend/src/App.jsx.
Why this function exists:
Keeps Phase 1 submission self-contained.
How it connects to other components:
- Calls: the same backend endpoints (/api/grade, /api/ocr, /api/history, /api/analytics)

File: phase2/config.py
Function: (configuration constants)
Purpose:
Defines Phase 2 settings: artifact paths for Model A, hybrid alpha, similarity thresholds, and TrOCR model name.
Input:
- Imported by phase2/api/main.py and phase2/run_train_eval.py
Output:
- Constant values used by Phase 2 training and runtime
Step-by-step logic:
1. Builds MODEL_A_PATH using an absolute path based on __file__.
2. Defines alpha and thresholds used by the hybrid pipeline.
3. Defines TROCR_MODEL used by HandwritingOCR.
Why this function exists:
Phase 2 adds deep models and a hybrid ensemble, so its configuration is different and needs a dedicated module.
How it connects to other components:
- Controls: how HybridGrader weights SVM vs SBERT
- Controls: which TrOCR model is downloaded at runtime

File: phase2/utils.py
Function: load_scientsbank
Purpose:
Phase 2 copy of dataset loading utility (same role as Phase 1).
Input:
- None
Output:
- DatasetDict
Step-by-step logic:
1. Calls load_dataset(config.DATASET_NAME).
Why this function exists:
Phase 2 training/evaluation needs the same dataset and label conversion.
How it connects to other components:
- Called by: phase2/run_train_eval.py

File: phase2/utils.py
Function: convert_labels
Purpose:
Phase 2 copy of label conversion utility (5-way to 3-way).
Input:
- dataset, scheme
Output:
- DatasetDict with 3-way labels
Step-by-step logic:
1. Aligns labels with mapping and casts ClassLabel.
Why this function exists:
Hybrid models still need consistent labels to compare outputs fairly.
How it connects to other components:
- Feeds: y labels used by ClassicalGrader training and evaluation

File: phase2/utils.py
Function: get_jaccard_similarity
Purpose:
Phase 2 copy: overlap feature helper (used mainly in analysis parity).
Input:
- two strings
Output:
- float in [0, 1]
Step-by-step logic:
1. Tokenize, stopword-filter, compute Jaccard overlap.
Why this function exists:
Provides consistent feature tools across phases and scripts.
How it connects to other components:
- Used by: any Phase 2 analysis scripts that reuse overlap measures

File: phase2/utils.py
Function: get_token_density
Purpose:
Phase 2 copy: coverage feature helper.
Input:
- student, reference strings
Output:
- float in [0, 1]
Step-by-step logic:
1. Tokenize and compute intersection/reference ratio.
Why this function exists:
Same reason as Phase 1: interpretable overlap/coverage measures are useful for analysis and comparisons.
How it connects to other components:
- Used by: analysis scripts (if extended) and consistent tooling across phases

File: phase2/utils.py
Function: prepare_dataframe
Purpose:
Phase 2 copy: returns aligned data series for training and evaluation.
Input:
- dataset, split
Output:
- (X_train, X_test, y_train, y_test, train_ref, test_ref)
Step-by-step logic:
1. Converts HF splits to pandas and selects the key columns.
Why this function exists:
Prevents duplication and keeps evaluation scripts consistent.
How it connects to other components:
- Called by: phase2/run_train_eval.py

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.__init__
Purpose:
Initializes Model A: TF-IDF vectorizer + SVM classifier (probability=True).
Input:
- None
Output:
- A ClassicalGrader instance
Step-by-step logic:
1. Builds a TF-IDF vectorizer (unigrams + bigrams, English stopwords).
2. Builds an RBF-kernel SVC with probability estimates enabled.
Why this function exists:
Phase 2 uses SVM as a strong non-linear baseline for short-answer grading.
How it connects to other components:
- Used by: phase2/run_train_eval.py (training and per-sample predictions)
- Used by: HybridGrader as the "Model A" probability provider

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.train
Purpose:
Fits the TF-IDF vectorizer and SVM model.
Input:
- X (iterable of text)
- y (iterable of integer labels)
Output:
- A trained vectorizer and SVM stored in the instance
Step-by-step logic:
1. Fits TF-IDF on X and transforms to a matrix.
2. Fits the SVM on the TF-IDF matrix and labels.
Why this function exists:
This is the training step for Model A, which later supports both standalone evaluation and the hybrid ensemble.
How it connects to other components:
- Called by: phase2/run_train_eval.py main()
- Output saved by: ClassicalGrader.save for runtime reuse in HybridGrader

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.predict
Purpose:
Predicts a single label index using the trained SVM.
Input:
- text (string)
Output:
- label_idx (int)
Step-by-step logic:
1. Transforms text using the trained TF-IDF vectorizer.
2. Runs model.predict and returns the first element.
Why this function exists:
Evaluation and hybrid pipelines need a per-example prediction option.
How it connects to other components:
- Used by: phase2/run_train_eval.py to compute metrics for Model A

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.predict_probs
Purpose:
Returns class probability estimates for a single text using the SVM.
Input:
- text (string)
Output:
- probs (numpy array length 3): probabilities for each class
Step-by-step logic:
1. Transforms text with TF-IDF.
2. Calls model.predict_proba and returns the first row.
Why this function exists:
The hybrid ensemble needs probabilities (not just a label) so it can combine them with semantic signals.
How it connects to other components:
- Called by: HybridGrader.grade to compute probs_a

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.save
Purpose:
Serializes the Model A vectorizer and SVM so they can be loaded later.
Input:
- path (string file path)
Output:
- A joblib file written at path containing {"v": vectorizer, "m": model}
Step-by-step logic:
1. Creates parent directories for path if needed.
2. Dumps the vectorizer and model dict with joblib.
Why this function exists:
Training SVM is slow; saving lets Phase 2 API start without retraining.
How it connects to other components:
- Called by: phase2/run_train_eval.py to create phase2/data/artifacts/model_a.pkl
- Loaded by: ClassicalGrader.load when starting the Phase 2 API

File: phase2/grading/classical_grader.py
Function: ClassicalGrader.load
Purpose:
Loads a saved Model A artifact file.
Input:
- path (string)
Output:
- The instance updated with loaded vectorizer and model
Step-by-step logic:
1. Reads joblib file and assigns vectorizer/model fields.
Why this function exists:
HybridGrader and the Phase 2 API need a trained SVM without re-running training.
How it connects to other components:
- Called by: HybridGrader.load during Phase 2 API startup

File: phase2/grading/semantic_scorer.py
Function: SemanticScorer.__init__
Purpose:
Initializes Model B: a SentenceTransformer model used to compute semantic similarity.
Input:
- model_name (string, default "all-MiniLM-L6-v2")
Output:
- A SemanticScorer instance ready to score pairs of sentences
Step-by-step logic:
1. Chooses device "cuda" if available, else "cpu".
2. Loads a SentenceTransformer model and moves it to that device.
Why this function exists:
Keyword overlap is not enough for semantic grading. SBERT provides meaning-aware embeddings.
How it connects to other components:
- Used by: phase2/run_train_eval.py (Model B and Model C scoring)
- Used by: HybridGrader as the semantic component

File: phase2/grading/semantic_scorer.py
Function: SemanticScorer.score
Purpose:
Computes cosine similarity between student and reference embeddings.
Input:
- student_answer (string)
- reference_answer (string)
Output:
- cos_sim (float; cosine similarity, typically in [-1, 1])
Step-by-step logic:
1. Encodes student and reference text into embedding tensors.
2. Computes cosine similarity with util.cos_sim.
3. Converts the result to a Python float.
Why this function exists:
It provides a single numeric "meaning similarity" signal that classical models often miss.
How it connects to other components:
- Called by: SemanticScorer.grade and HybridGrader.grade
- The returned value becomes HybridGrader's similarity_score in the Phase 2 API

File: phase2/grading/semantic_scorer.py
Function: SemanticScorer.grade
Purpose:
Turns the similarity score into a heuristic 3-way label for evaluation comparisons.
Input:
- student_answer (string)
- reference_answer (string)
Output:
- label_idx (int): 0 correct, 2 partial, 1 incorrect
Step-by-step logic:
1. Computes similarity score.
2. Compares against fixed thresholds (0.75 and 0.45).
3. Returns a label index based on the band.
Why this function exists:
It gives a "pure semantic" baseline (Model B) so you can compare it against SVM and the hybrid ensemble.
How it connects to other components:
- Used by: phase2/run_train_eval.py to compute metrics for Model B

File: phase2/grading/hybrid_grader.py
Function: HybridGrader.__init__
Purpose:
Initializes Model C: a hybrid ensemble of Model A (SVM probabilities) and Model B (SBERT similarity).
Input:
- alpha (float, default 0.4): weight for SVM probabilities
Output:
- A HybridGrader instance
Step-by-step logic:
1. Creates a ClassicalGrader (Model A) and a SemanticScorer (Model B).
2. Stores alpha for weighted combination.
Why this function exists:
SVM is good at keyword precision; SBERT is good at meaning. Combining them often improves robustness.
How it connects to other components:
- Used by: phase2/api/main.py to grade requests at runtime
- Evaluated by: phase2/run_train_eval.py as Model C

File: phase2/grading/hybrid_grader.py
Function: HybridGrader.load
Purpose:
Loads the trained Model A artifact into the hybrid grader so it can produce probabilities.
Input:
- model_a_path (string)
Output:
- HybridGrader with model_a ready for predict_probs()
Step-by-step logic:
1. Calls self.model_a.load(model_a_path).
Why this function exists:
The hybrid grader cannot run without a trained SVM. This provides a single, clear loading step.
How it connects to other components:
- Called by: phase2/api/main.py during startup
- Path provided by: phase2/config.py MODEL_A_PATH

File: phase2/grading/hybrid_grader.py
Function: HybridGrader.grade
Purpose:
Computes hybrid predictions by combining SVM probability distribution with SBERT similarity-based pseudo-probabilities.
Input:
- student_answer (string)
- reference_answer (string)
Output:
- dict with:
  - label_idx (int)
  - confidence (float)
  - similarity_score (float; SBERT cosine similarity)
  - components (debug info: svm_probs and sbert_sim)
Step-by-step logic:
1. Gets probs_a = model_a.predict_probs(student_answer).
2. Gets score_b = model_b.score(student_answer, reference_answer).
3. Converts score_b to a heuristic 3-class probability vector probs_b.
4. Computes final_probs = alpha * probs_a + (1 - alpha) * probs_b.
5. Picks label_idx = argmax(final_probs) and confidence = max(final_probs).
6. Returns label_idx, confidence, similarity_score, and component breakdown.
Why this function exists:
It is the core Phase 2 innovation: a simple ensemble that tries to use the best of both keyword-based and meaning-based grading.
How it connects to other components:
- Called by: phase2/api/main.py POST /predict
- Evaluated by: phase2/run_train_eval.py as "Model C (Hybrid)"

File: phase2/ocr/trocr_engine.py
Function: HandwritingOCR.__init__
Purpose:
Loads the TrOCR transformer model and processor used for Phase 2 neural OCR.
Input:
- model_name (string, default "microsoft/trocr-base-handwritten")
Output:
- A HandwritingOCR instance ready for transcribe()
Step-by-step logic:
1. Picks device "cuda" if available, else "cpu".
2. Loads TrOCRProcessor and VisionEncoderDecoderModel from Hugging Face.
3. Moves the model to the chosen device.
Why this function exists:
Tesseract can struggle with messy handwriting. TrOCR is a modern neural OCR model designed for handwriting.
How it connects to other components:
- Used by: phase2/api/main.py /ocr endpoint
- Downloads: model weights at runtime if not cached locally

File: phase2/ocr/trocr_engine.py
Function: HandwritingOCR.transcribe
Purpose:
Runs TrOCR inference on an image and returns the decoded text.
Input:
- image (PIL.Image)
Output:
- transcribed text (string)
Step-by-step logic:
1. Converts image to RGB if needed.
2. Uses TrOCRProcessor to create normalized pixel tensors.
3. Runs model.generate() to produce token ids.
4. Decodes ids into a string and trims whitespace.
Why this function exists:
It is the Phase 2 OCR pipeline that converts handwriting images into text for semantic grading.
How it connects to other components:
- Called by: phase2/api/main.py ocr() handler
- The output can be used as student_answer for /predict

File: phase2/api/main.py
Function: GradeRequest (Pydantic model)
Purpose:
Defines the expected request body for Phase 2 /predict.
Input:
- JSON:
  - student_answer (string)
  - reference_answer (string)
Output:
- Validated GradeRequest object
Step-by-step logic:
1. FastAPI/Pydantic validates input types and required fields.
Why this function exists:
It prevents runtime model errors due to missing fields.
How it connects to other components:
- Used by: phase2/api/main.py predict()
- Similar to: Phase 1 request schema, but Phase 2 omits question in its model input

File: phase2/api/main.py
Function: root
Purpose:
Health endpoint for Phase 2 API.
Input:
- None
Output:
- JSON status and list of supported modes
Step-by-step logic:
1. Returns a static JSON dict.
Why this function exists:
Lets you verify Phase 2 service is up before pointing the backend at it.
How it connects to other components:
- Used during setup when exporting ML_SERVICE_URL to :8001

File: phase2/api/main.py
Function: predict
Purpose:
Grades an answer using the Phase 2 HybridGrader and returns label/confidence/similarity/feedback.
Input:
- GradeRequest with student_answer and reference_answer
Output:
- JSON:
  - predicted_label (string)
  - confidence (float)
  - similarity_score (float)
  - feedback (string)
Step-by-step logic:
1. Calls grader.grade(student_answer, reference_answer).
2. Maps label_idx to label string.
3. Adds a Phase 2-specific feedback string for each class.
4. Returns the response JSON.
Why this function exists:
Defines the Phase 2 grading contract. It is the "hybrid semantic grading" endpoint.
How it connects to other components:
- Can be called by: backend/index.js if ML_SERVICE_URL points to Phase 2 service
- Uses: phase2/grading/hybrid_grader.py to combine SVM and SBERT signals

File: phase2/api/main.py
Function: ocr
Purpose:
Transcribes handwriting using TrOCR and returns the transcribed text.
Input:
- UploadFile "file"
Output:
- JSON { transcribed_text }
Step-by-step logic:
1. Reads file bytes and opens it with PIL.
2. Calls ocr_engine.transcribe(image).
3. Returns the text.
Why this function exists:
Phase 2 adds a stronger neural OCR pipeline for messy handwriting.
How it connects to other components:
- Can be called by: backend/index.js POST /api/ocr if ML_SERVICE_URL points to this service
- Often used before /predict to generate the student_answer text

File: phase2/run_train_eval.py
Function: plot_model_comparison
Purpose:
Creates a grouped bar chart comparing macro-F1 across models (A/B/C) and splits.
Input:
- all_results (list of dicts)
- out_dir (string)
Output:
- model_comparison.png saved to out_dir
Step-by-step logic:
1. Converts results list to a DataFrame.
2. Iterates models and plots F1 values grouped by split.
3. Saves the figure.
Why this function exists:
Phase 2 is about comparing multiple approaches; this chart summarizes performance clearly.
How it connects to other components:
- Called by: phase2/run_train_eval.py main()
- Uses results generated by: ClassicalGrader, SemanticScorer, HybridGrader

File: phase2/run_train_eval.py
Function: main
Purpose:
Runs the complete Phase 2 evaluation pipeline: trains Model A, evaluates Models A/B/C on splits, calibrates alpha, and saves artifacts/plots/CSVs.
Input:
- None
Output:
- phase2/data/artifacts/model_a.pkl
- CSVs/PNGs/metrics_summary.json under phase2/evaluation/
Step-by-step logic:
1. Loads dataset and converts labels.
2. Trains ClassicalGrader (Model A) on train split; saves model_a.pkl.
3. Initializes SemanticScorer (Model B) and HybridGrader (Model C) and loads model_a.pkl.
4. For each test split:
   - For each example: compute Model A label, Model B heuristic label, Model C label.
   - Compute metrics and save classification report CSVs.
   - Save confusion matrices for test_ua.
5. Sweeps alpha values to find best macro-F1 on a validation set and saves alpha_calibration.png.
6. Generates model comparison chart and writes ablation CSV + metrics_summary.json.
Why this function exists:
Phase 2 is a research-style pipeline. The report needs artifacts showing ablations, calibration, and comparisons across models.
How it connects to other components:
- Produces: Model A artifact used by phase2/api/main.py at runtime
- Evaluates: all three graders that make up Phase 2's "hybrid" story

File: notebooks/eda.py
Function: perform_eda
Purpose:
Performs exploratory data analysis on the SciEntsBank dataset and saves plots used to justify feature choices.
Input:
- dataset (DatasetDict)
Output:
- Printed summary stats
- Saved PNG plots (label distribution, text length plots, etc.)
Step-by-step logic:
1. Converts train split to a pandas DataFrame.
2. Prints dataset shape and label counts.
3. Saves label distribution plot.
4. Computes text length distribution and saves histogram.
5. Computes class-wise text length and saves boxplot.
6. Computes word counts, duplicates, split sizes, and uniqueness ratios.
Why this function exists:
Before training, you want to understand the dataset (imbalance, answer length variation, distribution shifts). This supports technically sound feature engineering.
How it connects to other components:
- Uses: ml-service/utils.py functions for dataset load/label conversion and similarity helpers
- Informs: why Phase 1 includes length and overlap engineered features

File: notebooks/eda.py
Function: (script execution block)
Purpose:
Runs EDA and then builds a small feature matrix to visualize the "semantic manifold" via PCA.
Input:
- No CLI args; downloads dataset
Output:
- pca_visual_proof.png and printed completion message
Step-by-step logic:
1. Loads and converts dataset labels.
2. Calls perform_eda(ds).
3. Samples the first N training rows and computes [jaccard, density, len_ratio] per row.
4. Scales features and runs PCA to 2D.
5. Saves a scatter plot colored by class.
Why this function exists:
It provides a visual justification that engineered features can separate classes, which supports the project’s modeling choices.
How it connects to other components:
- Reuses: the same similarity helpers used by TextClassifier.extract_features
- Supports: the report narrative (EDA -> features -> model)
