const express = require('express');
const cors = require('cors');
const Database = require('better-sqlite3');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());

// Database Setup
const db = new Database('database.sqlite');
db.exec(`
  CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    reference_answer TEXT,
    student_answer TEXT,
    predicted_label TEXT,
    similarity_score REAL,
    confidence REAL,
    feedback TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Multer for OCR Uploads
const upload = multer({ storage: multer.memoryStorage() });

// --------------------------------------------------------------------------
// Endpoints
// --------------------------------------------------------------------------

/**
 * Grade a student answer by proxying to the ML service.
 * Saves results to SQLite for persistence and analytics.
 */
app.post('/api/grade', async (req, res) => {
  try {
    const { question, student_answer, reference_answer } = req.body;
    
    if (!question || !student_answer || !reference_answer) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Proxy request to ML service
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, {
      question,
      student_answer,
      reference_answer
    });

    const result = response.data;

    // Persist to database
    const insert = db.prepare(`
      INSERT INTO submissions (
        question, reference_answer, student_answer, 
        predicted_label, similarity_score, confidence, feedback
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    insert.run(
      question,
      reference_answer,
      student_answer,
      result.predicted_label,
      result.similarity_score,
      result.confidence,
      result.feedback
    );

    res.json(result);
  } catch (error) {
    console.error('Grading error:', error.message);
    res.status(500).json({ error: 'Failed to grade answer' });
  }
});

/**
 * Perform OCR on an uploaded handwriting image.
 * Proxies the file to the ML service's Tesseract pipeline.
 */
app.post('/api/ocr', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    const response = await axios.post(`${ML_SERVICE_URL}/ocr`, form, {
      headers: { ...form.getHeaders() }
    });

    res.json(response.data);
  } catch (error) {
    console.error('OCR error:', error.message);
    res.status(500).json({ error: 'OCR processing failed' });
  }
});

/**
 * Retrieve submission history.
 */
app.get('/api/history', (req, res) => {
  try {
    const rows = db.prepare('SELECT * FROM submissions ORDER BY created_at DESC LIMIT 50').all();
    res.json(rows);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

/**
 * Retrieve aggregate analytics.
 */
app.get('/api/analytics', (req, res) => {
  try {
    const total = db.prepare('SELECT COUNT(*) as count FROM submissions').get();
    const avgSim = db.prepare('SELECT AVG(similarity_score) as avg FROM submissions').get();
    const distribution = db.prepare('SELECT predicted_label, COUNT(*) as count FROM submissions GROUP BY predicted_label').all();

    const distMap = {
      'correct': 0,
      'partially correct': 0,
      'incorrect': 0
    };
    distribution.forEach(row => {
      distMap[row.predicted_label] = row.count;
    });

    res.json({
      total_submissions: total.count,
      average_similarity: avgSim.avg || 0,
      distribution: distMap
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch analytics' });
  }
});

// Health check
app.get('/', (req, res) => {
  res.send('Automated EdTech Backend Status: Online');
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});