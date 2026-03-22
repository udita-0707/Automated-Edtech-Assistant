const express = require('express');
const cors = require('cors');
const sqlite3 = require('better-sqlite3');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// Initialize SQLite Database
const dbPath = path.join(__dirname, 'database.sqlite');
const db = new sqlite3(dbPath);

// Create submissions table if it doesn't exist
db.exec(`
  CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    student_answer TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    predicted_label TEXT,
    similarity_score REAL,
    length_ratio REAL,
    confidence REAL,
    feedback TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// -------------------------------------------------------------------------- //
// Helpers
// -------------------------------------------------------------------------- //

function generateFeedback(label, similarity, confidence) {
  if (label === 'correct') {
    return 'Excellent work! Your answer is semantically aligned with the reference.';
  } else if (label === 'partially correct') {
    if (similarity > 0.6) {
      return 'You are on the right track, but your answer might be missing key details or is slightly off-topic.';
    } else {
      return 'Some parts are correct, but there is significant room for improvement to fully address the question.';
    }
  } else {
    return 'Your answer does not match the expected reference. Please review the core concepts and try again.';
  }
}

// -------------------------------------------------------------------------- //
// Routes
// -------------------------------------------------------------------------- //

app.post('/api/grade', async (req, res) => {
  const { question, student_answer, reference_answer } = req.body;

  if (!question || !student_answer || !reference_answer) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  try {
    // 1. Call ML Service
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, {
      student_answer,
      reference_answer
    });
    
    const { predicted_label, similarity_score, length_ratio, confidence } = mlResponse.data;
    
    // 2. Generate generic feedback based on label & similarity
    const feedback = generateFeedback(predicted_label, similarity_score, confidence);

    // 3. Save to database
    const stmt = db.prepare(`
      INSERT INTO submissions (
        question, student_answer, reference_answer, 
        predicted_label, similarity_score, length_ratio, confidence, feedback
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    const info = stmt.run(
      question, student_answer, reference_answer,
      predicted_label, similarity_score, length_ratio, confidence, feedback
    );

    // 4. Return to client
    res.json({
      id: info.lastInsertRowid,
      predicted_label,
      similarity_score,
      length_ratio,
      confidence,
      feedback
    });

  } catch (error) {
    console.error('Error grading:', error.message);
    res.status(500).json({ error: 'Failed to grade answer. Is ML service running?' });
  }
});

app.get('/api/history', (req, res) => {
  try {
    const stmt = db.prepare('SELECT * FROM submissions ORDER BY created_at DESC');
    const history = stmt.all();
    res.json(history);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

app.get('/api/analytics', (req, res) => {
  try {
    const defaultStats = {
      total_submissions: 0,
      average_similarity: 0,
      distribution: {
        correct: 0,
        "partially correct": 0,
        incorrect: 0
      }
    };
    
    const totalStmt = db.prepare('SELECT COUNT(*) as count, AVG(similarity_score) as avg_sim FROM submissions');
    const totalResult = totalStmt.get();
    
    if (totalResult.count === 0) {
      return res.json(defaultStats);
    }

    const distStmt = db.prepare('SELECT predicted_label, COUNT(*) as count FROM submissions GROUP BY predicted_label');
    const distResults = distStmt.all();
    
    const distribution = {
      correct: 0,
      "partially correct": 0,
      incorrect: 0
    };
    
    distResults.forEach(row => {
      distribution[row.predicted_label] = row.count;
    });

    res.json({
      total_submissions: totalResult.count,
      average_similarity: totalResult.avg_sim || 0,
      distribution
    });

  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch analytics' });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});
