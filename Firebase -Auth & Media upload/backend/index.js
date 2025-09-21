require('dotenv').config();

const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');
const path = require('path');
const fs = require('fs');

// Allow overriding via env var, otherwise try to auto-detect the service account JSON
const serviceAccountEnv = process.env.SERVICE_ACCOUNT_PATH;
let serviceAccountPath;
if (serviceAccountEnv) {
  serviceAccountPath = path.isAbsolute(serviceAccountEnv) ? serviceAccountEnv : path.join(__dirname, serviceAccountEnv);
} else {
  // look for a file that matches the expected prefix in this directory
  const files = fs.readdirSync(__dirname);
  const match = files.find(f => f.startsWith('genai-973e8-firebase-adminsdk-fbsvc') && f.endsWith('.json'));
  if (match) {
    serviceAccountPath = path.join(__dirname, match);
  }
}

if (!serviceAccountPath || !fs.existsSync(serviceAccountPath)) {
  // give a helpful error message instead of letting firebase-admin throw ENOENT
  console.error('Firebase service account JSON not found. Expected file in backend directory.');
  console.error('Set the correct filename or export SERVICE_ACCOUNT_PATH to point to the file.');
  console.error('Tried:', serviceAccountPath || '<no candidate found>');
  process.exit(1);
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccountPath),
  projectId: process.env.FIREBASE_PROJECT_ID
});

const app = express();
const PORT = 5000;

app.use(cors({
  origin: 'http://localhost:3000'
}));

app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
