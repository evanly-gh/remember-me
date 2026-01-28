const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const port = 3000;
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

/** GET endpoint for sending back a Hello World message */
app.get('/hello', (req, res) => {
  res.type('text');
  res.send('Hello, World!');
});

/** POST endpoint to analyze a face using Python and AWS Rekognition */
app.post('/analyze-face', (req, res) => {
  console.log('Received /analyze-face request');
  const { image } = req.body;
  if (!image) {
    return res.status(400).json({ error: 'No image data provided' });
  }

  // Remove the data:image/jpeg;base64, prefix if it exists
  const base64Data = image.replace(/^data:image\/\w+;base64,/, '');

  // Determine which python command to use. 
  // We'll prefer 'python3' but add error handling if it fails.
  // Force arm64 architecture to match the installed libraries (since Node is x86_64/Rosetta)
  const pythonProcess = spawn('arch', [
    '-arm64',
    'python3',
    path.join(__dirname, '../../rekognition/cli_wrapper.py')
  ]);

  let result = '';
  let errorOutput = '';

  // Handle stdin errors (like EPIPE) to prevent the whole Node process from crashing
  pythonProcess.stdin.on('error', (err) => {
    console.error('Child process stdin error:', err.message);
  });

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
    console.error(`Python Stderr: ${data}`);
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python process:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to start analysis engine. Is Python installed?' });
    }
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python script exited with code ${code}. Stderr: ${errorOutput}`);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Internal server error during analysis', detail: errorOutput });
      }
      return;
    }

    try {
      const jsonResponse = JSON.parse(result);
      res.json(jsonResponse);
    } catch (e) {
      console.error('Failed to parse Python output:', result);
      res.status(500).json({ error: 'Invalid response from analysis engine' });
    }
  });

  // Write base64 data to stdin
  pythonProcess.stdin.write(base64Data);
  pythonProcess.stdin.end();
});

// Tells our app to listen on all network interfaces
app.listen(port, '0.0.0.0', () => {
  console.log(`Server is running on http://0.0.0.0:${port}`);
});