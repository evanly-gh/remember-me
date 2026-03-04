const express = require('express');
const cors = require('cors');

const port = 3000;
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Face analysis microservice URL (Python FastAPI on HuggingFace Spaces or local)
const FACE_SERVICE_URL = process.env.FACE_SERVICE_URL || 'http://localhost:8000';

/** GET endpoint for sending back a Hello World message */
app.get('/hello', (req, res) => {
    res.type('text');
    res.send('Hello, World!');
});

/**
 * POST endpoint to analyze a face.
 * Forwards the request to the Python face-service microservice via HTTP
 * instead of spawning a local Python process.
 */
app.post('/analyze-face', async (req, res) => {
    console.log('Received /analyze-face request');
    const { image } = req.body;
    if (!image) {
        return res.status(400).json({ error: 'No image data provided' });
    }

    // Strip data URI prefix if present
    const base64Data = image.replace(/^data:image\/\w+;base64,/, '');

    try {
        // Forward to Python face-service microservice
        const response = await fetch(`${FACE_SERVICE_URL}/analyze-base64`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Face service returned ${response.status}: ${errorText}`);
            return res.status(response.status).json({
                error: 'Face analysis service error',
                detail: errorText,
            });
        }

        const result = await response.json();
        res.json(result);
    } catch (error) {
        console.error('Failed to reach face analysis service:', error.message);
        res.status(503).json({
            error: 'Face analysis service unavailable',
            detail: `Could not connect to ${FACE_SERVICE_URL}. Is the service running?`,
        });
    }
});

// Tells our app to listen on all network interfaces
app.listen(port, '0.0.0.0', () => {
    console.log(`Server is running on http://0.0.0.0:${port}`);
    console.log(`Face service URL: ${FACE_SERVICE_URL}`);
});