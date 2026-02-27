#!/bin/bash
# Starts ngrok tunnel, updates client .env with the new URL, then starts the server.

PORT=3000
ENV_FILE="$(dirname "$0")/../client/.env"

# Check dependencies
if ! command -v ngrok &> /dev/null; then
    echo "Error: ngrok is not installed. Install it with: brew install ngrok"
    exit 1
fi

# Kill any existing ngrok process
pkill -f "ngrok http" 2>/dev/null

# Start ngrok in the background
echo "Starting ngrok tunnel on port $PORT..."
ngrok http $PORT > /dev/null &
NGROK_PID=$!
sleep 2

# Get the public URL from ngrok's local API
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin)['tunnels']
for t in tunnels:
    if t['proto'] == 'https':
        print(t['public_url'])
        break
" 2>/dev/null)

if [ -z "$NGROK_URL" ]; then
    echo "Error: Could not get ngrok URL. Is ngrok running?"
    kill $NGROK_PID 2>/dev/null
    exit 1
fi

echo "ngrok URL: $NGROK_URL"

# Update the EXPO_PUBLIC_FACE_ANALYSIS_URL in client/.env
if [ -f "$ENV_FILE" ]; then
    if grep -q "EXPO_PUBLIC_FACE_ANALYSIS_URL" "$ENV_FILE"; then
        sed -i '' "s|EXPO_PUBLIC_FACE_ANALYSIS_URL=.*|EXPO_PUBLIC_FACE_ANALYSIS_URL=${NGROK_URL}/analyze-face|" "$ENV_FILE"
    else
        echo "EXPO_PUBLIC_FACE_ANALYSIS_URL=${NGROK_URL}/analyze-face" >> "$ENV_FILE"
    fi
    echo "Updated $ENV_FILE with new URL"
else
    echo "Warning: $ENV_FILE not found, skipping .env update"
fi

# Clean up ngrok on exit
cleanup() {
    echo ""
    echo "Shutting down ngrok..."
    kill $NGROK_PID 2>/dev/null
}
trap cleanup EXIT

# Start the server (this blocks until Ctrl+C)
echo "Starting server on port $PORT..."
echo "(Remember to restart Expo with --clear to pick up the new .env)"
echo ""
cd "$(dirname "$0")" && npm start
