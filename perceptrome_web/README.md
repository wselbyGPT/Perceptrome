# Perceptrome Web GUI

Browser UI for running Perceptrome CLI workflows without Qt.

## Run

```bash
cd perceptrome_web/client
npm install
npm run build

cd ..
python perceptrome_web_cli.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Backend wiring

The web backend now executes real Perceptrome commands from the UI over WebSocket:

- `start_run` sends `{ command, cwd }`
- server streams logs and status
- `stop_run` cancels the active command

Train and Generate tabs both send CLI commands (defaulting to `perceptrome ...`).
Qt GUI remains untouched for troubleshooting.
