import logging
import threading
import requests
from flask import Flask, request, jsonify
from threading import Event
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global state for tracking call status
CALL_STATUS: Dict[str, Dict[str, Any]] = {}
CALL_STATUS_LOCK = threading.Lock()

app = Flask(__name__)

def get_public_url(webhook_port: int) -> str:
    """Get the public ngrok URL for webhooks"""
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        response.raise_for_status()
        tunnels = response.json()

        for tunnel in tunnels["tunnels"]:
            if tunnel["config"]["addr"] == f"http://localhost:{webhook_port}":
                public_url = tunnel["public_url"] + "/webhook"
                return public_url
        raise RuntimeError(f"No ngrok tunnel found pointing to port {webhook_port}")
    
    except requests.RequestException as e:
        logger.error(f"Error connecting to ngrok API: {e}")
        raise RuntimeError("Cannot get ngrok tunnel URL. Make sure ngrok is running!")

def wait_for_call_completion(call_id: str, timeout: int = 200) -> bool:
    """Wait for a call to complete and recording to be available"""
    with CALL_STATUS_LOCK:
        if call_id not in CALL_STATUS:
            CALL_STATUS[call_id] = {
                'status': 'pending',
                'recording_available': False,
                'completed_event': Event(),
                'call_ended': False
            }
        event = CALL_STATUS[call_id]['completed_event']
    
    completed = event.wait(timeout=timeout)
    if not completed:
        logger.warning(f"Timeout waiting for call {call_id} to complete")
    return completed

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    call_id = data.get("id")
    status = data.get("status")
    recording_available = data.get("recording_available", False)
        
    with CALL_STATUS_LOCK:
        if call_id not in CALL_STATUS:
            CALL_STATUS[call_id] = {
                'status': status,
                'recording_available': recording_available,
                'completed_event': Event(),
                'call_ended': False
            }
        else:
            CALL_STATUS[call_id].update({
                'status': status,
                'recording_available': recording_available
            })
            
            # Track call ended status
            if status.lower() in ["event_phone_call_ended", "phone_call_ended"]:
                CALL_STATUS[call_id]['call_ended'] = True
        
            # Set completion event when both call has ended and recording is available
            if CALL_STATUS[call_id].get('call_ended', False) and recording_available:
                CALL_STATUS[call_id]['completed_event'].set()

    if recording_available:
        logger.info(f"Call {call_id} completed with status {status}")
    return jsonify({"success": True}) 