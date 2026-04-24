import os
import uuid
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure essential directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "data", "vector_db"), exist_ok=True)

# ── Pipeline singleton (lazy-loaded) ─────────────────────────────────────────
_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_error = None


def get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline, None
    if _pipeline_error is not None:
        return None, _pipeline_error
        
    with _pipeline_lock:
        # Check again inside lock
        if _pipeline is not None:
            return _pipeline, None
        if _pipeline_error is not None:
            return None, _pipeline_error
            
        try:
            print("\n" + "="*50)
            print("[App] Initializing RAG Pipeline...")
            print("[App] This may take up to 30s as models are loaded.")
            print("="*50)
            from src.rag_pipeline import RAGPipeline
            _pipeline = RAGPipeline()
            print("[App] Pipeline initialized successfully.")
            return _pipeline, None
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _pipeline_error = str(e)
            print(f"[App] Pipeline initialization failed: {tb}")
            return None, _pipeline_error


# ── Helper ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image(filename: str) -> bool:
    return filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "tiff", "bmp"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/health")
def api_health():
    return jsonify({"status": "healthy", "pipeline_initialized": _pipeline is not None})


@app.route("/api/status", strict_slashes=False)
def api_status():
    pipeline, err = get_pipeline()
    if err:
        return jsonify({"status": "error", "error": err}), 500
    stats = pipeline.get_stats()
    return jsonify({
        "status": "ready",
        "total_chunks": stats["total_chunks"],
        "sources": stats["sources"]
    })


@app.route("/api/upload", methods=["POST"], strict_slashes=False)
def api_upload():
    pipeline, err = get_pipeline()
    if err:
        return jsonify({"success": False, "error": f"Pipeline failed to initialise: {err}"}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    use_ocr = request.form.get("use_ocr", "false").lower() == "true"

    filename = secure_filename(file.filename)
    # Avoid overwriting by prefixing with a short UUID
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    result = pipeline.ingest_document(save_path, filename, use_ocr=use_ocr or is_image(filename))

    # Clean up the saved file after ingestion
    try:
        os.remove(save_path)
    except OSError:
        pass

    if result["success"]:
        stats = pipeline.get_stats()
        return jsonify({
            "success": True,
            "message": f"✅ '{filename}' ingested successfully!",
            "chunks": result["chunks"],
            "characters": result["characters"],
            "total_chunks": stats["total_chunks"],
            "sources": stats["sources"]
        })
    else:
        return jsonify({"success": False, "error": result["error"]}), 422


@app.route("/api/query", methods=["POST"], strict_slashes=False)
def api_query():
    pipeline, err = get_pipeline()
    if err:
        return jsonify({"success": False, "error": f"Pipeline failed to initialise: {err}"}), 500

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty."}), 400

    # Handle custom API key from frontend
    custom_key = request.headers.get("X-API-Key")
    if custom_key and custom_key.startswith("gsk_"):
        pipeline.llm.update_api_key(custom_key)

    top_k = int(data.get("top_k", os.getenv("TOP_K_RESULTS", 5)))
    result = pipeline.query(question, top_k=top_k)

    return jsonify({
        "success": True,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "chunks_used": result.get("chunks_used", 0),
        "retrieved_chunks": [
            {
                "text": c["document"][:300] + ("…" if len(c["document"]) > 300 else ""),
                "source": c["metadata"].get("source", "Unknown"),
                "distance": round(c["distance"], 4)
            }
            for c in result.get("retrieved_chunks", [])
        ],
        "timestamp": datetime.now().strftime("%H:%M")
    })


@app.route("/api/clear", methods=["POST"], strict_slashes=False)
def api_clear():
    pipeline, err = get_pipeline()
    if err:
        return jsonify({"success": False, "error": err}), 500
    pipeline.clear_knowledge_base()
    return jsonify({"success": True, "message": "Knowledge base cleared."})


@app.route("/api/reset-chat", methods=["POST"], strict_slashes=False)
def api_reset_chat():
    pipeline, err = get_pipeline()
    if err:
        return jsonify({"success": False, "error": err}), 500
    pipeline.reset_conversation()
    return jsonify({"success": True, "message": "Conversation history reset."})


@app.errorhandler(404)
def page_not_found(e):
    print(f"[404 Error] Not Found: {request.path}")
    return jsonify({"success": False, "error": f"Path not found: {request.path}"}), 404


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e

    # Handle non-HTTP exceptions
    import traceback
    tb = traceback.format_exc()
    print(f"[500 Error] Unhandled Exception: {tb}")
    return jsonify({
        "success": False, 
        "error": "Internal Server Error",
        "message": str(e)
    }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🚀 Server starting on http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
