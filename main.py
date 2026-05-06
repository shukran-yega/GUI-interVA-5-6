from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict, Optional
import pandas as pd
import io
import csv
import sys
import os
import asyncio
import uuid
from collections import defaultdict
from datetime import datetime

# Add the vman3 module to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vman3'))

# Import vman3 functions
try:
    import vman3
    print("[OK] vman3 imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import vman3: {e}")
    vman3 = None

app = FastAPI(title="InterVA Analysis API", version="1.0.0")

# Add CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for chunks and SSE queues
sessions: Dict[str, Dict] = {}
# Structure: {session_id: {"chunks": {chunk_index: data}, "total_chunks": N, "algorithm": str, "who_version": str, "sse_queue": asyncio.Queue, "cancelled": bool, "task": Optional[asyncio.Task]}}

class DataPayload(BaseModel):
    algorithm: str
    csv: List[List[Any]]  # CSV data as nested list

class ChunkPayload(BaseModel):
    session_id: str
    chunk_index: int
    total_chunks: int
    algorithm: str
    who_version: str  # WHO version: "auto", "2016WHOv151", or "2022WHOv0101"
    id_column: str = "instanceID"  # Column name used as unique record identifier
    data: str  # Base64 or plain text chunk


@app.get("/")
async def home():
    """Serve the HTML interface"""
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    
    if os.path.exists(html_file):
        return FileResponse(html_file)
    else:
        return {
            "message": "InterVA Analysis API",
            "status": "running",
            "instructions": "Create an index.html file in the same directory as main.py"
        }


@app.get("/health")
async def health_check():
    """Health check for monitoring."""
    backend_dir = os.path.dirname(__file__)
    interva6_dir = os.path.join(backend_dir, 'interva6')
    
    # Check if required files exist
    probbase_exists = (
        os.path.exists(os.path.join(interva6_dir, 'probbase2022.csv')) or
        os.path.exists(os.path.join(backend_dir, 'probbase2022.csv'))
    )
    
    interva6_exists = os.path.exists(os.path.join(interva6_dir, '__init__.py'))
    
    return {
        "status": "healthy",
        "service": "InterVA Analysis API",
        "files_status": {
            "probbase2022.csv": "found" if probbase_exists else "missing",
            "interva6_module": "found" if interva6_exists else "missing"
        }
    }


@app.get("/stream/{session_id}")
async def stream_progress(session_id: str):
    """
    Server-Sent Events endpoint for streaming progress updates to the frontend.
    """
    async def event_generator():
        # Create session if it doesn't exist
        if session_id not in sessions:
            sessions[session_id] = {
                "chunks": {},
                "total_chunks": 0,
                "algorithm": "",
                "who_version": "auto",
                "sse_queue": asyncio.Queue(),
                "cancelled": False,
                "task": None
            }
        
        queue = sessions[session_id]["sse_queue"]
        
        try:
            # Send initial connection message
            yield f"data: {{'type': 'connected', 'message': 'Connected to server'}}\n\n"
            
            # Stream messages from the queue
            while True:
                message = await queue.get()
                
                # Check for completion signal
                if message.get("type") == "complete":
                    yield f"data: {str(message)}\n\n"
                    break
                    
                # Send the message
                yield f"data: {str(message)}\n\n"
                
        except asyncio.CancelledError:
            print(f"SSE connection cancelled for session {session_id}")
        finally:
            # Cleanup session after streaming is done
            if session_id in sessions:
                # Don't delete immediately, allow time for final download
                pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/cancel/{session_id}")
async def cancel_session(session_id: str):
    """
    Cancel a processing session and cleanup resources.
    """
    if session_id not in sessions:
        return {"status": "not_found", "message": "Session not found or already completed"}
    
    session = sessions[session_id]
    session["cancelled"] = True
    
    # Cancel the processing task if it exists
    if session.get("task") and not session["task"].done():
        session["task"].cancel()
    
    # Send cancellation message via SSE
    await session["sse_queue"].put({
        "type": "error",
        "message": "Operation cancelled by user"
    })
    
    # Cleanup session
    del sessions[session_id]
    
    print(f"-> Session {session_id} cancelled and cleaned up")
    
    return {"status": "cancelled", "session_id": session_id}


@app.post("/upload-chunk")
async def upload_chunk(payload: ChunkPayload):
    """
    Receive a chunk of CSV data and store it.
    When all chunks are received, automatically combine and process.
    """
    try:
        session_id = payload.session_id
        chunk_index = payload.chunk_index
        total_chunks = payload.total_chunks
        
        # Initialize session if needed
        if session_id not in sessions:
            sessions[session_id] = {
                "chunks": {},
                "total_chunks": total_chunks,
                "algorithm": payload.algorithm,
                "who_version": payload.who_version,
                "id_column": payload.id_column,
                "sse_queue": asyncio.Queue(),
                "cancelled": False,
                "task": None
            }
        
        session = sessions[session_id]
        
        # Check if cancelled
        if session.get("cancelled", False):
            raise HTTPException(status_code=400, detail="Session cancelled")
        
        session["chunks"][chunk_index] = payload.data
        session["total_chunks"] = total_chunks
        session["algorithm"] = payload.algorithm
        session["who_version"] = payload.who_version
        session["id_column"] = payload.id_column
        
        # Send progress update via SSE
        await session["sse_queue"].put({
            "type": "chunk_received",
            "message": f"Received chunk {chunk_index + 1}/{total_chunks}",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        })
        
        print(f"-> Session {session_id}: Received chunk {chunk_index + 1}/{total_chunks}")
        
        # Check if all chunks received
        if len(session["chunks"]) == total_chunks:
            print(f"-> Session {session_id}: All chunks received, combining...")
            await session["sse_queue"].put({
                "type": "progress",
                "message": "All chunks received, combining data..."
            })
            
            # Combine chunks in order
            combined_data = ""
            for i in range(total_chunks):
                if i not in session["chunks"]:
                    raise HTTPException(status_code=400, detail=f"Missing chunk {i}")
                combined_data += session["chunks"][i]
            
            await session["sse_queue"].put({
                "type": "progress",
                "message": "Data combined successfully, starting analysis..."
            })
            
            # Parse CSV data (using csv module for multiline quoted field support)
            reader = csv.reader(io.StringIO(combined_data))
            csv_data = [row for row in reader if any(cell.strip() for cell in row)]

            if len(csv_data) < 2:
                await session["sse_queue"].put({
                    "type": "error",
                    "message": "CSV must have at least header and one data row"
                })
                raise HTTPException(status_code=400, detail="Invalid CSV data")
            
            row_count = len(csv_data) - 1  # Exclude header
            MAX_ROWS = 100000000
            
            if row_count > MAX_ROWS:
                await session["sse_queue"].put({
                    "type": "error",
                    "message": f"CSV exceeds maximum row limit. You uploaded {row_count} rows but the maximum allowed is {MAX_ROWS}."
                })
                del sessions[session_id]
                raise HTTPException(status_code=400, detail=f"CSV exceeds {MAX_ROWS} row limit ({row_count} rows)")
            
            await session["sse_queue"].put({
                "type": "progress",
                "message": f"Parsed {row_count} data rows with {len(csv_data[0])} columns"
            })
            
            # Process based on algorithm - Create background task
            algorithm_lower = session["algorithm"].lower()
            if "interva-6" in algorithm_lower or "interva6" in algorithm_lower:
                task = asyncio.create_task(
                    process_vman3_interva6(csv_data, session_id, session["who_version"], session.get("id_column", "instanceID"))
                )
                session["task"] = task
                return {"status": "processing", "session_id": session_id}
            elif "interva-5" in algorithm_lower or "interva5" in algorithm_lower:
                task = asyncio.create_task(
                    process_vman3_interva5(csv_data, session_id, session["who_version"], session.get("id_column", "instanceID"))
                )
                session["task"] = task
                return {"status": "processing", "session_id": session_id}
            else:
                await session["sse_queue"].put({
                    "type": "error",
                    "message": f"Algorithm {session['algorithm']} not supported"
                })
                raise HTTPException(status_code=400, detail="Unsupported algorithm")
        
        return {
            "status": "chunk_received",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "received_chunks": len(session["chunks"])
        }
        
    except Exception as e:
        print(f"[ERROR] Error in upload_chunk: {str(e)}")
        if session_id in sessions:
            await sessions[session_id]["sse_queue"].put({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-result/{session_id}")
async def download_result(session_id: str):
    """
    Download the processed results for a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if "result" not in session:
        raise HTTPException(status_code=404, detail="Results not ready yet")
    
    result_blob = session["result"]
    
    # Mark as downloaded but don't delete yet (CSMF may still be fetched)
    session["downloaded"] = True
    
    return StreamingResponse(
        io.BytesIO(result_blob),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=interva_results.csv",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


def parse_csv_line(line: str) -> List[str]:
    """Parse a CSV line handling quoted fields."""
    result = []
    current = ''
    in_quotes = False
    
    for i, char in enumerate(line):
        next_char = line[i + 1] if i + 1 < len(line) else None
        
        if char == '"':
            if in_quotes and next_char == '"':
                current += '"'
                continue
            else:
                in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            result.append(current.strip())
            current = ''
        else:
            current += char
    
    result.append(current.strip())
    return result


async def monitor_progress(queue: asyncio.Queue, session: Dict, total_records: int, algorithm: str):
    """
    Emit steady progress updates to keep the UI feeling alive.
    Ticks 1% every 4 seconds up to 90%. The real 'complete' message handles 100%.
    """
    try:
        percent = 0
        
        while percent < 90:
            if session.get("cancelled", False):
                return
            
            await asyncio.sleep(4)
            
            if session.get("cancelled", False):
                return
            
            percent += 1
            
            await queue.put({
                "type": "progress",
                "message": f"{algorithm} processing... {percent}% completed",
                "percent": percent
            })
        
    except asyncio.CancelledError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CSV column sanitization — Kobo Toolbox format support
# ─────────────────────────────────────────────────────────────────────────────
import re as _re

# Kobo plain-English question labels → pyCrossVA source column IDs.
# Case-sensitive map checked FIRST (handles "Age in Days" vs "Age in days").
# Case-insensitive fallback map checked SECOND (handles casing variation
# across different Kobo forms).
# Extend as new Kobo label variants are discovered.

_KOBO_LABEL_MAP_CS: dict = {
    # Age group indicators  (frontend display + pyCrossVA)
    "The deceased person is a Neonate":  "isNeonatal",
    "The deceased person is a Child":    "isChild",
    "The deceased person is an Adult":   "isAdult",
    # Age computation columns  (pyCrossVA source column IDs)
    "Age in Years":                      "ageInYears",
    "Age in Days":                       "ageInDays",
    "Age in days":                       "ageInDaysNeonate",   # note lowercase 'd'
    "Age in Months":                     "ageInMonths",
    "[Enter adult's age in years:]":     "age_adult",
    "[Enter child's age in days:]":      "age_child_days",
    "[Enter child's age in months:]":    "age_child_months",
    "[Enter child's age in years:]":     "age_child_years",
    # WHO detection support — Kobo plain-label columns
    "Interview language":                "language",
}

# Case-insensitive fallback — excludes "age in days" (ambiguous across cases)
_KOBO_LABEL_MAP_CI: dict = {
    "the deceased person is a neonate":  "isNeonatal",
    "the deceased person is a child":    "isChild",
    "the deceased person is an adult":   "isAdult",
    "age in years":                      "ageInYears",
    "age in months":                     "ageInMonths",
    "[enter adult's age in years:]":     "age_adult",
    "[enter child's age in days:]":      "age_child_days",
    "[enter child's age in months:]":    "age_child_months",
    "[enter child's age in years:]":     "age_child_years",
    "interview language":                "language",
}

# Prefix map for Kobo note/instruction columns (WHO detection support).
# These are section-separator fields with long instruction text as headers
# and NO data.  First ~40 chars are stable across Kobo form versions.
_KOBO_PREFIX_MAP: dict = {
    "Some of the following questions may be repet":  "notenarr",
    "Explain to the respondent that the following":  "note_s_s",
    "Unless specified, the following questions on":   "nmh",
    "Civil registration":                             "botecrn",
    "Death certificate with cause of death":         "noteccd",
    "Inform the respondent that the VA interview":   "noteend",
    "[ Inform the respondent that the VA interview": "noteend",
}


def sanitize_column(headers: list, data_rows: list = None) -> dict:
    """
    Detect and normalize Kobo Toolbox CSV export column names.

    Handles three Kobo patterns:
      A) "(IdXXXXX) Question text"  →  "IdXXXXX"
      B) Known plain-English labels →  standard system name (e.g. isNeonatal)
      C) Duplicate column names     →  keep the most-populated column;
                                       suffix remaining occurrences _dup_N

    When data_rows is supplied, "most populated" = column with highest count
    of non-empty, non-"0" values.  When data_rows is None, first occurrence
    is kept (no popularity scoring).

    Returns dict with keys:
        normalized_headers  – list of cleaned names (same length as input)
        detected_format     – "kobo" | "standard"
        suggested_id_column – preferred ID column if Kobo detected, else None
        column_map          – {original_header: new_name} for logging/debug
        dropped_indices     – set of column indices that are duplicates
    """
    kobo_re = _re.compile(r'^\(+(Id\d+[a-zA-Z0-9_]*)\)+', _re.IGNORECASE)

    # ── Step 1: first-pass rename ─────────────────────────────────────────
    first_pass: list = []
    column_map: dict = {}

    for h in headers:
        # Pattern A: (IdXXXXX) Question text
        m = kobo_re.match(h)
        if m:
            new_name = m.group(1)
            column_map[h] = new_name
            first_pass.append(new_name)
            continue

        # Pattern B: known plain-English label
        # Try case-sensitive first (distinguishes "Age in Days" vs "Age in days")
        label_exact = h.strip()
        if label_exact in _KOBO_LABEL_MAP_CS:
            new_name = _KOBO_LABEL_MAP_CS[label_exact]
            column_map[h] = new_name
            first_pass.append(new_name)
            continue
        # Case-insensitive fallback (handles casing variation across forms)
        label_lower = label_exact.lower()
        if label_lower in _KOBO_LABEL_MAP_CI:
            new_name = _KOBO_LABEL_MAP_CI[label_lower]
            column_map[h] = new_name
            first_pass.append(new_name)
            continue

        # Pattern C: prefix matching for Kobo note/instruction columns
        matched_prefix = False
        for prefix, target_name in _KOBO_PREFIX_MAP.items():
            if label_exact.startswith(prefix):
                column_map[h] = target_name
                first_pass.append(target_name)
                matched_prefix = True
                break
        if matched_prefix:
            continue

        # No match: keep as-is (metadata: _uuid, start, end, _id, …)
        first_pass.append(h)

    # ── Detect format ─────────────────────────────────────────────────────
    kobo_hits = sum(1 for orig, renamed in column_map.items() if renamed != orig)
    kobo_ratio = kobo_hits / len(headers) if headers else 0
    detected_format = "kobo" if kobo_ratio >= 0.20 else "standard"

    if detected_format == "standard":
        return dict(
            normalized_headers=headers,
            detected_format="standard",
            suggested_id_column=None,
            column_map={},
            dropped_indices=set()
        )

    # ── Step 2: resolve duplicates (Pattern C) ────────────────────────────
    name_to_indices: dict = {}
    for i, name in enumerate(first_pass):
        name_to_indices.setdefault(name, []).append(i)

    normalized: list = list(first_pass)
    dropped_indices: set = set()

    for name, indices in name_to_indices.items():
        if len(indices) == 1:
            continue  # no conflict

        if data_rows is not None:
            def _score(col_idx: int, _rows=data_rows) -> int:
                count = 0
                for row in _rows:
                    if col_idx < len(row):
                        v = (row[col_idx] or "").strip()
                        if v and v != "0":
                            count += 1
                return count

            scores = [_score(i) for i in indices]
            best_idx = indices[scores.index(max(scores))]
        else:
            best_idx = indices[0]

        for rank, i in enumerate(indices):
            if i != best_idx:
                normalized[i] = f"{name}_dup_{rank}"
                dropped_indices.add(i)

    # ── Step 3: detect suggested ID column ───────────────────────────────
    norm_set = set(normalized)
    suggested = next(
        (c for c in ("instanceID", "_uuid", "_id") if c in norm_set),
        None
    )

    return dict(
        normalized_headers=normalized,
        detected_format=detected_format,
        suggested_id_column=suggested,
        column_map=column_map,
        dropped_indices=dropped_indices
    )
# ─────────────────────────────────────────────────────────────────────────────


async def process_vman3_interva6(csv_data: List[List[Any]], session_id: str, who_version: str, id_column: str = "instanceID"):
    """
    Process WHO VA data -> pycrossva -> InterVA6 with SSE progress updates.
    """
    try:
        if vman3 is None:
            raise ImportError("vman3 module not available")
        
        session = sessions[session_id]
        queue = session["sse_queue"]
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        await queue.put({
            "type": "progress",
            "message": "Starting WHO VA data processing..."
        })
        
        # Convert to DataFrame
        if not csv_data or len(csv_data) < 2:
            raise ValueError("CSV data must have at least 2 rows (header + data)")

        headers = csv_data[0]
        data_rows = csv_data[1:]

        # ── Kobo Toolbox column sanitization ─────────────────────────────
        norm = sanitize_column(headers, data_rows)
        headers = norm["normalized_headers"]
        if norm["detected_format"] == "kobo":
            n_renamed = len(norm["column_map"])
            n_dropped = len(norm["dropped_indices"])
            await queue.put({
                "type": "progress",
                "message": (
                    f"[INFO] Sanitizing format - "
                    f"renamed {n_renamed} column(s), "
                    f"resolved {n_dropped} duplicate(s). "
                    f"Suggested ID column: {norm['suggested_id_column']}"
                )
            })
        # ─────────────────────────────────────────────────────────────────

        input_df = pd.DataFrame(data_rows, columns=headers)

        await queue.put({
            "type": "progress",
            "message": f"Loaded {len(data_rows)} records with {len(headers)} columns"
        })

        # Detect or use WHO version
        if who_version == "auto":
            await queue.put({
                "type": "progress",
                "message": "Auto-detecting WHO questionnaire version..."
            })
            
            detected_version = vman3.detectwhoqn(input_df)
            await queue.put({
                "type": "progress",
                "message": f"[OK] Detected: {detected_version}"
            })
            
            # Map detected version to format strings
            if detected_version == "who2016":
                input_format = "2016WHOv151"
                output_format = "InterVA5"
            elif detected_version == "who2022":
                input_format = "2022WHOv0101"
                output_format = "InterVA_2022"
            else:
                raise ValueError(f"Unknown WHO version detected: {detected_version}. Please select version manually.")
        else:
            # Use manual selection
            input_format = who_version
            if "2016" in who_version:
                output_format = "InterVA5"
            elif "2022" in who_version:
                output_format = "InterVA_2022"
            else:
                raise ValueError(f"Invalid WHO version: {who_version}")
            
            await queue.put({
                "type": "progress",
                "message": f"Using selected WHO version: {who_version}"
            })
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Transform using pycrossva
        await queue.put({
            "type": "progress",
            "message": f"Transforming data ({input_format} -> {output_format})..."
        })
        
        ccva_df = vman3.pycrossva(
            input_data=input_df,
            input_format=input_format,
            output_format=output_format,
            raw_data_id=id_column,
            lower=True,
            verbose=0  # Suppress console output
        )
        if ccva_df is None:
            raise ValueError("Transformation returned no data. Check input file and mapping configuration.")
        
        await queue.put({
            "type": "progress",
            "message": f"[OK] Transformation complete: {ccva_df.shape[0]} rows, {ccva_df.shape[1]} columns"
        })
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Run InterVA6
        await queue.put({
            "type": "progress",
            "message": f"Running InterVA-6 analysis on {ccva_df.shape[0]} records..."
        })
        
        # Run in executor to avoid blocking
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        # Create progress monitoring task
        progress_task = asyncio.create_task(
            monitor_progress(queue, session, ccva_df.shape[0], "InterVA-6")
        )
        
        # Run InterVA6 using the class directly to get the object
        def run_interva6_with_object():
            # Import from vman3's interva module
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vman3', 'interva'))
            from interva6 import interva6
            
            interva6_obj = interva6()
            results = interva6_obj.run(
                input_data=ccva_df,
                hiv="h",
                malaria="h",
                covid="v",
                write=False,
                output="extended"
            )
            return interva6_obj, results
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            interva6_obj, results = await loop.run_in_executor(executor, run_interva6_with_object)
        
        session["interva6_obj"] = interva6_obj
        session["algorithm"] = "InterVA-6"
        session["input_df"] = input_df
        session["id_column"] = id_column
        
        # Stop progress monitoring
        progress_task.cancel()
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Extract COD DataFrame
        if isinstance(results, dict) and 'COD' in results:
            results_df = pd.DataFrame(results['COD'])
        elif isinstance(results, pd.DataFrame):
            results_df = results
        else:
            raise ValueError(f"Unexpected result type from InterVA6: {type(results)}")
        
        await queue.put({
            "type": "progress",
            "message": f"[OK] Analysis complete! Generated {results_df.shape[0]} result rows"
        })
        
        # Convert to CSV
        await queue.put({
            "type": "progress",
            "message": "Converting results to CSV format..."
        })
        
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_content = csv_buffer.getvalue()
        result_bytes = csv_content.encode('utf-8')
        
        # Store result
        session["result"] = result_bytes
        
        await queue.put({
            "type": "complete",
            "message": "Analysis complete! Results ready for download.",
            "result_size": len(result_bytes)
        })
        
        print(f"[OK] InterVA6 analysis complete for session {session_id}")
        
    except asyncio.CancelledError:
        print(f"-> Task cancelled for session {session_id}")
        await queue.put({
            "type": "error",
            "message": "Operation cancelled"
        })
    except Exception as e:
        print(f"[ERROR] Error in process_vman3_interva6: {str(e)}")
        import traceback
        traceback.print_exc()
        
        await queue.put({
            "type": "error",
            "message": f"Processing error: {str(e)}"
        })


@app.get("/get-csmf/{session_id}")
async def get_csmf(session_id: str, top: int = 10):
    """
    Get CSMF (Cause-Specific Mortality Fraction) data for a session.
    Returns CSMF broken down by categories:
      - all: entire population
      - male / female: by gender (from Id10019 column in input data)
      - adult / child / neonatal: by age group (from isAdult, isChild, isNeonatal columns in input data)
    
    The demographic columns live in the original input DataFrame (stored as session["input_df"]),
    NOT in the InterVA output. We join them via instanceID (input) -> ID (results).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session = sessions[session_id]
    
    if "interva5_obj" not in session and "interva6_obj" not in session:
        raise HTTPException(status_code=404, detail="CSMF data not available - no InterVA object found")
    
    try:
        algorithm = session.get("algorithm", "Unknown")
        
        # ── Step 1: Build the full COD results DataFrame ─────────────────
        
        if "interva5_obj" in session:
            interva5_obj = session["interva5_obj"]
            csmf_all_series = interva5_obj.get_csmf(top=top, groupcode=False, method="frequency")
            
            if csmf_all_series is None or len(csmf_all_series) == 0:
                return {"categories": {}, "message": "No CSMF data available", "algorithm": algorithm}
            
            csmf_all_dict = csmf_all_series.to_dict()
            
            # Get the COD DataFrame for per-record filtering
            results = interva5_obj.results
            if isinstance(results, dict) and "COD" in results:
                cod_df = pd.DataFrame(results["COD"]) if isinstance(results["COD"], list) else results["COD"].copy()
            else:
                # Fallback: return only "all" if we can't get per-record data
                return {
                    "categories": {"all": {"csmf": csmf_all_dict, "count": len(csmf_all_dict)}},
                    "top": top,
                    "algorithm": algorithm
                }
        
        elif "interva6_obj" in session:
            interva6_obj = session["interva6_obj"]
            
            if not interva6_obj.results or "COD" not in interva6_obj.results:
                return {"categories": {}, "message": "No results found in InterVA-6 object", "algorithm": algorithm}
            
            cod_results = interva6_obj.results["COD"]
            cod_df = pd.DataFrame(cod_results) if isinstance(cod_results, list) else cod_results.copy()
            
            # Compute "all" CSMF the same way as before
            cause1 = cod_df["CAUSE1"]
            valid = cause1[cause1 != " "]
            counts = valid.value_counts()
            total = len(valid)
            
            if total == 0:
                return {"categories": {}, "message": "No valid causes found", "algorithm": algorithm}
            
            csmf_all_dict = (counts / total).head(top).to_dict()
        
        # ── Step 2: Build ID-based demographic lookup from input data ────
        # The demographic columns (Id10019, isAdult, isChild, isNeonatal) are in
        # the original input DataFrame, not the InterVA output. We use instanceID
        # from the input to match against ID in the results.
        
        # Prepare category ID lists
        male_ids = []
        female_ids = []
        adult_ids = []
        child_ids = []
        neonatal_ids = []
        
        input_df = session.get("input_df")
        id_column = session.get("id_column", "instanceID")
        demographics_available = False
        
        if input_df is not None and id_column in input_df.columns:
            demographics_available = True
            
            for _, row in input_df.iterrows():
                rid = str(row.get(id_column, "")).strip()
                if not rid:
                    continue
                
                # Gender classification from Id10019
                gender = str(row.get("Id10019", "")).strip().lower()
                if gender == "male":
                    male_ids.append(rid)
                elif gender == "female":
                    female_ids.append(rid)
                
                # Age group classification from isAdult, isChild, isNeonatal
                # These are the FINAL columns (not the intermediate "1"-suffixed ones)
                is_adult = str(row.get("isAdult", "")).strip()
                is_child = str(row.get("isChild", "")).strip()
                is_neonatal = str(row.get("isNeonatal", "")).strip()
                
                if is_neonatal == "1":
                    neonatal_ids.append(rid)
                elif is_child == "1":
                    child_ids.append(rid)
                elif is_adult == "1":
                    adult_ids.append(rid)
        
        # ── Step 3: Helper to compute CSMF for a subset of IDs ──────────
        
        def compute_csmf_for_ids(id_list, top_n):
            """Given a list of instanceIDs, filter COD results and compute CSMF."""
            if not id_list:
                return {}, 0
            
            id_set = set(id_list)
            filtered = cod_df[cod_df["ID"].isin(id_set)]
            
            if filtered.empty or "CAUSE1" not in filtered.columns:
                return {}, 0
            
            cause1 = filtered["CAUSE1"]
            valid = cause1[cause1 != " "]
            total = len(valid)
            
            if total == 0:
                return {}, 0
            
            counts = valid.value_counts()
            csmf = (counts / total).head(top_n)
            return csmf.to_dict(), total
        
        # ── Step 4: Build categorized response ──────────────────────────
        
        # Count total records for "all"
        all_cause1 = cod_df["CAUSE1"] if "CAUSE1" in cod_df.columns else pd.Series()
        all_valid_count = len(all_cause1[all_cause1 != " "]) if len(all_cause1) > 0 else 0
        
        categories = {
            "all": {
                "csmf": csmf_all_dict,
                "count": all_valid_count,
                "label": "All Population"
            }
        }
        
        if demographics_available:
            # Gender categories
            if male_ids:
                male_csmf, male_count = compute_csmf_for_ids(male_ids, top)
                if male_csmf:
                    categories["male"] = {
                        "csmf": male_csmf,
                        "count": male_count,
                        "label": "Male"
                    }
            
            if female_ids:
                female_csmf, female_count = compute_csmf_for_ids(female_ids, top)
                if female_csmf:
                    categories["female"] = {
                        "csmf": female_csmf,
                        "count": female_count,
                        "label": "Female"
                    }
            
            # Age group categories
            if adult_ids:
                adult_csmf, adult_count = compute_csmf_for_ids(adult_ids, top)
                if adult_csmf:
                    categories["adult"] = {
                        "csmf": adult_csmf,
                        "count": adult_count,
                        "label": "Adult"
                    }
            
            if child_ids:
                child_csmf, child_count = compute_csmf_for_ids(child_ids, top)
                if child_csmf:
                    categories["child"] = {
                        "csmf": child_csmf,
                        "count": child_count,
                        "label": "Child"
                    }
            
            if neonatal_ids:
                neonatal_csmf, neonatal_count = compute_csmf_for_ids(neonatal_ids, top)
                if neonatal_csmf:
                    categories["neonatal"] = {
                        "csmf": neonatal_csmf,
                        "count": neonatal_count,
                        "label": "Neonatal"
                    }
        
        return {
            "categories": categories,
            "top": top,
            "algorithm": algorithm,
            "demographics_available": demographics_available
        }
        
    except Exception as e:
        print(f"[ERROR] Error getting CSMF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting CSMF: {str(e)}")


@app.get("/get-error-log/{session_id}")
async def get_error_log(session_id: str):
    """
    Get error log data for a session.
    Returns the error log content from the InterVA processing.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session = sessions[session_id]
    algorithm = session.get("algorithm", "Unknown")
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_count = 0
        discrepancy_count = 0

        # Handle InterVA-5
        if "interva5_obj" in session:
            interva5_obj = session["interva5_obj"]

            warnings_list = getattr(interva5_obj, 'warnings', [])
            excluded      = getattr(interva5_obj, 'excluded_records', [])
            first_pass    = getattr(interva5_obj, 'first_pass_log', [])
            second_pass   = getattr(interva5_obj, 'second_pass_log', [])
            error_count = len(excluded)
            discrepancy_count = len(first_pass) + len(second_pass)

            lines = [f"Error & warning log built for InterVA5 {timestamp}", "", ""]

            # Column-name mismatch warnings (pre-processing)
            if warnings_list:
                lines.append("Column name warnings:")
                lines.append("")
                lines.extend(warnings_list)
                lines.append("")

            lines.append("The following records are incomplete and excluded from further processing:")
            lines.append("")
            if excluded:
                lines.extend(excluded)
            else:
                lines.append("(none)")
            lines.append("")
            lines.append("The following data discrepancies were identified and handled:")
            lines.append("")
            if first_pass:
                lines.extend(first_pass)
            if second_pass:
                lines.append("")
                lines.append("Second pass")
                lines.append("")
                lines.extend(second_pass)
            if not first_pass and not second_pass:
                lines.append("(none)")

            error_log_content = "\n".join(lines)

        # Handle InterVA-6
        elif "interva6_obj" in session:
            interva6_obj = session["interva6_obj"]

            excluded    = getattr(interva6_obj, 'excluded_records', [])
            first_pass  = getattr(interva6_obj, 'first_pass_log', [])
            second_pass = getattr(interva6_obj, 'second_pass_log', [])
            error_count = len(excluded)
            discrepancy_count = len(first_pass) + len(second_pass)

            lines = [f"Error & warning log built for InterVA6 {timestamp}", "", ""]
            lines.append("The following records are incomplete and excluded from further processing:")
            lines.append("")
            if excluded:
                lines.extend(excluded)
            else:
                lines.append("(none)")
            lines.append("")
            lines.append("The following data discrepancies were identified and handled:")
            lines.append("")
            if first_pass:
                lines.extend(first_pass)
            if second_pass:
                lines.append("")
                lines.append("Second pass")
                lines.append("")
                lines.extend(second_pass)
            if not first_pass and not second_pass:
                lines.append("(none)")

            error_log_content = "\n".join(lines)

        else:
            error_log_content = f"Error Log\nGenerated: {timestamp}\n\nNo InterVA object found in session.\n"

        return {
            "error_log": error_log_content,
            "algorithm": algorithm,
            "error_count": error_count,
            "discrepancy_count": discrepancy_count
        }
        
    except Exception as e:
        print(f"[ERROR] Error getting error log: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting error log: {str(e)}")


@app.post("/cleanup-session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up a session after all downloads are complete.
    """
    if session_id in sessions:
        del sessions[session_id]
        print(f"[OK] Session {session_id} cleaned up")
        return {"status": "cleaned", "session_id": session_id}
    
    return {"status": "not_found", "session_id": session_id}


async def process_vman3_interva5(csv_data: List[List[Any]], session_id: str, who_version: str, id_column: str = "instanceID"):
    """
    Process WHO VA data -> pycrossva -> InterVA5 with SSE progress updates.
    """
    try:
        if vman3 is None:
            raise ImportError("vman3 module not available")
        
        session = sessions[session_id]
        queue = session["sse_queue"]
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        await queue.put({
            "type": "progress",
            "message": "Starting WHO VA data processing..."
        })
        
        # Convert to DataFrame
        if not csv_data or len(csv_data) < 2:
            raise ValueError("CSV data must have at least 2 rows (header + data)")

        headers = csv_data[0]
        data_rows = csv_data[1:]

        # ── Kobo Toolbox column sanitization ─────────────────────────────
        norm = sanitize_column(headers, data_rows)
        headers = norm["normalized_headers"]
        if norm["detected_format"] == "kobo":
            n_renamed = len(norm["column_map"])
            n_dropped = len(norm["dropped_indices"])
            await queue.put({
                "type": "progress",
                "message": (
                    f"[INFO] Sanitizing format - "
                    f"renamed {n_renamed} column(s), "
                    f"resolved {n_dropped} duplicate(s). "
                    f"Suggested ID column: {norm['suggested_id_column']}"
                )
            })
        # ─────────────────────────────────────────────────────────────────

        input_df = pd.DataFrame(data_rows, columns=headers)

        await queue.put({
            "type": "progress",
            "message": f"Loaded {len(data_rows)} records with {len(headers)} columns"
        })

        # Detect or use WHO version (InterVA5 typically uses WHO 2016)
        if who_version == "auto":
            await queue.put({
                "type": "progress",
                "message": "Auto-detecting WHO questionnaire version..."
            })
            
            detected_version = vman3.detectwhoqn(input_df)
            await queue.put({
                "type": "progress",
                "message": f"[OK] Detected: {detected_version}"
            })
            
            # Force WHO 2016 for InterVA5
            input_format = "2016WHOv151"
            output_format = "InterVA5"
        else:
            # Use manual selection
            input_format = who_version
            if "2016" in who_version:
                output_format = "InterVA5"
            else:
                # If 2022 selected but running IV5, show warning
                await queue.put({
                    "type": "progress",
                    "message": "[WARN] Warning: InterVA5 works best with WHO 2016 data"
                })
                input_format = "2016WHOv151"
                output_format = "InterVA5"
            
            await queue.put({
                "type": "progress",
                "message": f"Using format: {input_format}"
            })
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Transform using pycrossva
        await queue.put({
            "type": "progress",
            "message": f"Transforming data ({input_format} -> {output_format})..."
        })
        
        ccva_df = vman3.pycrossva(
            input_data=input_df,
            input_format=input_format,
            output_format=output_format,
            raw_data_id=id_column,
            lower=True,
            verbose=0
        )
        
        await queue.put({
            "type": "progress",
            "message": f"[OK] Transformation complete: {ccva_df.shape[0]} rows, {ccva_df.shape[1]} columns"
        })
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Run InterVA5
        await queue.put({
            "type": "progress",
            "message": f"Running InterVA-5 analysis on {ccva_df.shape[0]} records..."
        })
        
        # Run in executor to avoid blocking
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        # Create progress monitoring task
        progress_task = asyncio.create_task(
            monitor_progress(queue, session, ccva_df.shape[0], "InterVA-5")
        )
        
        # Run InterVA5 using vman3 wrapper (which returns the InterVA5 object with results)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # vman3.interva5 returns the results, but we need the object too
            # We'll call it directly to get the object
            def run_interva5_with_object():
                # Import from vman3's interva module
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vman3', 'interva'))
                from interva5 import InterVA5
                
                interva5_obj = InterVA5(
                    va_input=ccva_df,
                    hiv="h",
                    malaria="h",
                    write=False,
                    return_checked_data=False
                )
                
                results = interva5_obj.run()
                return interva5_obj, results
            
            interva5_obj, results = await loop.run_in_executor(executor, run_interva5_with_object)
        
        session["interva5_obj"] = interva5_obj
        session["algorithm"] = "InterVA-5"
        session["input_df"] = input_df
        session["id_column"] = id_column
        
        # Stop progress monitoring
        progress_task.cancel()
        
        # Check for cancellation
        if session.get("cancelled", False):
            return
        
        # Extract COD DataFrame
        if isinstance(results, dict) and 'COD' in results:
            results_df = pd.DataFrame(results['COD'])
        elif isinstance(results, pd.DataFrame):
            results_df = results
        else:
            raise ValueError(f"Unexpected result type from InterVA5: {type(results)}")
        
        await queue.put({
            "type": "progress",
            "message": f"[OK] Analysis complete! Generated {results_df.shape[0]} result rows"
        })
        
        # Convert to CSV
        await queue.put({
            "type": "progress",
            "message": "Converting results to CSV format..."
        })
        
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_content = csv_buffer.getvalue()
        result_bytes = csv_content.encode('utf-8')
        
        # Store result
        session["result"] = result_bytes
        
        await queue.put({
            "type": "complete",
            "message": "Analysis complete! Results ready for download.",
            "result_size": len(result_bytes)
        })
        
        print(f"[OK] InterVA5 analysis complete for session {session_id}")
        
    except asyncio.CancelledError:
        print(f"-> Task cancelled for session {session_id}")
        await queue.put({
            "type": "error",
            "message": "Operation cancelled"
        })
    except Exception as e:
        print(f"[ERROR] Error in process_vman3_interva5: {str(e)}")
        import traceback
        traceback.print_exc()
        
        await queue.put({
            "type": "error",
            "message": f"Processing error: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)