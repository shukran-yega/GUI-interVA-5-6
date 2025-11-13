from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict, Optional
import pandas as pd
import io
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
    print("✓ vman3 imported successfully")
except ImportError as e:
    print(f"❌ Failed to import vman3: {e}")
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
    
    print(f"→ Session {session_id} cancelled and cleaned up")
    
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
        
        # Send progress update via SSE
        await session["sse_queue"].put({
            "type": "chunk_received",
            "message": f"Received chunk {chunk_index + 1}/{total_chunks}",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        })
        
        print(f"→ Session {session_id}: Received chunk {chunk_index + 1}/{total_chunks}")
        
        # Check if all chunks received
        if len(session["chunks"]) == total_chunks:
            print(f"→ Session {session_id}: All chunks received, combining...")
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
            
            # Parse CSV data
            lines = combined_data.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            if len(lines) < 2:
                await session["sse_queue"].put({
                    "type": "error",
                    "message": "CSV must have at least header and one data row"
                })
                raise HTTPException(status_code=400, detail="Invalid CSV data")
            
            # Parse CSV
            csv_data = [parse_csv_line(line) for line in lines]
            
            await session["sse_queue"].put({
                "type": "progress",
                "message": f"Parsed {len(csv_data) - 1} data rows with {len(csv_data[0])} columns"
            })
            
            # Process based on algorithm - Create background task
            algorithm_lower = session["algorithm"].lower()
            if "interva-6" in algorithm_lower or "interva6" in algorithm_lower:
                # Create async task for processing
                task = asyncio.create_task(
                    process_vman3_interva6(csv_data, session_id, session["who_version"])
                )
                session["task"] = task
                return {"status": "processing", "session_id": session_id}
            elif "interva-5" in algorithm_lower or "interva5" in algorithm_lower:
                # Create async task for processing
                task = asyncio.create_task(
                    process_vman3_interva5(csv_data, session_id, session["who_version"])
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
        print(f"❌ Error in upload_chunk: {str(e)}")
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
    
    # Clean up session after download
    del sessions[session_id]
    
    return StreamingResponse(
        io.BytesIO(result_blob),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=interva6_results.csv",
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
    Monitor and emit progress updates during InterVA analysis.
    Emits periodic progress messages while processing.
    """
    try:
        # Simulate progress updates every 2 seconds
        progress_points = [8, 15, 23, 31, 38, 46, 54, 62, 69, 77, 85, 92, 100]
        
        for percent in progress_points:
            # Check if cancelled
            if session.get("cancelled", False):
                return
            
            await asyncio.sleep(2)  # Wait 2 seconds between updates
            
            # Check again after sleep
            if session.get("cancelled", False):
                return
            
            await queue.put({
                "type": "progress",
                "message": f"{algorithm} processing... {percent}% completed"
            })
            
    except asyncio.CancelledError:
        # Task was cancelled, processing is complete
        pass


async def process_vman3_interva6(csv_data: List[List[Any]], session_id: str, who_version: str):
    """
    Process WHO VA data → pycrossva → InterVA6 with SSE progress updates.
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
                "message": f"✓ Detected: {detected_version}"
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
            "message": f"Transforming data ({input_format} → {output_format})..."
        })
        
        ccva_df = vman3.pycrossva(
            input_data=input_df,
            input_format=input_format,
            output_format=output_format,
            raw_data_id="instanceID",
            lower=True,
            verbose=0  # Suppress console output
        )
        
        await queue.put({
            "type": "progress",
            "message": f"✓ Transformation complete: {ccva_df.shape[0]} rows, {ccva_df.shape[1]} columns"
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
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(executor, vman3.interva6, ccva_df)
        
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
            "message": f"✓ Analysis complete! Generated {results_df.shape[0]} result rows"
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
        
        print(f"✓ InterVA6 analysis complete for session {session_id}")
        
    except asyncio.CancelledError:
        print(f"→ Task cancelled for session {session_id}")
        await queue.put({
            "type": "error",
            "message": "Operation cancelled"
        })
    except Exception as e:
        print(f"❌ Error in process_vman3_interva6: {str(e)}")
        import traceback
        traceback.print_exc()
        
        await queue.put({
            "type": "error",
            "message": f"Processing error: {str(e)}"
        })


async def process_vman3_interva5(csv_data: List[List[Any]], session_id: str, who_version: str):
    """
    Process WHO VA data → pycrossva → InterVA5 with SSE progress updates.
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
                "message": f"✓ Detected: {detected_version}"
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
                    "message": "⚠ Warning: InterVA5 works best with WHO 2016 data"
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
            "message": f"Transforming data ({input_format} → {output_format})..."
        })
        
        ccva_df = vman3.pycrossva(
            input_data=input_df,
            input_format=input_format,
            output_format=output_format,
            raw_data_id="instanceID",
            lower=True,
            verbose=0
        )
        
        await queue.put({
            "type": "progress",
            "message": f"✓ Transformation complete: {ccva_df.shape[0]} rows, {ccva_df.shape[1]} columns"
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
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(executor, vman3.interva5, ccva_df)
        
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
            "message": f"✓ Analysis complete! Generated {results_df.shape[0]} result rows"
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
        
        print(f"✓ InterVA5 analysis complete for session {session_id}")
        
    except asyncio.CancelledError:
        print(f"→ Task cancelled for session {session_id}")
        await queue.put({
            "type": "error",
            "message": "Operation cancelled"
        })
    except Exception as e:
        print(f"❌ Error in process_vman3_interva5: {str(e)}")
        import traceback
        traceback.print_exc()
        
        await queue.put({
            "type": "error",
            "message": f"Processing error: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)