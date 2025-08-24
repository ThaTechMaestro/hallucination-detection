"""
Hallucination Judge (Product Docs) — Programmatic API
- No RAG stack required: works on {id, question, context, response}
- Sentence-level scoring via a separate "judge" LLM (OpenAI or Anthropic)
- 0 = grounded, 1 = hallucinated
- Strict JSON output from model, parse & retry once, aggregate, label, classify
"""

from __future__ import annotations
import json, os, re, csv, hashlib, tiktoken
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# -----------------------------
# Sentence splitting (deterministic)
# -----------------------------

_ABBREV = r"(?:e\.g|i\.e|etc|vs|v|Mr|Mrs|Dr|Prof|Inc|Ltd|Co)"
# Use negative lookahead instead of lookbehind
_SENT_SPLIT = re.compile(rf"(?<=[.!?])\s+(?!{_ABBREV}\b)(?=[A-Z(])")

def split_sentences(text: str) -> List[str]:
    """
    Deterministically split a response into "sentences".
    - Split on ., ?, ! with uppercase/paren following
    - Split on newlines; treat bullets as sentences
    - Trim & drop empties
    """
    if not text or not text.strip():
        return []
    parts = _SENT_SPLIT.split(text.strip())
    out: List[str] = []
    for p in parts:
        for ln in (ln.strip() for ln in p.splitlines() if ln.strip()):
            # normalize bullets ("- ", "* ", "1. ")
            ln = re.sub(r"^\s*([*-]|\d+\.)\s+", "", ln)
            if ln:
                out.append(ln)
    return out


# -----------------------------
# Prompting
# -----------------------------

SYSTEM_PROMPT = (
    "You are an evidence checker. Only use the provided CONTEXT. "
    "Do not use outside knowledge. "
    "Return STRICT JSON only, with no commentary."
)

def build_user_prompt(context: str, response: str, sentences: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return f"""CONTEXT:
{context}

RESPONSE:
{response}

SENTENCES (do not alter):
{numbered}

TASK:
For each sentence i, output a hallucination score s_i in [0,1]:
- s_i = 0.0 if the sentence is directly based on the CONTEXT.
- s_i = 1.0 if the sentence is not based on the CONTEXT.
- Use intermediate values when uncertain; higher = more likely hallucinated.

Output STRICT JSON only, exactly in this format:
{{"scores":[s_1, s_2, ..., s_{len(sentences)}]}}

Rules:
- The length of "scores" MUST equal the number of sentences.
- No extra fields. No trailing text.
"""

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Providers (OpenAI / Anthropic)
# -----------------------------

class LLMProvider:
    def __init__(self, model: str):
        self.model = model
    def call(self, system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self._client = OpenAI(api_key=api_key)
    
    def call(self, system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt},
            ],
            timeout=timeout,
        )
        text = resp.choices[0].message.content
        if text is None:
            raise ValueError("OpenAI API returned None content")
        return text.strip()

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self._client = Anthropic(api_key=api_key)
    
    def call(self, system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=system_prompt,
            messages=[{"role":"user","content":user_prompt}],
        )
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_content = getattr(block, "text", None)
                if text_content is not None:
                    parts.append(text_content)
        
        result = "".join(parts).strip()
        if not result:
            raise ValueError("Anthropic API returned empty content")
        return result

def get_provider(provider: str, model: str) -> LLMProvider:
    p = provider.lower().strip()
    if p == "openai": return OpenAIProvider(model)
    if p == "anthropic": return AnthropicProvider(model)
    raise ValueError("provider must be 'openai' or 'anthropic'")


# -----------------------------
# Parse & validation
# -----------------------------

def parse_scores(raw: str, n: int) -> List[float]:
    """
    Expect STRICT JSON object with only {'scores':[...]} of length n, floats in [0,1].
    Raise ValueError if any constraint fails.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from LLM")
    
    try:
        data = json.loads(raw.strip())
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    if not isinstance(data, dict) or set(data.keys()) != {"scores"}:
        raise ValueError("JSON must be an object with exactly one key: 'scores'")
    
    scores = data["scores"]
    if not isinstance(scores, list) or len(scores) != n:
        raise ValueError(f"'scores' must be a list of length {n}, got {len(scores) if isinstance(scores, list) else type(scores)}")
    
    out = []
    for i, x in enumerate(scores, 1):
        try:
            fx = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"Score at position {i} not a float: {x!r}")
        if not (0.0 <= fx <= 1.0):
            raise ValueError(f"Score at position {i} out of range [0,1]: {fx}")
        out.append(fx)
    return out


# -----------------------------
# Labels & aggregation
# -----------------------------

def label_of(s: float) -> str:
    """
    Grounded / Partially Supported / Unsupported / Refuted
    (Tune cuts later on a dev slice.)
    """
    if s <= 0.15: return "Grounded"
    if s <= 0.50: return "Partially Supported"
    if s < 0.85:  return "Unsupported"
    return "Refuted"

def weighted_groundedness(sentences: List[str], scores: List[float]) -> float:
    """
    Token-weighted mean of (1 - s). Falls back to char-weight if tiktoken unavailable.
    """
    if not sentences or not scores or len(sentences) != len(scores):
        return 0.0
    
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        weights = [max(1, len(enc.encode(s))) for s in sentences]
    except Exception:
        weights = [max(1, len(s)) for s in sentences]
    
    grounded = [1.0 - s for s in scores]
    num = sum(w * g for w, g in zip(weights, grounded))
    denom = sum(weights)
    
    if denom == 0:
        return 0.0
    
    return num / denom


@dataclass
class JudgeResult:
    id: str
    sentences: List[str]
    scores: List[float]
    labels: List[str]
    overall_groundedness: float
    decision: str
    provider: str
    model: str
    prompt_sha256: str
    timestamp_utc: str


# -----------------------------
# Core evaluation
# -----------------------------

def evaluate_row(row: Dict[str,Any], provider: LLMProvider, tau: float = 0.80,
                 timeout: int = 60, retry_once: bool = True,
                 save_traces_dir: Optional[str] = None) -> JudgeResult:
    """
    Evaluate a single row {id, question, context, response}.
    Returns a JudgeResult; optionally writes traces (prompt, raw, retry) if save_traces_dir is provided.
    """
    rid = str(row.get("id", "unknown"))
    context = str(row.get("context", ""))
    response = str(row.get("response", ""))
    
    if not context.strip():
        raise ValueError(f"Empty context for row {rid}")
    if not response.strip():
        raise ValueError(f"Empty response for row {rid}")

    # 1) split
    sentences = split_sentences(response)
    if not sentences:
        # Handle case where response has no sentences
        return JudgeResult(
            id=rid,
            sentences=[],
            scores=[],
            labels=[],
            overall_groundedness=0.0,
            decision="UNKNOWN",
            provider=provider.__class__.__name__.replace("Provider","").lower(),
            model=provider.model,
            prompt_sha256="",
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )

    # 2) prompt
    user_prompt = build_user_prompt(context=context, response=response, sentences=sentences)
    prompt_hash = sha256(SYSTEM_PROMPT + "\n" + user_prompt)

    # optional traces
    if save_traces_dir:
        os.makedirs(save_traces_dir, exist_ok=True)
        safe_id = _safe(rid)
        with open(os.path.join(save_traces_dir, f"{safe_id}.prompt.txt"), "w", encoding="utf-8") as f:
            f.write(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}")

    # 3) call LLM
    try:
        raw = provider.call(SYSTEM_PROMPT, user_prompt, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"LLM call failed for row {rid}: {e}")

    if save_traces_dir:
        safe_id = _safe(rid)
        with open(os.path.join(save_traces_dir, f"{safe_id}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

    # 4) parse/validate (retry once)
    try:
        scores = parse_scores(raw, n=len(sentences))
    except Exception as e:
        if not retry_once:
            raise RuntimeError(f"Failed to parse scores for row {rid}: {e}")
        
        correction = (
            f"Invalid JSON or wrong length. Re-output only "
            f'{{"scores":[...]}} with exactly {len(sentences)} floats in [0,1].'
        )
        try:
            raw2 = provider.call(SYSTEM_PROMPT, correction, timeout=timeout)
        except Exception as e2:
            raise RuntimeError(f"LLM retry call failed for row {rid}: {e2}")
        
        if save_traces_dir:
            safe_id = _safe(rid)
            with open(os.path.join(save_traces_dir, f"{safe_id}.raw.retry.txt"), "w", encoding="utf-8") as f:
                f.write(raw2)
        
        try:
            scores = parse_scores(raw2, n=len(sentences))
        except Exception as e3:
            raise RuntimeError(f"Failed to parse scores after retry for row {rid}: {e3}")

    labels = [label_of(s) for s in scores]
    overall = weighted_groundedness(sentences, scores)
    decision = "FACT" if overall >= tau else "HALLUCINATION"

    return JudgeResult(
        id=rid,
        sentences=sentences,
        scores=scores,
        labels=labels,
        overall_groundedness=overall,
        decision=decision,
        provider=provider.__class__.__name__.replace("Provider","").lower(),
        model=provider.model,
        prompt_sha256=prompt_hash,
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )

@dataclass
class ProcessingStats:
    total_lines: int
    empty_lines: int
    invalid_json: int
    missing_fields: int
    empty_fields: int
    processing_errors: int
    successfully_processed: int
    skipped_rows: List[Dict[str, Any]]

def evaluate_jsonl(input_path: str, provider: str, model: str, tau: float = 0.80,
                   max_rows: int = 0, out_dir: Optional[str] = None, timeout: int = 60) -> tuple[List[JudgeResult], ProcessingStats]:
    """
    Run evaluation over a JSONL of rows. Returns (results, processing_stats).
    If out_dir is provided, also writes:
      - {out_dir}/results.jsonl (detailed)
      - {out_dir}/summary.csv (compact)
      - {out_dir}/traces/ (prompts & raw outputs)
      - {out_dir}/processing_log.jsonl (skipped rows and reasons)
    """
    prov = get_provider(provider, model)
    results: List[JudgeResult] = []
    
    # Track processing statistics
    stats = ProcessingStats(
        total_lines=0, empty_lines=0, invalid_json=0, 
        missing_fields=0, empty_fields=0, processing_errors=0,
        successfully_processed=0, skipped_rows=[]
    )

    traces_dir = os.path.join(out_dir, "traces") if out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # fresh files
        results_path = os.path.join(out_dir, "results.jsonl")
        summary_path = os.path.join(out_dir, "summary.csv")
        processing_log_path = os.path.join(out_dir, "processing_log.jsonl")
        
        with open(results_path, "w", encoding="utf-8") as _:
            pass
        with open(summary_path, "w", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow(["id","overall_groundedness","decision","provider","model","timestamp_utc"])
        with open(processing_log_path, "w", encoding="utf-8") as _:
            pass

    def log_skip(reason: str, line_num: int, data: Dict[str, Any] = None):
        skip_entry = {
            "reason": reason,
            "line_number": line_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }
        stats.skipped_rows.append(skip_entry)
        if out_dir:
            _append_jsonl(processing_log_path, skip_entry)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats.total_lines += 1
                
                if not line.strip():
                    stats.empty_lines += 1
                    continue
                
                # Parse JSON
                try:
                    row = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    stats.invalid_json += 1
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    log_skip("invalid_json", line_num, {"error": str(e), "line_content": line.strip()[:200]})
                    continue
                
                # Preserve original row for logging
                original_row = row.copy()
                
                # Schema validation
                required_fields = ["id", "question", "context", "response"]
                missing_fields = [k for k in required_fields if k not in row]
                if missing_fields:
                    stats.missing_fields += 1
                    print(f"Warning: Skipping row on line {line_num} - missing fields: {missing_fields}")
                    log_skip("missing_fields", line_num, {"missing": missing_fields, "row": original_row})
                    continue
                
                # Convert to strings and validate non-empty
                empty_fields = []
                for k in required_fields:
                    if not isinstance(row[k], str):
                        print(f"Info: Converting field '{k}' to string for row on line {line_num}")
                        row[k] = str(row[k])
                    if not row[k].strip():
                        empty_fields.append(k)
                
                if empty_fields:
                    stats.empty_fields += 1
                    print(f"Warning: Skipping row on line {line_num} - empty fields: {empty_fields}")
                    log_skip("empty_fields", line_num, {"empty": empty_fields, "row": original_row})
                    continue
                
                # Process the row
                try:
                    res = evaluate_row(row, prov, tau=tau, timeout=timeout, retry_once=True, save_traces_dir=traces_dir)
                    results.append(res)
                    stats.successfully_processed += 1
                    
                    if out_dir:
                        # Add original input data to result for verification
                        result_with_input = _result_to_dict(res)
                        result_with_input["input_row"] = original_row
                        result_with_input["line_number"] = line_num
                        
                        _append_jsonl(results_path, result_with_input)
                        with open(summary_path, "a", newline="", encoding="utf-8") as fcsv:
                            csv.writer(fcsv).writerow([
                                res.id, 
                                f"{res.overall_groundedness:.6f}", 
                                res.decision, 
                                res.provider, 
                                res.model, 
                                res.timestamp_utc
                            ])
                    
                    if max_rows and stats.successfully_processed >= max_rows:
                        print(f"Reached max_rows limit ({max_rows}), stopping processing")
                        break
                        
                except Exception as e:
                    stats.processing_errors += 1
                    error_msg = f"Error processing row {row.get('id', 'unknown')} on line {line_num}: {e}"
                    print(error_msg)
                    log_skip("processing_error", line_num, {"error": str(e), "row": original_row})
                    continue
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading input file {input_path}: {e}")
    
    # Write final processing summary
    if out_dir:
        processing_summary = {
            "input_file": input_path,
            "processing_stats": {
                "total_lines": stats.total_lines,
                "empty_lines": stats.empty_lines,
                "invalid_json": stats.invalid_json,
                "missing_fields": stats.missing_fields,
                "empty_fields": stats.empty_fields,
                "processing_errors": stats.processing_errors,
                "successfully_processed": stats.successfully_processed,
            },
            "configuration": {
                "provider": provider,
                "model": model,
                "tau": tau,
                "max_rows": max_rows,
                "timeout": timeout,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(os.path.join(out_dir, "processing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(processing_summary, f, indent=2)
    
    return results, stats

# -----------------------------
# Helpers
# -----------------------------

def _result_to_dict(r: JudgeResult) -> Dict[str, Any]:
    return {
        "id": r.id,
        "sentences": r.sentences,
        "scores": r.scores,
        "labels": r.labels,
        "overall_groundedness": round(r.overall_groundedness, 6),
        "decision": r.decision,
        "provider": r.provider,
        "model": r.model,
        "prompt_sha256": r.prompt_sha256,
        "timestamp_utc": r.timestamp_utc,
    }

def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:180]