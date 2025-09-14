# Author: Wilfred from team Magellan with Ali and Adrian

from fastmcp import FastMCP
from typing import Dict, Literal, Optional, Any
import uuid

try:
    import akinator  # pip install akinator
except ImportError as e:
    raise SystemExit("The 'akinator' package is required. Install with: pip install akinator") from e

mcp = FastMCP("Akinator Connector 🧞")

# In-memory session store (single-process)
SESSIONS: Dict[str, "akinator.Akinator"] = {}

# Allowed answers for the library
Answer = Literal["yes", "no", "idk", "probably", "probably not"]


def _ensure_session(session_id: str) -> "akinator.Akinator":
    if session_id not in SESSIONS:
        raise ValueError(f"Unknown session: {session_id}")
    return SESSIONS[session_id]


def _nl_question(aki: "akinator.Akinator", ret: Any) -> Optional[str]:
    """
    Robustly extract a natural-language question as a string.
    - Some versions of akinator return the question from start_game(...)
    - Others set aki.question and return None
    """
    if isinstance(ret, str) and ret.strip():
        return ret
    # Prefer the property on the instance
    q = getattr(aki, "question", None)
    if isinstance(q, str) and q.strip():
        return q
    # Fallbacks seen in a few forks
    q2 = getattr(aki, "_question", None)
    if isinstance(q2, str) and q2.strip():
        return q2
    return None


def _nl_progression(aki: "akinator.Akinator") -> Optional[float]:
    prog = getattr(aki, "progression", None)
    try:
        return float(prog) if prog is not None else None
    except Exception:
        return None


def _nl_step(aki: "akinator.Akinator") -> Optional[int]:
    step = getattr(aki, "step", None)
    try:
        return int(step) if step is not None else None
    except Exception:
        return None


def _serialize_guess(guess: Any) -> Optional[dict]:
    """
    Convert Akinator's guess object into a flat, JSON-safe dict.
    Known attributes: name, description, ranking, id, proba, picture_path, absolute_picture_path.
    """
    if guess is None:
        return None
    # Defensive getattr to avoid surfacing non-JSON types
    def g(obj, attr, default=None):
        try:
            val = getattr(obj, attr, default)
            # Only allow JSON-safe primitives
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            return default
        except Exception:
            return default

    # Prefer absolute picture if available
    pic_abs = g(guess, "absolute_picture_path")
    pic_rel = g(guess, "picture_path")
    proba = g(guess, "proba")
    try:
        proba_num = float(proba) if proba is not None else None
    except Exception:
        proba_num = None

    return {
        "name": g(guess, "name"),
        "description": g(guess, "description"),
        "ranking": g(guess, "ranking"),
        "id": g(guess, "id"),
        "probability": proba_num,  # 0..100 or None selon les forks
        "image": pic_abs or pic_rel,  # URL absolue si possible
    }


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool(output_schema=None)
def start_game(language: str = "fr", child_mode: bool = False, theme: Optional[str] = None) -> dict:
    """
    Start a new Akinator game.
    Returns: { session_id, question, step, progression, answers }
    - language: code langue ('fr', 'en', 'de', ...)
    - child_mode: restreint le contenu pour enfants (si supporté)
    - theme: thème/serveur (ex. 'characters'), selon les forks
    """
    aki = akinator.Akinator()

    # Certains forks utilisent des signatures différentes; on reste permissifs
    kwargs = {"language": language}
    # child_mode: suivant les versions c'est 'child_mode' ou 'child_mode=True' silencieux
    try:
        if child_mode:
            kwargs["child_mode"] = True
    except Exception:
        pass
    # theme si supporté
    try:
        if theme:
            kwargs["theme"] = theme
    except Exception:
        pass

    ret = aki.start_game(**kwargs)
    question = _nl_question(aki, ret)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = aki

    return {
        "session_id": session_id,
        "question": question,
        "step": _nl_step(aki),
        "progression": _nl_progression(aki),
        "answers": ["yes", "no", "idk", "probably", "probably not"],
    }


@mcp.tool(output_schema=None)
def answer(session_id: str, user_answer: Answer) -> dict:
    """
    Submit an answer to the current question.
    user_answer ∈ {'yes','no','idk','probably','probably not'}.
    Returns:
      - status: 'question' | 'guess'
      - question (si status='question')
      - guess (si status='guess', dict plat)
      - step, progression
    """
    aki = _ensure_session(session_id)

    # Akinator attend parfois 'y', 'n', 'i', 'p', 'pn' — la lib prend habituellement les mots entiers.
    # On passe directement user_answer.
    next_prompt = aki.answer(user_answer)

    # Si la progression est élevée, on force une proposition
    progression = _nl_progression(aki) or 0.0
    if progression >= 80.0:
        try:
            aki.win()
        except Exception:
            pass
        guess = _serialize_guess(getattr(aki, "first_guess", None))
        return {
            "status": "guess",
            "guess": guess,
            "step": _nl_step(aki),
            "progression": _nl_progression(aki),
        }

    # Sinon, on renvoie la prochaine question en texte naturel
    question = _nl_question(aki, next_prompt)
    return {
        "status": "question",
        "question": question,
        "step": _nl_step(aki),
        "progression": _nl_progression(aki),
    }


@mcp.tool(output_schema=None)
def back(session_id: str) -> dict:
    """
    Go back to the previous question.
    Returns: { question, step, progression }
    """
    aki = _ensure_session(session_id)
    prev_prompt = aki.back()
    question = _nl_question(aki, prev_prompt)
    return {
        "question": question,
        "step": _nl_step(aki),
        "progression": _nl_progression(aki),
    }


@mcp.tool(output_schema=None)
def force_guess(session_id: str) -> dict:
    """
    Force Akinator to propose a guess.
    Returns: { guess, step, progression }
    """
    aki = _ensure_session(session_id)
    try:
        aki.win()
    except Exception:
        pass
    guess = _serialize_guess(getattr(aki, "first_guess", None))
    return {
        "guess": guess,
        "step": _nl_step(aki),
        "progression": _nl_progression(aki),
    }


@mcp.tool(output_schema=None)
def get_state(session_id: str) -> dict:
    """
    Get the current game state.
    Returns: { question, step, progression }
    """
    aki = _ensure_session(session_id)
    return {
        "question": _nl_question(aki, getattr(aki, "question", None)),
        "step": _nl_step(aki),
        "progression": _nl_progression(aki),
    }


@mcp.tool(output_schema=None)
def end_game(session_id: str) -> bool:
    """
    End the game and free the server-side session.
    Returns True if the session was removed.
    """
    _ = _ensure_session(session_id)
    del SESSIONS[session_id]
    return True


if __name__ == "__main__":
    # Expose over HTTP for remote access (e.g., via ngrok)
    mcp.run(transport="http", host="0.0.0.0", port=8000)
