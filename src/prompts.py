"""
Prompt templates used across the pipeline.

- JUDGE_RUBRIC: 5-point quality rubric for response scoring
- format_judge_prompt(): safe concatenation (no str.format, brace-safe)
"""

# ---------------------------------------------------------------------------
# Judge rubric (5-point scale)
# ---------------------------------------------------------------------------

JUDGE_RUBRIC = """\
Below is a user instruction and an AI response. Evaluate the quality of the AI's response based \
on how well it fulfills the user's request. Assign a score based on the following 5-point scale:

1: The response is incomplete, off-topic, or contains irrelevant, vague, or missing information. \
It may repeat the user's question, include personal opinions, or be written from a non-AI \
perspective (e.g., blog-like). It may also have promotional or irrelevant content.

2: The response addresses some of the user's request but lacks detail or direct relevance. It \
provides only a general approach instead of a specific solution.

3: The response is helpful but lacks an AI perspective. It covers the user's request but appears \
taken from a personal blog, webpage, or similar source. It may include personal opinions, \
experiences, or mentions of external content.

4: The response is clear, complete, and written from an AI's perspective. It directly addresses \
the user's request, but there may be minor room for improvement, such as clarity or conciseness.

5: The response is excellent, written from an AI's perspective, with a clear focus on the user's \
request. It is thorough, well-organized, and shows expert knowledge without irrelevant content. \
The response is logical, easy to follow, and engaging.

Provide a brief justification for your score and then write "Score: <rating>" in the last line."""


def format_judge_prompt(instruction: str, response: str) -> str:
    """Build the judge prompt via concatenation — safe for any content
    including curly braces, code, format strings, etc."""
    return JUDGE_RUBRIC + "\n\n" + instruction + "\n\n" + response
