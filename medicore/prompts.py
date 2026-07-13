"""Prompt templates for the two LLM stages."""

# --- Stage 1: clinical concept extraction -----------------------------------
EXTRACT_SYSTEM = (
    "You are an expert clinical documentation specialist. You read a medical "
    "note and abstract it into the discrete, codeable clinical concepts a "
    "professional medical coder would look up. Report only what the "
    "documentation supports. Return strict JSON."
)

EXTRACT_USER = """Read the medical note and list every codeable clinical concept: \
confirmed diagnoses, conditions, injuries (with laterality if stated), \
signs/symptoms not integral to a diagnosis, procedures/encounters, external \
causes, and relevant status/history (e.g. long-term drug use, prior surgery).

For each concept give a short precise phrase suitable for searching an \
ICD-10-CM index. Include qualifiers actually documented (acute/chronic, \
laterality, initial/subsequent encounter, causative organism, etc.).

Return JSON exactly as:
{{"concepts": ["<phrase>", "<phrase>", ...]}}

MEDICAL NOTE:
\"\"\"
{note}
\"\"\"
"""

# --- Stage 2: code assignment from candidates -------------------------------
ASSIGN_SYSTEM = (
    "You are a certified ICD-10-CM medical coder. From a candidate list of "
    "codes retrieved for a note, you select the codes that are directly "
    "supported by the documentation, choosing the most specific correct code. "
    "You NEVER invent codes: every code you output must appear verbatim in the "
    "candidate list. Follow ICD-10-CM specificity rules (laterality, encounter "
    "type, acuity, combination codes). Return strict JSON."
)

ASSIGN_USER = """Assign ICD-10-CM codes to the medical note below using ONLY the \
candidate codes provided. Select each code that the documentation supports; \
prefer the most specific matching code and avoid unsupported or duplicate codes.

Return JSON exactly as:
{{"assignments": [{{"code": "<code from candidates>", "reason": "<short justification>"}}, ...]}}

MEDICAL NOTE:
\"\"\"
{note}
\"\"\"

CANDIDATE CODES (code — description):
{candidates}
"""
