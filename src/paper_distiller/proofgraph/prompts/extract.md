You are a mathematical knowledge extractor. Your task is to extract structured nodes from the following paper segment.

## Running memory (what has been established so far)
{memory}

## Current segment
Kind hint: {kind_hint}
Section: {section}

```
{segment_text}
```

## Extraction depth
Depth: {depth}

## Instructions

Extract all mathematically significant assertions from THIS segment only. Do NOT reference content from other segments or papers.

Return ONLY a JSON object with this exact structure — no prose, no markdown fences:

{{"nodes": [
  {{
    "kind": "<theorem|lemma|definition|assumption|proof_step|claim>",
    "label": "<label if stated, e.g. 'Theorem 4.3', or null>",
    "text": "<normalized assertion in your own words>",
    "source_quote": "<VERBATIM text copied character-for-character from the segment above>",
    "techniques": ["<technique name>", ...],
    "refs": [
      {{"rel": "<depends_on|uses_lemma|uses_def|uses_assumption>", "target": "<label>"}}
    ]
  }}
]}}

## Critical rules (violation = discarded node)

1. `source_quote` MUST be a verbatim substring copied from the segment text above. Do NOT paraphrase, summarize, or invent. Copy the exact characters.
2. If you cannot find a verbatim quote for a claim, OMIT that node entirely. Abstain rather than fabricate.
3. `refs` must only list labels that actually appear in this segment or in the running memory above. Do not invent labels.
4. For `depth=theorem`: extract only theorem/lemma/definition/claim nodes; skip proof_step decomposition.
5. For `depth=step`: fully decompose proof blocks into individual proof_step nodes, each with its own verbatim quote.
6. Return an empty nodes list `{{"nodes": []}}` if there is nothing to extract from this segment.
7. Do not include commentary, explanation, or any text outside the JSON object.
