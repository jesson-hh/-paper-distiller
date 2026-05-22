You are comparing two mathematical assertions from different papers. Determine the single best relation between them, or "none" if they are unrelated or you are unsure.

## Node A
**Statement:** {text_a}
**Source quote:** {source_quote_a}

## Node B
**Statement:** {text_b}
**Source quote:** {source_quote_b}

## Instructions

Choose ONE relation from this list, or "none":

- **same_as** — A and B state the same mathematical fact (possibly with different notation).
- **specializes** — A is a special case of B (A has stronger hypotheses or narrower scope).
- **generalizes** — A subsumes B (A covers a strictly larger set of cases).
- **uses_lemma** — A's proof depends on B as a lemma or building block (or vice versa).
- **contradicts** — A and B make incompatible claims under the same hypotheses.
- **none** — no meaningful mathematical relationship, or you are unsure.

Prefer **abstaining** ("none") over inventing a link. Only assign a relation when the source quotes make it clear.

Cite the specific spans from the source quotes that justify the relation.

## Output format (JSON only)

```json
{{"rel": "<relation or none>", "justification": "<one sentence citing the relevant spans>"}}
```
