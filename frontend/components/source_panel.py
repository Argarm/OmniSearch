from __future__ import annotations

import chainlit as cl

# Badge labels per source type
_SOURCE_TYPE_LABELS: dict[str, str] = {
    "pdf": "PDF",
    "notion": "Notion",
    "confluence": "Confluence",
}


def build_source_elements(sources: list[dict]) -> list[cl.Text]:
    """Build Chainlit Text elements for each source document.

    Each element shows the document title, source type, location reference
    (page number or URL), and the raw chunk text so users can verify every claim.
    """
    elements: list[cl.Text] = []

    for i, src in enumerate(sources, start=1):
        source_type = src.get("source_type", "unknown")
        label = _SOURCE_TYPE_LABELS.get(source_type, source_type.upper())
        title = src.get("title", "Unknown document")
        score = src.get("score", 0.0)

        # Build the location reference line
        location_parts: list[str] = []
        if src.get("page"):
            location_parts.append(f"Page {src['page']}")
        if src.get("url"):
            location_parts.append(src["url"])
        location = " · ".join(location_parts) if location_parts else ""

        # Format the element content
        header = f"[{label}] {title}"
        if location:
            header += f"  •  {location}"
        header += f"  •  relevance: {score:.2%}"

        content = f"{header}\n{'─' * 60}\n{src.get('chunk_text', '')}"

        elements.append(
            cl.Text(
                name=f"source_{i}",
                content=content,
                display="side",
            )
        )

    return elements
