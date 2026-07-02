TEMPLATES = {
    "Research Paper": (
        "Title:\n- {title}\n\n"
        "{sections}\n\n"
        "Guidelines:\n"
        "- Generate summaries only from the retrieved context.\n"
        "- Do not use outside knowledge. Do not infer missing information.\n"
        "- Do not create headings that are unsupported by the document. If a section is absent, omit it completely.\n"
        "- Preserve any headings found in the document.\n"
        "- Output clean Markdown."
    ),

    "Textbook / Study Material": (
        "Title:\n- {title}\n\n"
        "Overview:\n- {overview}\n\n"
        "Main Topics:\n{headings_list}\n\n"
        "Guidelines:\n"
        "- Generate summaries only from the retrieved context.\n"
        "- Preserve existing headings and subheadings.\n"
        "- Summarise tables and diagrams if present.\n"
        "- Do not invent sections. Omit absent sections.\n"
        "- Output clean Markdown."
    ),

    "Resume": (
        "Candidate Summary (extract from context):\n- Name: {title}\n\n"
        "Sections (only include if present):\n{sections}\n\n"
        "Guidelines:\n"
        "- Generate summaries only from the retrieved context.\n"
        "- Do not invent skills, projects or dates.\n"
        "- Output clean Markdown."
    ),

    "Legal Document": (
        "Document Title: {title}\n\n"
        "Key Items (only list if present):\n{sections}\n\n"
        "Guidelines:\n"
        "- Preserve original clause headings.\n"
        "- Summarise important clauses, dates and obligations.\n"
        "- Do not invent parties or obligations.\n"
        "- Output clean Markdown."
    ),

    "User Manual": (
        "Title: {title}\n\n"
        "Sections:\n{headings_list}\n\n"
        "Guidelines:\n"
        "- Focus on steps, commands, examples and warnings present in the context.\n"
        "- Do not invent features or commands.\n"
        "- Output clean Markdown."
    ),

    "Business Report": (
        "Title: {title}\n\n"
        "Executive Summary (if present):\n- {overview}\n\n"
        "Findings / Recommendations (only if present):\n{sections}\n\n"
        "Guidelines:\n"
        "- Use only the retrieved context.\n"
        "- Preserve headings and tables.\n"
        "- Output clean Markdown."
    ),

    "Technical Documentation": (
        "Title: {title}\n\n"
        "Sections:\n{headings_list}\n\n"
        "Guidelines:\n"
        "- Preserve code blocks, API endpoints, configuration details found in context.\n"
        "- Do not invent APIs or configurations.\n"
        "- Output clean Markdown."
    ),

    "Book Chapter": (
        "Title: {title}\n\n"
        "Chapter Overview:\n- {overview}\n\n"
        "Main Sections:\n{headings_list}\n\n"
        "Guidelines:\n"
        "- Preserve original chapter headings.\n"
        "- Summarise diagrams, tables, and highlighted concepts if present.\n"
        "- Output clean Markdown."
    ),

    "General Document": (
        "Title: {title}\n\n"
        "Summary (extract from context):\n{overview}\n\n"
        "Guidelines:\n"
        "- Generate summaries only from the retrieved context.\n"
        "- Do not infer or invent missing sections.\n"
        "- Preserve headings when present.\n"
        "- Output clean Markdown."
    ),
}
