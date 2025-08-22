---
date: August 21, 2025
---

# Appendix B: GitHub README Conventions

---

## Core Conventions

1. **Purpose and Description (first few lines)**
   - A concise statement of what the project does and why it exists.
   - Often paired with badges (build status, license, PyPI version, DOI).

2. **Installation Instructions**
   - Step-by-step setup, covering dependencies, environments, and package managers.
   - Explicit code blocks for copy-pasting.

3. **Usage Examples**
   - Minimal working examples (command-line invocation, Python/Julia snippets, etc.).
   - Show expected output or figures when relevant.

4. **Project Structure/Documentation Links**
   - Point to API docs, wiki, or Jupyter Book if detailed documentation is elsewhere.
   - Optional: include a file tree for orientation.

5. **Contributing**
   - Guidelines for pull requests, coding style, testing, and code of conduct.
   - Link to `CONTRIBUTING.md` if it exists.

6. **License**
   - Name and link to the `LICENSE` file.
   - GitHub automatically recognizes [SPDX](https://spdx.org/) identifiers.

7. **Acknowledgments/Citation**
   - Funding, collaborators, or citation guidelines (with DOI/Zenodo badge).
   - For academic projects, a `CITATION.cff` file is recommended.

---

## Style Conventions

- **Markdown**: Commonly used for formatting (`README.md`).
- **Headings and hierarchy**: Start with `# Project Name`, then use `##`, `###` consistently.
- **Conciseness**: Avoid long prose; link out for detail.
- **Code fences**: Always specify the language for syntax highlighting.
- **Badges at top**: Compact visual indicators (build, coverage, DOI).
- **Accessibility**: Descriptive alt text for images; avoid unnecessary jargon.

---

## Widely Referenced Guidelines

- **GitHub Docs**: [About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- **Awesome README**: Curated examples and templates: <https://github.com/matiassingers/awesome-readme>
- **CITATION.cff standard**: <https://citation-file-format.github.io/>
