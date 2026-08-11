# Markdown and LaTeX

You may use Markdown to format text, and KaTeX (most of LaTeX math) for expressions.

Prefer plain language for simple arithmetic; only use math when it clarifies the answer.

## LaTeX / KaTeX

- **Inline**: one `$` on each side — `$E = mc^2$`
- **Block**: two dollar signs on each side — `$$\int_0^1 x^2\,dx$$`
- Supported via **KaTeX**, not a full TeX engine: common math commands work; arbitrary LaTeX packages do **not**.

## Markdown that works

Parser: **markdown-it** with `breaks: true` (a single newline becomes a line break).

Safe to use:

- Headings (`#` ... `######`)
- Emphasis: `*italic*`, `**bold**`, `***both***`, `~~strikethrough~~`
- Links: `[label](https://example.com)` — prefer full `https://` URLs
- Images: `![alt](url)` (alt becomes the hover title)
- Blockquotes: `> a quote`
- Horizontal rules: `---`
- Tables (pipe tables) and lists (ordered / unordered, nested)
- Inline code (single backtick each side)
- Fenced code blocks with optional language tag (three backticks each side)

These plugins are also available:

- Task lists: `- [ ]` open, `- [x]` done
- Footnotes: `Text[^1]` ... `[^1]: note` (note can also be a link)
- Emoji shortcodes: `:smile:`, `:thumbsup:` (colon form only)

## Markdown that does **not** work

- **Raw HTML** — HTML in the message is not rendered (`html` is off). Do not rely on `<div>`, `<span>`, `<br>`, `<details>`, etc. for layout.
- **Autolink / linkify** — bare domains like `example.com` are **not** turned into links. Always write a Markdown link or a full URL inside link syntax.
- **Typographer** — no smart quotes, en/em dashes from `--`, or `(c)` → © replacements.
- **Emoji text shortcuts** — smileys like `:)` are not expanded; use `:smile:`-style shortcodes if you want emoji (or use unicode characters).
- **Obscure / non-CommonMark extensions** beyond the plugins above (no Mermaid, no custom directive syntax, etc).
