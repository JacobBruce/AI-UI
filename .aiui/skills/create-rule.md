---
name: create-rule
description: Create AI UI rules for persistent prompt guidance. Use when adding coding standards, conventions, or always-on instructions under rules/, or when the user asks how AI UI rules work.
---

# Creating AI UI rules

Rules are markdown files under `rules/` that are injected into the system prompt when enabled.

## Locations

| Scope | Path |
|-------|------|
| Home (all chats) | `~/.aiui/rules/` |
| Project | `<project>/.aiui/rules/` |

Project files override Home on the same filename. Prefer project rules for repo-specific conventions; Home for personal defaults.

## Gather requirements

1. **Purpose** — what the rule should enforce or teach
2. **Scope** — Home vs project `.aiui`
3. **Filename** — short kebab-case `.md` (e.g. `python-style.md`)

Infer from the conversation when possible. If scope is unclear, ask Home vs project.

## File format

Plain markdown. Optional YAML frontmatter is stripped at load time (body only enters the prompt):

```markdown
---
description: Short label for humans (optional; not shown in the rules UI)
---

# Rule title

Actionable instructions…
```

AI UI does **not** use `.mdc` globs / `alwaysApply`. Each rule can be enabled/disabled in the Rules UI.

## Authoring

- Keep under ~100 lines; one concern per file
- Actionable and concrete — write like internal docs
- Prefer several small rules over one large blob

## Enable new rules

After writing a rule tell the user how to enable it in AI UI:

  - Click the pen icon (on right side of prompt bar near top of screen)
  - Click "Rules / Skills" button in the bottom left of the prompt editor dialog
  - Click "Reload From Files" button so new rules appear in the catalog
  - Click the checkbox next to a rule name to enable the rule
  - Close the Rules & Skills dialog and click the Apply button in prompt editor

## Checklist

- [ ] Path is `~/.aiui/rules/` or `<project>/.aiui/rules/`
- [ ] Filename ends in `.md`
- [ ] Body is the guidance (frontmatter optional)
- [ ] User knows to enable it in the Rules toggles
