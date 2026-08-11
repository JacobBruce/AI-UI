---
name: create-skill
description: Create AI UI skills (flat .md or Cursor-style SKILL.md folders). Use when authoring a new skill, converting a workflow into a skill, or explaining skills/ layout and frontmatter.
---

# Creating AI UI skills

Skills teach the agent a specific workflow. AI UI lists them in a catalog; the model **reads the skill file on demand** when needed (and any linked siblings beside a folder `SKILL.md`).

## Locations

| Scope | Path |
|-------|------|
| Home | `~/.aiui/skills/` |
| Project | `<project>/.aiui/skills/` |

Project ids override Home on clash.

## Layout (pick one)

**Flat** — `skills/<id>.md` (id includes `.md`, e.g. `my-skill.md`).

**Cursor-style folder** — preferred for multi-file skills:

```
skills/<skill-name>/
  SKILL.md          # required
  reference.md      # optional; link from SKILL.md
  scripts/          # optional helpers
```

Folder id = directory name. Never invent paths outside `skills/`.

## Gather requirements

1. Purpose / workflow
2. Home vs project
3. Flat vs folder (use a folder if you need reference files or scripts)
4. Trigger terms for the description
5. Any verbatim wording from the user (keep it exact)

Infer from context; ask only what is still unclear.

## SKILL.md frontmatter

```markdown
---
name: skill-name
description: What it does and when to use it (third person; include trigger terms)
argument-hint: "Optional hint for the skill chip"
---
```

| Field | Notes |
|-------|--------|
| `name` | Short id/label; lowercase hyphens preferred |
| `description` | Required for discovery — **what** + **when**; max ~1024 chars |
| `argument-hint` | Optional |

Body: concise steps the agent should follow. Keep `SKILL.md` under ~500 lines; put detail in linked sibling files (one level deep).

## Authoring principles

- Assume a capable model — only add domain-specific or project-specific knowledge
- Progressive disclosure: essentials in `SKILL.md`, depth in linked files
- Consistent terminology; no time-sensitive “before date X” guidance
- Prefer forward-slash paths in instructions

## Enable new skills

After writing a skill tell the user how to enable it in AI UI:

  - Click the pen icon (on right side of prompt bar near top of screen)
  - Click "Rules / Skills" button in the bottom left of the prompt editor dialog
  - Click "Reload From Files" button so new skills appear in the catalog
  - Click the checkbox next to a skill name to enable the skill
  - Close the Rules & Skills dialog and click the Apply button in prompt editor

## Checklist

- [ ] Under `~/.aiui/skills/` or `<project>/.aiui/skills/`
- [ ] Flat `.md` or folder with `SKILL.md`
- [ ] Description has what + when (third person)
- [ ] Links to siblings are one level deep
- [ ] User knows to enable it in the Skills toggles
