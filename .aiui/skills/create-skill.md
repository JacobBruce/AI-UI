---
name: create-skill
description: Create AI UI skills (flat .md or a folder with SKILL.md). Use when authoring a new skill, converting a workflow into a skill, or explaining skills/ layout and frontmatter.
---

# Creating AI UI skills

Skills teach the agent a specific workflow. AI UI lists them in a catalog; the skill files are read on demand when needed.

## Locations

| Scope | Path |
|-------|------|
| Home | `~/.aiui/skills/` |
| Project | `<project>/.aiui/skills/` |

Project files override Home on the same filename.

## Layout (pick one)

**Flat** — `skills/<id>.md`

**Folder** — for multi-file skills:

```
skills/<skill-name>/
  SKILL.md
  reference.md
```

Linked sibling files sit next to `SKILL.md` and are also read when needed.

## Gather requirements

1. Purpose / workflow
2. Home vs project
3. Flat vs folder (use a folder if you need reference files)
4. Trigger terms for the description
5. Any verbatim wording from the user (keep it exact)

Infer from context; ask only what is still unclear.

## Frontmatter

```markdown
---
name: skill-name
description: What it does and when to use it (third person; include trigger terms)
---
```

The body is the steps to follow. Keep `SKILL.md` focused. Put longer reference in a sibling file and link it.

## Authoring principles

- Progressive disclosure: essentials in `SKILL.md`, depth in linked files
- Consistent terminology; no time-sensitive “before date X” guidance
- Prefer forward-slash paths in instructions

## Enable new skills

After writing a skill tell the user how to enable it in AI UI:

1. Click the pen icon (on right side of prompt bar near top of screen)
2. Click the "Rules / Skills" button in the bottom left of the system prompt editor
3. Click the checkbox next to a skill name to enable the skill
4. Close the Rules & Skills dialog and click the Apply button in prompt editor

## Checklist

- [ ] Under `~/.aiui/skills/` or `<project>/.aiui/skills/`
- [ ] Flat `.md` or folder with `SKILL.md`
- [ ] Description has what + when (third person)
- [ ] User knows to enable it in the Skills toggles
