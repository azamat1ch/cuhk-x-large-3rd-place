"""Discoverable investigator guidance loaded only from a run's frozen assets."""

import hashlib
import re
from pathlib import PurePosixPath

PREFIX = "prompts/investigators/skills/"


class SkillLibrary:
    def __init__(self, assets):
        self.assets = {k: v for k, v in assets.items() if k.startswith(PREFIX)}
        self.skills = {}
        for path, content in sorted(self.assets.items()):
            relative = PurePosixPath(path.removeprefix(PREFIX))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe frozen skill asset")
            if len(relative.parts) != 2 or relative.name != "SKILL.md":
                continue
            skill_id = relative.parts[0]
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_id):
                raise RuntimeError("Invalid frozen skill identifier")
            lines = content.splitlines()
            if not lines or lines[0] != "---" or "---" not in lines[1:]:
                raise RuntimeError("Missing skill metadata")
            fields = dict(
                line.split(":", 1) for line in lines[1 : lines.index("---", 1)] if ":" in line
            )
            if (
                fields.get("name", "").strip() != skill_id
                or not fields.get("description", "").strip()
            ):
                raise RuntimeError("Invalid skill metadata")
            root = PREFIX + skill_id + "/"
            self.skills[skill_id] = {
                "skill_id": skill_id,
                "description": fields["description"].strip(),
                "resources": sorted(
                    k.removeprefix(root) for k in self.assets if k.startswith(root) and k != path
                ),
            }

    def list(self):
        return [dict(value, resources=list(value["resources"])) for value in self.skills.values()]

    def read(self, skill_id, resource=None):
        if skill_id not in self.skills:
            raise ValueError("Unavailable skill or resource")
        resource = "SKILL.md" if resource is None else resource
        if resource != "SKILL.md" and resource not in self.skills[skill_id]["resources"]:
            raise ValueError("Unavailable skill or resource")
        content = self.assets[PREFIX + skill_id + "/" + resource]
        return {
            "skill_id": skill_id,
            "resource": resource,
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
