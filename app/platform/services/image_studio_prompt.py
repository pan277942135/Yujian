"""Structured prompt compiler for Image Studio V1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ALLOWED_MODES = {
    "BASE_EDIT",
    "IDENTITY_LOCK",
    "LOCAL_EDIT",
    "MULTI_REFERENCE",
    "SCENE_TRANSFER",
}
ALLOWED_PRESERVATION = {"NORMAL", "STRONG", "MAX"}
ALLOWED_REFERENCE_ROLES = {
    "IDENTITY",
    "FACE_ANGLE",
    "POSE",
    "BODY",
    "OUTFIT",
    "HAIR",
    "SCENE",
    "STYLE",
    "OBJECT",
}

_ROLE_RULES = {
    "IDENTITY": "Use the identity reference as the authoritative source for facial identity and stable personal features.",
    "FACE_ANGLE": "Use the face-angle reference only to resolve facial geometry for the requested camera angle.",
    "POSE": "Use the pose reference for body pose and gesture, without replacing identity.",
    "BODY": "Use the body reference for body proportions and silhouette, without replacing facial identity.",
    "OUTFIT": "Use the outfit reference for clothing only; do not copy the donor person's identity.",
    "HAIR": "Use the hair reference for hairstyle, hair shape, and hair color only.",
    "SCENE": "Use the scene reference for environment, spatial mood, and background composition only.",
    "STYLE": "Use the style reference for visual treatment only, not identity or scene content.",
    "OBJECT": "Use the object reference for the requested object appearance only.",
}

_PRESERVE_RULES = {
    "NORMAL": "Preserve important source identity and scene details unless the edit instruction requires a change.",
    "STRONG": (
        "Preserve the base image camera, composition, pose, lighting, body geometry, hands, background, and all "
        "unrequested details. Change only what the edit instruction requires."
    ),
    "MAX": (
        "Treat the base image as authoritative. Keep camera, crop, pose, body, hands, clothing, lighting, scene, "
        "background, and all unrequested pixels visually unchanged as far as model generation permits. "
        "Do not redesign or average the subject."
    ),
}


@dataclass(frozen=True)
class CompiledPrompt:
    prompt: str
    negative_prompt: str
    mode: str
    preservation: str
    reference_roles: tuple[str, ...]


def _normalise_enum(value: str, allowed: set[str], label: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}")
    return normalized


def compile_image_studio_prompt(
    instruction: str,
    *,
    mode: str = "BASE_EDIT",
    preservation: str = "STRONG",
    reference_roles: Iterable[str] = (),
    negative_prompt: str = "",
) -> CompiledPrompt:
    instruction_value = str(instruction or "").strip()
    if not instruction_value:
        raise ValueError("instruction cannot be empty")
    if len(instruction_value) > 4000:
        raise ValueError("instruction cannot exceed 4000 characters")

    mode_value = _normalise_enum(mode, ALLOWED_MODES, "mode")
    preservation_value = _normalise_enum(
        preservation, ALLOWED_PRESERVATION, "preservation"
    )
    roles = tuple(
        _normalise_enum(role, ALLOWED_REFERENCE_ROLES, "reference role")
        for role in reference_roles
    )
    if len(roles) > 2:
        raise ValueError("Image Studio V1 supports at most two reference images")

    sections = [
        "IMAGE STUDIO EDIT INSTRUCTION:",
        instruction_value,
        "",
        "BASE IMAGE AUTHORITY:",
        _PRESERVE_RULES[preservation_value],
    ]

    if mode_value == "IDENTITY_LOCK":
        sections.extend(
            [
                "",
                "IDENTITY LOCK:",
                (
                    "Keep the same person. Do not blend, average, reinterpret, or replace facial identity with "
                    "the person from the base image or other references. Preserve stable face proportions, eye "
                    "geometry, nose, mouth, jawline, forehead proportion, hairline, and skin tone unless explicitly edited."
                ),
            ]
        )
    elif mode_value == "LOCAL_EDIT":
        sections.extend(
            [
                "",
                "LOCAL EDIT:",
                "Edit only the requested region. The application will restore pixels outside the supplied mask after generation.",
            ]
        )
    elif mode_value == "SCENE_TRANSFER":
        sections.extend(
            [
                "",
                "SCENE TRANSFER:",
                "Keep the subject identity stable while applying the requested scene or environment change.",
            ]
        )

    if roles:
        sections.extend(["", "REFERENCE ROLES:"])
        for index, role in enumerate(roles, start=2):
            sections.append(f"Picture {index} role = {role}. {_ROLE_RULES[role]}")

    negative_parts = [
        "unrequested identity change",
        "face averaging",
        "duplicate person",
        "deformed anatomy",
        "extra fingers",
        "missing fingers",
        "unrequested camera change",
        "unrequested crop change",
        "unrequested background change",
    ]
    supplied_negative = str(negative_prompt or "").strip()
    if supplied_negative:
        negative_parts.append(supplied_negative)

    return CompiledPrompt(
        prompt="\n".join(sections).strip(),
        negative_prompt=", ".join(negative_parts),
        mode=mode_value,
        preservation=preservation_value,
        reference_roles=roles,
    )


__all__ = [
    "ALLOWED_MODES",
    "ALLOWED_PRESERVATION",
    "ALLOWED_REFERENCE_ROLES",
    "CompiledPrompt",
    "compile_image_studio_prompt",
]
