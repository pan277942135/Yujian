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
    "STRICT_HEAD_SWAP",
    "HEAD_SWAP_SCENE_TRANSFER",
    "IDENTITY_BLEND",
    "FULL_CHARACTER_REBUILD",
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
    "IDENTITY": "Use this picture as the sole authoritative source for the target person's facial identity and stable personal features. The base image person is not identity authority.",
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


def _clean_output_contract() -> str:
    return (
        "Return a clean image without overlaid watermark text, app chrome, status bars, navigation bars, toolbars, "
        "floating UI buttons, screenshot controls, subtitle/caption overlays, decorative borders, or other interface "
        "elements that are not part of the photographed scene. Preserve genuine in-scene signage, printed clothing "
        "graphics, product branding, and naturally photographed text unless the user explicitly asks to remove them."
    )


def compile_clean_frame_prompt(instruction: str = "") -> CompiledPrompt:
    user_instruction = str(instruction or "").strip()
    sections = [
        "CLEAN FRAME STAGE:",
        (
            "Remove only non-scene overlays and screenshot/interface artifacts. "
            "Do not redesign, beautify, restyle, or replace any person or object."
        ),
        "",
        "CLEAN OUTPUT CONTRACT:",
        _clean_output_contract(),
        "",
        "PRESERVATION CONTRACT:",
        (
            "Preserve person identity, face, hair, body, clothing, pose, hands, background, scene geometry, camera, "
            "crop, perspective, lighting, shadows, color, and all photographic content. Fill only pixels revealed by "
            "removed overlays so they continue the immediately surrounding scene naturally."
        ),
    ]
    if user_instruction:
        sections.extend(["", "USER CONTEXT:", user_instruction])
    negative = ", ".join(
        [
            "person redesign",
            "identity change",
            "face change",
            "body reshaping",
            "outfit replacement",
            "background replacement",
            "camera change",
            "crop change",
            "removing real-world signage",
            "removing printed clothing graphics",
            "removing genuine product logos",
            "watermark overlay",
            "app UI",
            "toolbar",
            "status bar",
            "navigation bar",
            "screenshot controls",
            "subtitle overlay",
            "floating button",
            "decorative border",
        ]
    )
    return CompiledPrompt(
        prompt="\n".join(sections).strip(),
        negative_prompt=negative,
        mode="BASE_EDIT",
        preservation="MAX",
        reference_roles=(),
    )


def compile_strict_head_swap_prompt(
    instruction: str,
    *,
    has_angle_reference: bool = False,
    keep_hair_color: bool = False,
    keep_base_hair_shape: bool = False,
    identity_strength: str = "HIGH",
    negative_prompt: str = "",
    clean_output: bool = True,
) -> CompiledPrompt:
    instruction_value = str(instruction or "").strip()
    if not instruction_value:
        instruction_value = "Replace the local head identity with the identity reference."
    strength = str(identity_strength or "HIGH").strip().upper()
    if strength not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("identity_strength must be LOW, MEDIUM, or HIGH")

    roles = ("IDENTITY", "FACE_ANGLE") if has_angle_reference else ("IDENTITY",)
    hair_rule = (
        "Preserve the Base ROI hair shape and silhouette."
        if keep_base_hair_shape
        else "The identity reference may guide hairline and hairstyle only where needed for identity continuity."
    )
    color_rule = (
        "Preserve the Base ROI hair color."
        if keep_hair_color
        else "The identity reference may guide hair color when it is a stable character trait."
    )
    strength_rule = {
        "LOW": "Prefer a conservative identity transfer while still changing the person identity.",
        "MEDIUM": "Make the target identity clearly recognizable without exaggerating facial features.",
        "HIGH": "Make the target identity unmistakable. Identity fidelity is the highest priority inside this head ROI.",
    }[strength]

    sections = [
        "STRICT LOCAL HEAD-SWAP INSTRUCTION:",
        instruction_value,
        "",
        "PICTURE AUTHORITY:",
        (
            "Picture 1 is a LOCAL HEAD ROI cropped from the Base image. It is authoritative only for local pose, "
            "camera angle, lighting direction, neck alignment, and crop geometry. It is NOT identity authority."
        ),
        (
            "Picture 2 is a tightly cropped IDENTITY head reference and is the SOLE authority for who this person is. "
            "Replace the original Base face identity rather than blending with it."
        ),
    ]
    if has_angle_reference:
        sections.append(
            "Picture 3 is FACE_ANGLE geometry support only. Use its viewing angle but never let it override Picture 2 identity."
        )
    sections.extend(
        [
            "",
            "STRICT REGION CONTRACT:",
            (
                "Edit only the head/face/hair content inside this local crop. Do not invent shoulders, torso, outfit, "
                "chest shape, body proportions, hands, background, or scene content from any reference."
            ),
            "",
            "IDENTITY FIDELITY:",
            strength_rule,
            (
                "Match stable facial structure, eye shape and spacing, brows, nose, lips, jawline, forehead proportion, "
                "hairline, skin tone, and age impression from Picture 2. Do not average Picture 1 and Picture 2 faces."
            ),
            hair_rule,
            color_rule,
            "",
            "COMPOSITING AWARENESS:",
            (
                "Keep head orientation, neck connection, local illumination, perspective, and edge geometry compatible "
                "with Picture 1 because the edited ROI will be feather-composited back into the untouched Base image."
            ),
        ]
    )

    if clean_output:
        sections.extend(["", "CLEAN OUTPUT CONTRACT:", _clean_output_contract()])

    negative_parts = [
        "original Base facial identity",
        "hybrid face",
        "face averaging",
        "identity drift",
        "copying identity-reference clothing",
        "copying identity-reference chest or shoulders",
        "new background",
        "new torso",
        "duplicate person",
        "deformed face",
        "asymmetric eyes",
        "extra facial features",
    ]
    supplied = str(negative_prompt or "").strip()
    if supplied:
        negative_parts.append(supplied)

    return CompiledPrompt(
        prompt="\n".join(sections).strip(),
        negative_prompt=", ".join(negative_parts),
        mode="STRICT_HEAD_SWAP",
        preservation="MAX",
        reference_roles=roles,
    )


def compile_scene_transfer_stage_prompt(
    instruction: str,
    *,
    negative_prompt: str = "",
    clean_output: bool = True,
) -> CompiledPrompt:
    instruction_value = str(instruction or "").strip()
    if not instruction_value:
        instruction_value = "Transfer the Base subject into the scene reference."

    sections = [
        "SCENE TRANSFER STAGE:",
        instruction_value,
        "",
        "PICTURE AUTHORITY:",
        (
            "Picture 1 / Base is authoritative for the person pose, body geometry, hands, clothing, subject scale, "
            "camera framing, and subject placement."
        ),
        (
            "Picture 2 / SCENE is authoritative only for the environment, architecture, background, spatial atmosphere, "
            "and ambient lighting. Do not copy any person identity, body, outfit, or pose from Picture 2."
        ),
        "",
        "STAGE CONTRACT:",
        (
            "Change the environment while preserving the Base person as faithfully as possible. This is Stage 1; "
            "a separate strict head-swap stage will handle final identity after this scene edit."
        ),
    ]
    if clean_output:
        sections.extend(["", "CLEAN OUTPUT CONTRACT:", _clean_output_contract()])

    negative_parts = [
        "new person",
        "identity replacement",
        "body reshaping",
        "outfit replacement",
        "pose change",
        "duplicate person",
        "extra limbs",
    ]
    supplied = str(negative_prompt or "").strip()
    if supplied:
        negative_parts.append(supplied)
    return CompiledPrompt(
        prompt="\n".join(sections).strip(),
        negative_prompt=", ".join(negative_parts),
        mode="SCENE_TRANSFER",
        preservation="STRONG",
        reference_roles=("SCENE",),
    )


def compile_image_studio_prompt(
    instruction: str,
    *,
    mode: str = "BASE_EDIT",
    preservation: str = "STRONG",
    reference_roles: Iterable[str] = (),
    negative_prompt: str = "",
    clean_output: bool = True,
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

    if mode_value == "STRICT_HEAD_SWAP":
        if not roles or roles[0] != "IDENTITY":
            raise ValueError("STRICT_HEAD_SWAP requires first reference role IDENTITY")
        if len(roles) > 1 and roles[1] != "FACE_ANGLE":
            raise ValueError("STRICT_HEAD_SWAP second reference must be FACE_ANGLE")
        return compile_strict_head_swap_prompt(
            instruction_value,
            has_angle_reference=len(roles) > 1,
            negative_prompt=negative_prompt,
            clean_output=clean_output,
        )

    identity_indexes = [
        index
        for index, role in enumerate(roles, start=2)
        if role == "IDENTITY"
    ]
    identity_picture = identity_indexes[0] if identity_indexes else None

    if identity_picture is not None and mode_value == "IDENTITY_BLEND":
        base_authority = (
            "Picture 1 / Base remains the primary person identity and is authoritative for camera, crop, pose, body, "
            "clothing, lighting, scene, and overall facial continuity. The IDENTITY reference should influence selected "
            "facial qualities softly without replacing the person wholesale."
        )
    elif identity_picture is not None and mode_value == "FULL_CHARACTER_REBUILD":
        base_authority = (
            "Picture 1 / Base is authoritative only for camera, framing, pose, subject placement, lighting direction, "
            "and scene geometry. The original person appearance, face, hair, clothing, and body styling may be rebuilt "
            "from the character reference."
        )
    elif identity_picture is not None:
        base_authority = (
            "Picture 1 / Base is authoritative only for camera, crop, body pose, hands, clothing, lighting, "
            "background, scene geometry, and other non-identity details. It is NOT authoritative for the person's "
            "face or identity. The original Base face must not be preserved when it conflicts with the IDENTITY reference."
        )
    else:
        base_authority = _PRESERVE_RULES[preservation_value]

    sections = [
        "IMAGE STUDIO EDIT INSTRUCTION:",
        instruction_value,
        "",
        "BASE IMAGE AUTHORITY:",
        base_authority,
    ]

    if identity_picture is not None and mode_value == "IDENTITY_BLEND":
        sections.extend(
            [
                "",
                "IDENTITY BLEND:",
                (
                    f"Picture {identity_picture} is a soft identity influence, not a replacement authority. "
                    "Keep the Base person recognizable while gently borrowing selected facial qualities and character mood. "
                    "Do not copy the reference outfit, body, background, or full hairstyle."
                ),
            ]
        )
    elif identity_picture is not None and mode_value == "FULL_CHARACTER_REBUILD":
        sections.extend(
            [
                "",
                "FULL CHARACTER REBUILD:",
                (
                    f"Picture {identity_picture} is the authoritative character source for face identity, hair, age impression, "
                    "and overall character appearance. Rebuild the Base person as this character while preserving Base camera, "
                    "pose, placement, and scene. Do not preserve the original Base person's identity."
                ),
            ]
        )
    elif identity_picture is not None:
        sections.extend(
            [
                "",
                "IDENTITY REPLACEMENT AUTHORITY:",
                (
                    f"Picture {identity_picture} is the SOLE identity authority. Rebuild the person in Picture 1 / Base "
                    f"so the result unmistakably depicts the same person as Picture {identity_picture}. Do not preserve "
                    "the original Base person's facial identity. Do not blend or average the Base face with the identity "
                    "reference. Match the identity reference's stable facial structure, eye shape and spacing, brows, "
                    "nose, lips, jawline, forehead proportion, hairline, skin tone, and age impression, while preserving "
                    "the Base pose, camera, body, clothing, lighting, and scene unless the edit instruction says otherwise."
                ),
                "",
                "IDENTITY PRIORITY:",
                (
                    "If identity conflicts with Base-image facial appearance, IDENTITY wins. "
                    "If pose/composition conflicts with the identity reference, Base wins for pose/composition only."
                ),
            ]
        )

    elif mode_value == "IDENTITY_LOCK":
        sections.extend(
            [
                "",
                "IDENTITY LOCK:",
                (
                    "Preserve the Base person's existing identity because no IDENTITY reference was supplied. "
                    "Do not drift or average facial identity."
                ),
            ]
        )

    if mode_value == "LOCAL_EDIT":
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
                "Keep the target identity stable while applying the requested scene or environment change.",
            ]
        )

    if clean_output:
        sections.extend(["", "CLEAN OUTPUT CONTRACT:", _clean_output_contract()])

    if roles:
        sections.extend(["", "REFERENCE ROLES:"])
        for index, role in enumerate(roles, start=2):
            if role == "IDENTITY" and mode_value == "IDENTITY_BLEND":
                rule = (
                    "Use this picture only as a soft facial-character influence. "
                    "The Base person remains identity authority."
                )
            elif role == "IDENTITY" and mode_value == "FULL_CHARACTER_REBUILD":
                rule = (
                    "Use this picture as the authoritative full character source for face identity, hair, "
                    "age impression, and overall person appearance."
                )
            else:
                rule = _ROLE_RULES[role]
            sections.append(f"Picture {index} role = {role}. {rule}")

    negative_parts = [
        "face averaging",
        "duplicate person",
        "deformed anatomy",
        "extra fingers",
        "missing fingers",
        "unrequested camera change",
        "unrequested crop change",
        "unrequested background change",
    ]
    if identity_picture is not None and mode_value == "IDENTITY_BLEND":
        negative_parts.extend(
            [
                "full identity replacement",
                "copying reference outfit",
                "copying reference body",
            ]
        )
    elif identity_picture is not None:
        negative_parts.extend(
            [
                "preserving the original Base face identity",
                f"identity drift away from Picture {identity_picture}",
                "hybrid face between Base and identity reference",
            ]
        )
    else:
        negative_parts.insert(0, "unrequested identity change")
    if clean_output:
        negative_parts.extend(
            [
                "watermark overlay",
                "app UI",
                "toolbar",
                "status bar",
                "navigation bar",
                "screenshot controls",
                "subtitle overlay",
                "floating button",
                "decorative border",
            ]
        )
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
    "compile_clean_frame_prompt",
    "compile_image_studio_prompt",
    "compile_scene_transfer_stage_prompt",
    "compile_strict_head_swap_prompt",
]
