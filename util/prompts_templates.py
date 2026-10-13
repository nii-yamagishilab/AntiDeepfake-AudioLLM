"""Prompt template rendering from config."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptTemplate:
    """Prompt text from config, filled in per example."""

    template: str = ""
    template_body: str = ""
    template_suffix: str = ""
    name: str = "default"
    system_message: str = "You are a helpful assistant."
    static_context: dict[str, Any] = field(default_factory=dict)

    def _fmt(self, text: str, merged: dict) -> str:
        return text.format_map(merged).strip()

    def render(
        self, example_payload: dict[str, Any], acoustic_text: str = ""
    ) -> str:
        """Fill template placeholders from the CSV row and metadata."""
        merged = _SafeFormatDict()
        merged.update(self.static_context)
        merged.update(example_payload)
        metadata = example_payload.get("metadata", {})
        if isinstance(metadata, dict):
            merged.update(metadata)

        if self.template_body:
            body = self._fmt(self.template_body, merged)
            suffix = (
                self._fmt(self.template_suffix, merged)
                if self.template_suffix
                else ""
            )
            parts = [body]
            if acoustic_text:
                parts.append(acoustic_text)
            if suffix:
                parts.append(suffix)
            return "\n\n".join(parts)

        # Legacy single-template path
        return self._fmt(self.template, merged)


def build_prompt_template(prompt_config: dict[str, Any]) -> PromptTemplate:
    """Build a PromptTemplate from the prompt config."""
    return PromptTemplate(
        name=prompt_config.get("name", "default"),
        template=prompt_config.get("template", ""),
        template_body=prompt_config.get("template_body", ""),
        template_suffix=prompt_config.get("template_suffix", ""),
        system_message=prompt_config.get(
            "system_message", "You are a helpful assistant."
        ),
        static_context=prompt_config.get("static_context", {}),
    )


class _SafeFormatDict(dict):
    """Leave unknown {placeholders} as-is instead of raising."""

    def __missing__(self, key):
        return "{" + key + "}"
