"""Versioned runtime configuration shared by the API, coordinator and workers.

This file format is the declared source for: agent roles and model assignments,
prompt templates, OCR languages and routing, retrieval thresholds, concurrency,
retries, timeouts, budgets and the text policy. It never contains secret values;
providers name the environment variable that holds their credential.

Loading resolves ``${VAR}``, ``${VAR:-default}`` and ``${VAR:?hint}`` references
from the environment, embeds prompt files, validates everything, and produces a
content-addressed snapshot. Jobs record the snapshot id and workers execute a job
with the snapshot it was created under, so a rolling deploy cannot mix versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from tribunal.errors import ConfigurationError

_VAR = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?:(?P<op>:-|:\?)(?P<arg>[^}]*))?\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(_Strict):
    kind: Literal["openai", "anthropic", "stub"]
    api_key_env: str | None = None
    base_url: str | None = None
    max_concurrent: PositiveInt
    timeout_seconds: PositiveFloat
    options: dict[str, Any] = Field(default_factory=dict)


class RoleConfig(_Strict):
    provider: str
    model: str
    prompt: str
    max_output_tokens: PositiveInt
    temperature: float | None = None
    effort: str | None = None


class PromptTemplate(_Strict):
    id: str
    version: str
    system: str
    user: str
    type_instructions: dict[str, str] = Field(default_factory=dict)


class EmbeddingConfig(_Strict):
    provider: Literal["hashing", "sentence_transformers", "openai"]
    model: str
    dimensions: PositiveInt
    openai_provider: str | None = None


class RetrievalConfig(_Strict):
    embedding: EmbeddingConfig
    chunk_chars: PositiveInt
    chunk_overlap_chars: int = Field(ge=0)
    candidate_k: PositiveInt
    top_k: PositiveInt
    min_score: float = Field(ge=0, le=1)
    min_passages: PositiveInt
    vector_weight: float = Field(ge=0)
    lexical_weight: float = Field(ge=0)
    topic_page_boost: float = Field(ge=0)
    require_term_match: bool
    min_term_coverage: float = Field(ge=0, le=1)
    source_kinds: list[str]


class OcrConfig(_Strict):
    engine: str
    languages: list[str]
    rotate_pages: bool
    deskew: bool
    timeout_seconds: PositiveInt
    jobs: PositiveInt


class LayoutConfig(_Strict):
    engine: str
    review_required: bool


class OutlineConfig(_Strict):
    use_bookmarks: bool
    use_contents_page: bool
    use_headings: bool
    contents_search_pages: PositiveInt
    max_topics: PositiveInt


class IngestionConfig(_Strict):
    max_upload_bytes: PositiveInt
    max_pages: PositiveInt
    batch_pages: PositiveInt
    native_min_chars: PositiveInt
    mixed_image_coverage: float = Field(ge=0, le=1)
    ocr: OcrConfig
    layout: LayoutConfig
    flag_formulas_on: list[Literal["native", "ocr", "layout"]]
    flag_tables_on: list[Literal["native", "ocr", "layout"]]
    flagged_page_policy: Literal["exclude", "include"]
    outline: OutlineConfig


class QuestionTypeConfig(_Strict):
    label: str
    enabled: bool
    writer_role: str = "writer"


class LevelConfig(_Strict):
    key: str
    label: str
    description: str


class GenerationConfig(_Strict):
    question_types: dict[str, QuestionTypeConfig]
    bloom_levels: list[LevelConfig]
    difficulty_levels: list[LevelConfig]
    mcq_option_count: PositiveInt
    max_revisions: int = Field(ge=0)
    reviewer_enabled: bool
    escalate_disputed_reviews: bool
    numeric_relative_tolerance: PositiveFloat
    duplicate_similarity: float = Field(gt=0, le=1)
    placeholder_patterns: list[str]


class ConcurrencyConfig(_Strict):
    worker_threads: PositiveInt
    per_paper: PositiveInt
    per_user: PositiveInt
    ingestion_per_user: PositiveInt


class RetryConfig(_Strict):
    max_attempts: PositiveInt
    base_delay_seconds: PositiveFloat
    max_delay_seconds: PositiveFloat
    lease_seconds: PositiveInt
    poll_interval_seconds: PositiveFloat
    job_retry_delay_seconds: float = Field(ge=0)


class BudgetConfig(_Strict):
    paper_tokens: PositiveInt
    paper_deadline_seconds: PositiveInt
    daily_tokens: PositiveInt | None = None


class EventsConfig(_Strict):
    poll_interval_seconds: PositiveFloat
    heartbeat_seconds: PositiveFloat


class TextPolicyConfig(_Strict):
    preserve_codepoints: list[str]


class ExportConfig(_Strict):
    pdf_font_size: PositiveFloat
    pdf_margin_mm: PositiveFloat
    include_citations_in_answer_key: bool


class ObservabilityConfig(_Strict):
    include_paper_text_in_logs: bool


class VerifierConfig(_Strict):
    cpu_seconds: int = Field(gt=0)
    wall_seconds: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    max_processes: int = Field(gt=0)
    max_file_bytes: int = Field(gt=0)
    max_output_bytes: int = Field(gt=0)


class RuntimeConfig(_Strict):
    version: str
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleConfig]
    prompts: dict[str, PromptTemplate]
    retrieval: RetrievalConfig
    ingestion: IngestionConfig
    generation: GenerationConfig
    concurrency: ConcurrencyConfig
    retries: RetryConfig
    budgets: BudgetConfig
    events: EventsConfig
    text_policy: TextPolicyConfig
    export: ExportConfig
    observability: ObservabilityConfig
    verifier: VerifierConfig

    @model_validator(mode="after")
    def _references(self) -> RuntimeConfig:
        problems: list[str] = []
        for name, role in self.roles.items():
            if role.provider not in self.providers:
                problems.append(f"roles.{name}.provider '{role.provider}' is not defined under providers")
            elif self.providers[role.provider].kind == "anthropic" and role.temperature is not None:
                problems.append(f"roles.{name}.temperature is not supported by Anthropic models; remove it")
            if role.prompt not in self.prompts:
                problems.append(f"roles.{name}.prompt '{role.prompt}' is not defined under prompts")
        if "writer" not in self.roles:
            problems.append("roles.writer is required")
        if self.generation.reviewer_enabled and "reviewer" not in self.roles:
            problems.append("roles.reviewer is required when generation.reviewer_enabled is true")
        if self.generation.escalate_disputed_reviews and "reasoner" not in self.roles:
            problems.append("roles.reasoner is required when generation.escalate_disputed_reviews is true")
        for key, qt in self.generation.question_types.items():
            if qt.writer_role not in self.roles:
                problems.append(f"generation.question_types.{key}.writer_role '{qt.writer_role}' is not a defined role")
        emb = self.retrieval.embedding
        if emb.provider == "openai" and (not emb.openai_provider or emb.openai_provider not in self.providers):
            problems.append("retrieval.embedding.openai_provider must name an openai provider")
        if self.retrieval.top_k > self.retrieval.candidate_k:
            problems.append("retrieval.top_k must not exceed retrieval.candidate_k")
        if self.retrieval.chunk_overlap_chars >= self.retrieval.chunk_chars:
            problems.append("retrieval.chunk_overlap_chars must be smaller than retrieval.chunk_chars")
        if self.retries.lease_seconds <= self.budgets.paper_deadline_seconds:
            problems.append("retries.lease_seconds must be greater than budgets.paper_deadline_seconds")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # Convenience lookups -------------------------------------------------
    def question_type_keys(self) -> list[str]:
        return [k for k, v in self.generation.question_types.items() if v.enabled]

    def bloom_keys(self) -> list[str]:
        return [b.key for b in self.generation.bloom_levels]

    def difficulty_keys(self) -> list[str]:
        return [d.key for d in self.generation.difficulty_levels]


class ConfigSnapshot(BaseModel):
    """A validated config plus its content address."""

    model_config = ConfigDict(frozen=True)

    config: RuntimeConfig
    content_hash: str
    source_path: str | None = None

    @property
    def snapshot_id(self) -> str:
        return f"{self.config.version}+{self.content_hash[:12]}"

    def as_json(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")


def _interpolate(value: Any, missing: list[str], path: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v, missing, f"{path}.{k}" if path else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, missing, f"{path}[{i}]") for i, v in enumerate(value)]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name, op, arg = match.group("name"), match.group("op"), match.group("arg")
        env_value = os.environ.get(name)
        if env_value not in (None, ""):
            return env_value
        if op == ":-":
            return arg or ""
        hint = f" ({arg.strip()})" if op == ":?" and arg else ""
        missing.append(f"environment variable {name} is required by runtime config at {path}{hint}")
        return ""

    replaced = _VAR.sub(repl, value)
    # Keep numbers and booleans typed when a whole value was a single reference
    if _VAR.fullmatch(value) and replaced != "":
        try:
            return yaml.safe_load(replaced)
        except yaml.YAMLError:
            return replaced
    return replaced


def canonical_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_runtime_config(path: str | Path) -> ConfigSnapshot:
    """Load, interpolate, embed prompts and validate. Raises ConfigurationError naming each problem."""
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError([f"runtime config file '{path}' does not exist"])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError([f"runtime config '{path}' is not valid YAML: {exc}"]) from None

    missing: list[str] = []
    resolved = _interpolate(raw, missing)

    prompts: dict[str, Any] = {}
    for key, ref in (resolved.get("prompts") or {}).items():
        if isinstance(ref, dict) and "file" in ref:
            prompt_path = (path.parent / ref["file"]).resolve()
            if not prompt_path.is_file():
                missing.append(f"prompts.{key}.file '{ref['file']}' does not exist (resolved to {prompt_path})")
                continue
            try:
                prompts[key] = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                missing.append(f"prompts.{key}.file '{ref['file']}' is not valid YAML: {exc}")
        else:
            prompts[key] = ref
    resolved["prompts"] = prompts

    if missing:
        raise ConfigurationError(missing)
    try:
        config = RuntimeConfig.model_validate(resolved)
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            problems.append(f"{loc}: {err['msg']}")
        raise ConfigurationError(problems) from None
    content = config.model_dump(mode="json")
    return ConfigSnapshot(config=config, content_hash=canonical_hash(content), source_path=str(path))


def snapshot_from_json(content: dict[str, Any]) -> ConfigSnapshot:
    config = RuntimeConfig.model_validate(content)
    return ConfigSnapshot(config=config, content_hash=canonical_hash(config.model_dump(mode="json")))


def hosted_profile_problems(snapshot: ConfigSnapshot, *, hosted: bool, allow_stub: bool) -> list[str]:
    problems: list[str] = []
    if hosted and not allow_stub:
        for name, provider in snapshot.config.providers.items():
            if provider.kind == "stub":
                problems.append(
                    f"providers.{name} uses the stub provider, which is refused on hosted profiles "
                    "(set ALLOW_STUB_PROVIDER=true only for infrastructure drills)"
                )
    return problems


def provider_credential_problems(snapshot: ConfigSnapshot) -> list[str]:
    problems: list[str] = []
    used = {role.provider for role in snapshot.config.roles.values()}
    emb = snapshot.config.retrieval.embedding
    if emb.provider == "openai" and emb.openai_provider:
        used.add(emb.openai_provider)
    for name in sorted(used):
        provider = snapshot.config.providers[name]
        if provider.kind == "stub":
            continue
        if not provider.api_key_env:
            problems.append(f"providers.{name}.api_key_env must name the environment variable holding the credential")
        elif not os.environ.get(provider.api_key_env):
            problems.append(f"{provider.api_key_env} is required by providers.{name} but not set")
    return problems
