"""
Pydantic config schema for validation.

Validates config.yaml at startup so typos and missing fields are caught
immediately with clear error messages, not as KeyErrors mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SamplingConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 512
    min_p: float = 0.0
    presence_penalty: float = 0.0


class DedupeConfig(BaseModel):
    minhash_threshold: float = Field(0.7, ge=0.0, le=1.0)
    minhash_num_perm: int = Field(128, ge=16)


class InstructionGenerationConfig(BaseModel):
    n: int = Field(20000, ge=1)
    quick_n: int = Field(1000, ge=1)
    chunk_size: int = Field(5000, ge=1)
    sampling: SamplingConfig
    min_length: int = Field(10, ge=0)
    max_length: int = Field(2048, ge=1)


class ResponseGenerationConfig(BaseModel):
    L: int = Field(3, ge=1, alias="L")
    chunk_size: int = Field(1000, ge=1)
    sampling: SamplingConfig

    model_config = {"populate_by_name": True}


class FilteringConfig(BaseModel):
    chunk_size: int = Field(1000, ge=1)
    min_score: float = Field(3.0, ge=1.0, le=5.0)
    sampling: SamplingConfig


class CommitteeMember(BaseModel):
    type: str
    model: str
    base_url: str | None = None
    api_key: str | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"vllm_local", "openai_api"}
        if v not in allowed:
            raise ValueError(f"Committee member type must be one of {allowed}, got '{v}'")
        return v


class CommitteeConfig(BaseModel):
    generators: list[CommitteeMember] = Field(min_length=1)
    judges: list[CommitteeMember] = Field(min_length=1)


class SystemPromptEntry(BaseModel):
    prompt: str
    weight: float = Field(ge=0)


class PublicSource(BaseModel):
    dataset: str
    split: str = "train"
    n: int = Field(5000, ge=1)
    streaming: bool = True


class PublicInstructionsConfig(BaseModel):
    min_length: int = Field(10, ge=0)
    max_length: int = Field(2048, ge=1)
    sources: list[PublicSource] = Field(min_length=1)


class FollowupSamplingConfig(BaseModel):
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 30
    max_tokens: int = 512


class MultiturnConfig(BaseModel):
    fraction: float = Field(0.3, ge=0.0, le=1.0)
    min_turns: int = Field(2, ge=1)
    max_turns: int = Field(5, ge=1)
    chunk_size: int = Field(500, ge=1)
    followup_sampling: FollowupSamplingConfig = FollowupSamplingConfig()

    @model_validator(mode="after")
    def validate_turns(self) -> "MultiturnConfig":
        if self.min_turns > self.max_turns:
            raise ValueError(f"min_turns ({self.min_turns}) > max_turns ({self.max_turns})")
        return self


class MixingConfig(BaseModel):
    domain_fraction: float = Field(0.17, gt=0.0, lt=1.0)
    domain_data_path: str
    val_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = 42


class PathsConfig(BaseModel):
    instructions: str
    public_instructions: str
    candidates: str
    replay: str
    public_replay: str = ""
    multiturn: str
    train: str
    val: str


class PipelineConfig(BaseModel):
    model: str
    tensor_parallel_size: int = Field(1, ge=1)
    seed: int = 42
    dedupe: DedupeConfig = DedupeConfig()
    stop_tokens: list[str] = ["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"]
    instruction_generation: InstructionGenerationConfig
    response_generation: ResponseGenerationConfig
    filtering: FilteringConfig
    committee: CommitteeConfig
    replay_system_prompts: list[SystemPromptEntry] = Field(min_length=1)
    public_instructions: PublicInstructionsConfig | None = None
    multiturn: MultiturnConfig = MultiturnConfig()

    @field_validator("replay_system_prompts")
    @classmethod
    def validate_weights_nonzero(cls, v: list[SystemPromptEntry]) -> list[SystemPromptEntry]:
        total = sum(e.weight for e in v)
        if total <= 0:
            raise ValueError("Total weight of replay_system_prompts must be > 0")
        return v
    mixing: MixingConfig
    paths: PathsConfig
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()


def load_and_validate_config(config_path: str) -> dict:
    """Load config.yaml, validate with pydantic, return raw dict.

    Raises a clear error on any validation failure.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    try:
        PipelineConfig(**raw)
    except Exception as e:
        raise ValueError(f"Config validation failed:\n{e}") from e

    return raw
