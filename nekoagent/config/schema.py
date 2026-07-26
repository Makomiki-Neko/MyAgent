"""Pydantic 校验 schema。

约定：
- inner 模型 extra="ignore"，避免 YAML 里新增字段破坏加载（向后兼容）。
- 全部数量/阈值字段使用严格类型，Pydantic 报错时给出友好中文消息。
- LLMProfile 允许 api_key 或 api_key_env 二选一；运行时由 loader 解析出最终 key。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LLMProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str
    api_key: str | None = None             # 明文落盘（MVP 简化）
    api_key_env: str | None = None         # 可选：优先从环境变量解析
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ensure_key_source(self) -> "LLMProfile":
        # MVP：放宽要求；本地兼容服务（如 Ollama）可完全不配 key
        if not self.api_key and not self.api_key_env:
            self.api_key = "dummy-not-set"
        return self


class AuxiliaryModels(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary_llm_profile: str
    embedding_model_path: str
    reranker_model_path: str
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    rerank_batch_size: int = Field(default=16, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm_profile: str
    system_prompt_file: str
    mcp_servers: list[str] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    backend: str = "mysql"
    url: str
    echo: bool = False


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summarize_token_threshold: int = Field(default=6000, gt=0)
    suppress_archived_after_summary: bool = True
    user_memory_namespace: str = "main_user_memory"
    proc_memory_namespace: str = "supervisor_proc"


class RAGConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chroma_persist_dir: str
    bm25_index_path: str
    top_k_before_rerank: int = Field(default=20, gt=0)
    top_k_after_rerank: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    inject_when_above_threshold: bool = True


class OutputRetryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_attempts: int = Field(default=3, ge=1)
    backoff_initial: float = Field(default=0.5, gt=0)
    backoff_max: float = Field(default=5.0, gt=0)


class TaskQueueConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_concurrent: int = Field(default=4, gt=0)
    task_timeout_seconds: int = Field(default=1800, gt=0)
    poll_interval_seconds: float = Field(default=0.2, gt=0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: str = "INFO"
    file: str = "./logs/nekoagent.log"
    rotate: bool = True
    max_bytes: int = Field(default=10485760, gt=0)
    backup_count: int = Field(default=5, gt=0)
    expose_stacktrace_to_dev_log: bool = True

    @field_validator("level")
    @classmethod
    def _level_ok(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"logging.level 仅允许 {sorted(allowed)}，收到 {v}")
        return upper


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transport: str = "stdio"
    # stdio 字段
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # SSE 字段
    url: str | None = None
    host: str | None = None
    port: int | None = None


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str = "0.1.0"
    llm_profiles: dict[str, LLMProfile]
    auxiliary_models: AuxiliaryModels
    agents: dict[str, AgentConfig]
    database: DatabaseConfig
    memory: MemoryConfig
    rag: RAGConfig
    skills_dir: str = "./skills"
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    output_retry: OutputRetryConfig = Field(default_factory=OutputRetryConfig)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _validate_agent_refs(self) -> "RootConfig":
        for name, agent in self.agents.items():
            if agent.llm_profile not in self.llm_profiles:
                raise ValueError(
                    f"agents.{name}.llm_profile = '{agent.llm_profile}' 不在 llm_profiles 列表中"
                )
            aux = self.auxiliary_models
            if aux.summary_llm_profile not in self.llm_profiles:
                raise ValueError(
                    f"auxiliary_models.summary_llm_profile = '{aux.summary_llm_profile}' 不在 llm_profiles 列表中"
                )
        for name, agent in self.agents.items():
            for srv in agent.mcp_servers:
                if srv not in self.mcp_servers:
                    raise ValueError(
                        f"agents.{name}.mcp_servers 引用了未定义的 MCP 服务器 '{srv}'"
                    )
        return self


class PersonaConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    version: int = 1
    description: str = ""
    system_prompt: str