# 多Agent用户交互系统 \- 全局基准规范 \

## Purpose

本文档为基于 Python、FastAPI、LangGraph 构建的三级多Agent交互系统全局强制规范。定义系统整体架构、Agent角色边界、记忆体系、人机交互流程、长任务调度规则、人在回路机制、MVP交付标准。项目所有模块SPEC、代码实现、流程设计**必须完全遵循本文档约束**，为全项目唯一全局真相源。

系统核心架构：**用户交互主Agent → 复杂任务主管Agent → 执行子Agent**，实现人设对话、情绪价值、长期记忆、任务拆分调度、人在回路确认、多任务并发、任务状态回溯能力，MVP阶段基于命令行完成全流程闭环，预留FastAPI接口扩展能力。

## Global Technical Stack Requirements

### Requirement: 核心技术栈强制约束

系统 **MUST** 基于 Python、FastAPI、LangGraph 构建；所有Agent状态流转、节点编排、任务流程管理 **MUST** 由 LangGraph 状态机驱动；所有对外服务能力**MUST** 通过 FastAPI 统一暴露，禁止自定义原生状态管理。

### Requirement: 模块化解耦约束

系统分层严格隔离：对话层、Agent编排层、记忆管理层、任务调度层、接口层、工具执行层。各模块 **SHALL** 通过标准化接口通信，禁止跨层直接调用内部逻辑、禁止硬编码依赖。

## Global Architecture \& Agent Boundary Requirements

### Requirement: 三级Agent固定架构

系统 **MUST** 严格采用三级分层架构，层级职责不可逆、不可越权；用户仅与主Agent交互，所有复杂规划类任务统一下沉至主管Agent，原子执行任务交由子Agent完成。

- **主Agent（用户交互层）**：唯一用户入口，负责人设对话、情绪价值、简单工具调用、任务中转、人机交互、状态查询应答

- **复杂任务主管Agent（调度规划层）**：仅负责任务理解、计划拆解、子任务编排、进度管理、结果汇总，不直接对接用户

- **执行子Agent（原子执行层）**：仅执行主管分配的原子任务，无规划、无决策、无用户交互能力

### Requirement: 主Agent职责严格边界

主Agent **SHALL** 维持全局固定人设，持续为用户提供情绪价值、闲聊交互、基础信息查询；**MUST NOT** 自主执行需要多步骤规划、多阶段执行的长任务。所有复杂任务 **MUST** 提交至复杂任务主管Agent处理。

主Agent在后台长任务执行期间，**MUST** 保持正常响应用户新对话、新请求，实现任务后台运行与前台交互并行不阻塞。

### Requirement: 复杂任务主管Agent职责严格边界

复杂任务主管Agent **SHALL**仅负责复杂任务的需求解析、任务拆分、依赖排序、子Agent调度、执行监控、结果汇总；**MUST NOT** 直接与用户对话、**MUST NOT** 直接执行工具原子操作。

### Requirement: 执行子Agent职责严格边界

执行子Agent **SHALL** 仅接收主管Agent下发的原子任务，完成单一执行动作并返回结果；无任务规划、无全局上下文、无用户交互权限。

## Global Memory System Requirements

### Requirement: 主Agent全维度长期记忆

主Agent **MUST** 具备持久化长期记忆能力，覆盖三类核心记忆，服务用户个性化、连续性对话体验：

- **情景记忆**：存储用户历史对话、交互场景、个人偏好、历史行为

- **语义记忆**：存储用户核心信息、知识偏好、固定设定

- **程序性记忆**：存储用户常用对话习惯、应答偏好

系统重启后记忆不丢失，每轮交互自动检索关联记忆并入对话上下文。

### Requirement: 复杂任务主管Agent程序性记忆

复杂任务主管Agent **MUST** 具备任务程序性记忆，存储历史任务的拆解逻辑、执行步骤、调度方案、异常处理范式；同类任务复用历史成熟方案，提升规划效率与稳定性。

### Requirement: 记忆隔离约束

主Agent记忆、主管Agent记忆 **MUST** 物理隔离、逻辑隔离，禁止跨Agent非法读写，仅允许标准化查询调用。

## Global Human\-In\-The\-Loop Requirements

### Requirement: 长任务强制人在回路确认

复杂任务主管Agent **MUST** 在生成完整任务拆解计划、任务清单后，立即暂停执行，通过回调通知主Agent；由主Agent向用户展示完整任务方案，提供**执行 / 修改 / 取消**三选项，等待用户指令后方可继续流转。

#### Scenario: 用户确认执行

\- **GIVEN**：主管Agent已生成合法任务计划，等待用户确认
\- **WHEN**：用户通过主Agent确认执行任务
\- **THEN**：主管Agent启动子任务调度，正常推进全流程

#### Scenario: 用户要求修改任务

\- **GIVEN**：存在待确认任务计划
\- **WHEN**：用户提出计划修改需求
\- **THEN**：主管Agent根据用户指令重构任务清单，重新生成计划后再次触发人在回路确认

#### Scenario: 用户取消任务

\- **GIVEN**：存在待确认或执行中任务
\- **WHEN**：用户发起取消指令
\- **THEN**：主管Agent立即终止任务、清空执行上下文、停止所有子任务调度

## Global Task Scheduling Requirements

### Requirement: 多任务并发队列能力

系统 **MUST** 支持消息队列形式提交多个复杂任务到复杂任务主管Agent；多任务上下文相互隔离，单任务异常不影响其他任务执行，实现后台并发调度。

### Requirement: 任务状态实时查询能力

系统 **MUST** 提供任务状态查询能力；主Agent可根据用户需求，实时查询任意后台任务的进度、当前执行节点、已完成子任务、异常日志、剩余步骤。

### Requirement: 任务完成主动通知机制

复杂任务主管Agent完成全部子任务执行、结果汇总后，**MUST** 主动回调通知主Agent；主Agent整理结果、结合人设优化话术，**主动向用户汇总报告任务结果**，无需用户主动查询。

## Global Agent Persona Requirements

### Requirement: 可自定义主Agent人设

系统 **SHALL** 支持通过自定义 System Prompt 配置主Agent全局人设；所有对话输出、情绪表达、应答风格、交互语气 **MUST** 严格遵循人设配置，全局统一生效。

人设配置支持动态更新，更新后即时生效，不影响历史记忆数据。

## Global MVP Phase Requirements

### Requirement: MVP命令行交互闭环

MVP阶段 **MUST** 基于命令行实现完整功能闭环，覆盖：人设对话、情绪交互、长任务提交、任务计划人在回路确认、任务状态查询、后台并发执行、任务完成主动通知、长期记忆生效。

### Requirement: MVP接口预留规范

MVP阶段 **MUST** 完整保留 FastAPI 接口架构，预留人设配置、记忆管理、任务提交、任务查询、任务操控标准化接口，为后续Web端、客户端扩展提供基础。

## Global Code \& Engineering Requirements

### Requirement: 配置外置规范

所有可变参数：Agent人设Prompt、模型参数、记忆存储配置、队列配置、任务超时参数、日志级别 **MUST** 外置配置文件，禁止业务代码硬编码。

### Requirement: 日志与异常规范

所有Agent流转、任务调度、记忆读写、用户交互**MUST** 留存完整链路日志；所有异常必须捕获处理，对用户输出友好文案，禁止暴露原始堆栈报错。

### Requirement: 可追溯规范

所有功能开发、流程实现 **MUST** 完全对标本全局SPEC，所有模块子SPEC必须继承、细化本文档约束，不得违反全局架构与边界规则。

