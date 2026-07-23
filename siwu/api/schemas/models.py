"""思悟 Agent -- 核心数据模型（6阶段认知循环）"""
from __future__ import annotations
from dataclasses import field as dc_field
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════
# 改进四：推导链模型 —— 从事实到判断的推理过程显式化
# ═══════════════════════════════════════════════════════════════════

class DerivationStep(BaseModel):
    """推导链中的一步 —— 从事实到判断的原子推理"""
    step_id: str = ""
    fact_basis: list[str] = Field(default_factory=list)
    inference: str = ""
    conclusion: str = ""
    confidence: float = 0.7
    reversible: bool = True
    # I线：迭代增量维护
    depends_on_step_ids: list[str] = Field(default_factory=list)  # 此步骤的前提是哪些步骤的结论
    needs_revalidation: bool = False  # 上游步骤被修正后，此步骤的结论待重验证


class DerivationChain(BaseModel):
    """完整的推导链 —— 从事实到最终判断的推理路径"""
    chain_id: str = ""
    steps: list[DerivationStep] = Field(default_factory=list)
    summary: str = ""
    factual_foundation: list[str] = Field(default_factory=list)
    generated_at_iteration: int = 0
    # I线：演化追踪
    parent_chain_id: Optional[str] = None  # 若此链是对历史链的修正，记录原链 chain_id
    revision_reason: str = ""              # 修正原因（哪条新事实触发了这次 fork）


# ═══════════════════════════════════════════════════════════════════
# 改进一：解剖麻雀 —— 典型案例模型
# ═══════════════════════════════════════════════════════════════════

class IllustrativeCase(BaseModel):
    """解剖麻雀 —— 能浓缩全局矛盾的典型案例"""
    name: str = ""
    description: str = ""
    why_typical: str = ""
    deep_analysis: str = ""
    contradictions_revealed: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    lessons_generalizable: str = ""


class ContradictionType(str, Enum):
    INTERNAL = "internal"; EXTERNAL = "external"
    PRIMARY = "primary"; SECONDARY = "secondary"

class CognitivePhaseName(str, Enum):
    INVESTIGATION = "investigation"
    CONTRADICTION = "contradiction"
    RATIONAL      = "rational"
    DECISION      = "decision"
    PRACTICE      = "practice"
    REFLECTION    = "reflection"

_CREDIBILITY_MAP = {
    "高": 0.9, "high": 0.9,
    "中": 0.5, "中等": 0.5, "medium": 0.5,
    "低": 0.2, "low": 0.2,
}

def _parse_credibility(raw) -> float:
    if raw is None: return 0.5
    if isinstance(raw, (int, float)): return max(0.0, min(1.0, float(raw)))
    s = str(raw).strip().lower()
    if s in _CREDIBILITY_MAP: return _CREDIBILITY_MAP[s]
    try: return max(0.0, min(1.0, float(s)))
    except ValueError: pass
    for k, v in _CREDIBILITY_MAP.items():
        if k in s: return v
    return 0.5

# --- Question Preprocessing ---
class PreprocessedQuestion(BaseModel):
    """问题预处理结果：在调查之前对用户问题进行深度解析"""
    original_question: str = ""
    expanded_question: str = ""
    question_intent: str = ""
    question_domains: list[str] = Field(default_factory=list)
    contradiction_in_question: str = ""
    core_anxiety: str = ""
    questionable_premises: list[str] = Field(default_factory=list)
    overlooked_factors: list[str] = Field(default_factory=list)
    wants_detailed_report: bool = False
    structured_sub_questions: list[str] = Field(default_factory=list)
    # 智能体主动提问（即时）：预处理判定必须先问用户才能答好时填入 1-3 条问题，否则空
    clarifying_questions: list[str] = Field(default_factory=list)

    # ── 任务性质与阶段必要性（v0.2.0 - 五步预处理管线）──
    task_nature: str = ""
    # 取值：code_generation | fact_lookup | causal_explanation |
    #       comparison_decision | exploration_understanding |
    #       creative_design | other
    task_complexity: str = "standard"
    # 取值：simple | standard | complex

    # 各阶段必要性（由 Step 2 查表 + complexity 微调确定）
    # 取值：required | light | skip
    investigation_necessity: str = "required"
    contradiction_necessity: str = "required"
    rational_necessity: str = "required"
    decision_necessity: str = "required"
    practice_necessity: str = "required"
    reflection_necessity: str = "required"
    # ── 是否需调查：LLM 能否仅凭自身知识直接回答？false → 跳过全部阶段，直接答
    needs_investigation: bool = True

# --- Investigation ---
class Fact(BaseModel):
    content: str
    source_type: str = "unknown"
    credibility: float = 0.5
    related_to: list[str] = Field(default_factory=list)
    id: str = ""
    # 改进一：解剖麻雀标记
    is_illustrative_case: bool = False
    case_name: str = ""
    @field_validator("credibility", mode="before")
    @classmethod
    def coerce_credibility(cls, v): return _parse_credibility(v)

class InformationGap(BaseModel):
    description: str; importance: str = "medium"; suggested_query: str = ""
    @field_validator("suggested_query", "description", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v): return v or ""

class FactReport(BaseModel):
    facts: list[Fact] = Field(default_factory=list)
    gaps: list[InformationGap] = Field(default_factory=list)
    summary: str = ""; raw_context: str = ""; autonomous_assessment: str = ""
    # 改进一：解剖麻雀
    illustrative_case: Optional[IllustrativeCase] = None

# --- System Model ---
class SystemElement(BaseModel):
    name: str = ""; description: str = ""; function_in_system: str = ""
    property_outside_system: str = ""; based_on_fact_ids: list[str] = Field(default_factory=list)

class SystemRelationship(BaseModel):
    source_element: str = ""; target_element: str = ""; relationship_type: str = ""
    description: str = ""; direction: str = "bidirectional"; intensity: str = "medium"
    is_contradiction_source: bool = False; based_on_fact_ids: list[str] = Field(default_factory=list)

class FeedbackLoop(BaseModel):
    description: str = ""; loop_type: str = "positive"
    elements_involved: list[str] = Field(default_factory=list)
    mechanism: str = ""; effect_on_system: str = ""

class EmergentProperty(BaseModel):
    property_name: str = ""; description: str = ""
    emerges_from: list[str] = Field(default_factory=list)
    cannot_be_reduced_to: str = ""; based_on_fact_ids: list[str] = Field(default_factory=list)

class SystemModel(BaseModel):
    system_boundary: str = ""; external_environment: str = ""
    elements: list[SystemElement] = Field(default_factory=list)
    relationships: list[SystemRelationship] = Field(default_factory=list)
    feedback_loops: list[FeedbackLoop] = Field(default_factory=list)
    emergent_properties: list[EmergentProperty] = Field(default_factory=list)
    uncertainty_areas: list[str] = Field(default_factory=list)

# --- Contradiction ---
class Contradiction(BaseModel):
    description: str; tension_poles: list[str] = Field(default_factory=list)
    contradiction_type: ContradictionType = ContradictionType.SECONDARY; rank: int = 1
    primary_aspect: str = ""; transformation_condition: str = ""
    basis_fact_ids: list[str] = Field(default_factory=list); basis_summary: str = ""
    involving_elements: list[str] = Field(default_factory=list)
    position_in_feedback: str = ""; systemic_drive: str = ""
    # 改进五：矛盾特殊性（此时此事此地的具体形式）
    particularity_description: str = ""
    # 改进四：从事实到矛盾判断的推导过程
    derivation_chain: Optional[DerivationChain] = None

class ContradictionPositionShift(BaseModel):
    """矛盾地位转换事件——次要矛盾升为主要矛盾，或反之"""
    from_role: str = ""                      # "principal" 或 "secondary"
    to_role: str = ""
    contradiction_description: str = ""      # 发生转换的矛盾描述
    trigger_facts: list[str] = Field(default_factory=list)  # 触发转换的新事实
    trigger_iteration: int = 0               # 发生在第几轮迭代
    transformation_condition_met: str = ""   # 哪个转化条件被满足了


class ContradictionGraph(BaseModel):
    principal_contradiction: Optional[Contradiction] = None
    secondary_contradictions: list[Contradiction] = Field(default_factory=list)
    dynamic_note: str = ""; synthesis: str = ""; autonomous_contradiction_view: str = ""
    system_model: Optional[SystemModel] = None
    # 改进四：矛盾分析阶段的整体推导链
    contradiction_derivation: Optional[DerivationChain] = None
    # I线：跨迭代矛盾地位转换记录
    position_shifts: list[ContradictionPositionShift] = Field(default_factory=list)
    iteration: int = 1  # 本图是第几轮迭代的产物

    @property
    def all_contradictions(self) -> list[Contradiction]:
        result = []
        if self.principal_contradiction: result.append(self.principal_contradiction)
        result.extend(self.secondary_contradictions)
        return result

# --- Rational ---
class RationalSynthesis(BaseModel):
    essence: str = ""; patterns: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list); synthesis_text: str = ""
    contradiction_motion: str = ""; quantitative_changes: list[str] = Field(default_factory=list)
    qualitative_threshold: str = ""; negation_of_negation: str = ""; fact_foundation: str = ""
    # 改进三：辩证运动 —— 从抽象回到具体
    abstract_from: str = ""
    return_to_concrete: str = ""
    unexplained_phenomena: list[str] = Field(default_factory=list)

# --- Decision ---
class ActionItem(BaseModel):
    description: str; priority: int = 1; timeline: str = ""; expected_outcome: str = ""
    targets_contradiction: str = ""; contradiction_resolution: str = ""; based_on_facts: str = ""
    practice_feasibility: str = "unknown"   # "direct" | "indirect" | "unknown" — 决策阶段预判
    suggested_practice_form: str = ""        # 智能体建议的实践形式描述
    why_cannot_practice: str = ""            # 如果 indirect，需要用户在现实中做什么

class DecisionReport(BaseModel):
    strategic_assessment: str = ""; tactical_plan: str = ""
    action_items: list[ActionItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list); summary: str = ""
    autonomous_recommendation: str = ""
    # 智能体主动提问：交付实践前需向用户确认的关键信息（仅在确有必要时非空）
    clarifying_questions: list[str] = Field(default_factory=list)

# --- Perspectives ---
class PerspectiveView(BaseModel):
    perspective_name: str; critique: str
    key_points: list[str] = Field(default_factory=list)
    consensus: list[str] = Field(default_factory=list); divergence: list[str] = Field(default_factory=list)

class PerspectiveSynthesis(BaseModel):
    views: list[PerspectiveView] = Field(default_factory=list)
    synthesized_insight: str = ""; critical_warnings: list[str] = Field(default_factory=list)
    consensus_points: list[str] = Field(default_factory=list); divergence_points: list[str] = Field(default_factory=list)

# --- Practice ---
class PracticeStep(BaseModel):
    description: str; action_taken: str = ""; observed_result: str = ""; matched_expectation: bool = True

class PracticeReport(BaseModel):
    mode: str = "executed"                 # "executed" | "partial" | "epistemic_only"
    steps_taken: list[PracticeStep] = Field(default_factory=list)
    observed_outcomes: list[str] = Field(default_factory=list)
    unexpected_findings: list[str] = Field(default_factory=list)
    practice_summary: str = ""
    success_indicators: list[str] = Field(default_factory=list)
    failure_indicators: list[str] = Field(default_factory=list)
    # 改进六：实践对矛盾分析的反馈
    contradiction_feedback: list[dict] = Field(default_factory=list)
    # ── 从 PracticeBoundaryReport 合并过来 ──
    analysis_summary: str = ""
    confidence_ceiling: str = ""           # "V2" 或 "V3"；V3=有直接实践检验，V2=仅有知性分析
    claim_assessments: list[dict] = Field(default_factory=list)
    unexpected_insights: list[str] = Field(default_factory=list)
    real_world_practice_needed: list[RealWorldPracticeTask] = Field(default_factory=list)
    reinvestigation_needed: bool = False
    reinvestigation_focus: str = ""

# --- Practice Boundary ---
class RealWorldPracticeTask(BaseModel):
    hypothesis: str = ""; why_important: str = ""; practice_method: str = ""
    observable_outcome: str = ""; estimated_duration: str = ""; why_agent_cannot: str = ""

class PracticeBoundaryReport(BaseModel):
    mode: str = "boundary"; analysis_summary: str = ""; confidence_ceiling: str = "V2"
    claim_assessments: list[dict] = Field(default_factory=list)
    unexpected_insights: list[str] = Field(default_factory=list)
    real_world_practice_needed: list[RealWorldPracticeTask] = Field(default_factory=list)
    reinvestigation_needed: bool = False; reinvestigation_focus: str = ""
    # 改进六：实践对矛盾分析的反馈
    contradiction_feedback: list[dict] = Field(default_factory=list)

# --- Reflection ---
class ReflectionReport(BaseModel):
    quality_assessment: str = ""; cognitive_biases_found: list[str] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list); should_reinvestigate: bool = True
    reinvestigation_focus: str = ""; convergence_score: float = 0.5
    investigation_retrospective: str = ""; contradiction_retrospective: str = ""
    decision_retrospective: str = ""
    skip_phases: list[str] = Field(default_factory=list)
    focus_hints: dict[str, str] = Field(default_factory=dict); recommended_mode: str = ""
    contradiction_stability: float = 0.5; contradiction_shift_detected: bool = False
    contradiction_shift_description: str = ""; understanding_level: str = "感性"
    qualitative_leap: bool = False; level_progression: str = ""
    final_answer: str = ""
    recommend_detailed_report: bool = False           # 反思自荐：本问题值得输出详实报告
    # S线：Steering 集成输出
    updated_effective_question: str = ""              # 吸收 steering 后更新的有效问题
    phases_to_redo: list[str] = Field(default_factory=list)  # 下一轮需要重做的阶段列表
    steering_summary: str = ""                         # 本轮 steering 的集成摘要
    steering_impact: str = ""                          # "positive" / "neutral" / "negative"
    skill_draft_candidates: list[dict] = Field(default_factory=list)  # 反思蒸馏候选技能
    practice_report_integrated: bool = False           # 本轮是否处理了 practice_report 类型的 steering
    contradiction_feedback_from_steering: list[dict] = Field(default_factory=list)  # 从 practice_report 提取的矛盾反馈

# --- Trace ---
class TraceMetadata(BaseModel):
    session_id: str = ""; start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None; total_tokens_used: int = 0; iterations: int = 0
    phase_durations: dict[str, float] = Field(default_factory=dict)
    credibility_chain_summary: str = ""

class CognitiveTrace(BaseModel):
    investigation: Optional[FactReport] = None
    contradictions: Optional[ContradictionGraph] = None
    rational_synthesis: Optional[RationalSynthesis] = None
    decision: Optional[DecisionReport] = None
    perspectives: Optional[PerspectiveSynthesis] = None
    practice: Optional[PracticeReport] = None
    reflection: Optional[ReflectionReport] = None
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    @property
    def has_practice(self) -> bool:
        return self.practice is not None
    @property
    def practice_mode(self) -> str:
        if self.practice is not None:
            return self.practice.mode
        return "none"

# --- Response ---
class GeneratedFile(BaseModel):
    path: str = ""
    description: str = ""
    size_bytes: int = 0

# ═══════════════════════════════════════════════════════════════════
# 技能系统数据模型
# ═══════════════════════════════════════════════════════════════════

class SkillMetadata(BaseModel):
    """技能元数据"""
    name: str
    description: str
    type: str = "execution"
    version: str = "1.0.0"
    active_phases: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    hard_rules: list[str] = Field(default_factory=list)
    not_for: list[str] = Field(default_factory=list)
    created_by: str = "human"
    created_at: str = ""
    status: str = "draft"
    validation_count: int = 0
    validation_required: int = 3
    usage_count: int = 0
    success_rate: float = 0.0
    last_used_at: Optional[str] = None
    file_path: str = ""
    depends_on: list[str] = Field(default_factory=list)


class SkillBody(BaseModel):
    """技能正文"""
    metadata: SkillMetadata
    body: str
    phase_methods: dict[str, str] = Field(default_factory=dict)


class SkillDraftCandidate(BaseModel):
    """反思阶段蒸馏出的候选技能"""
    suggested_name: str
    suggested_description: str
    suggested_type: str = "execution"
    suggested_active_phases: list[str] = Field(default_factory=list)
    trigger_pattern: str
    core_operations: str
    extracted_from_session: str = ""
    confidence: float = 0.5


class SkillValidationRecord(BaseModel):
    """技能草稿的单次验证记录"""
    session_id: str
    question: str
    practice_outcome: str
    notes: str = ""
    validated_at: str = ""


class SkillUsageEvent(BaseModel):
    """技能使用事件"""
    skill_name: str
    skill_version: str
    phase: str
    session_id: str
    outcome: str
    notes: str = ""
    used_at: str = ""


class AgentResponse(BaseModel):
    summary: str; action_items: list[str] = Field(default_factory=list)
    full_trace: Optional[CognitiveTrace] = None
    session_id: str = ""; question: str = ""; conversation_id: str = ""
    generated_files: list[GeneratedFile] = Field(default_factory=list)
