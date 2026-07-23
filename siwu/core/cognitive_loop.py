"""思悟 Agent —— 认知循环控制器"""
from __future__ import annotations
import asyncio, uuid
from datetime import datetime
from typing import AsyncIterator, Callable, Optional
import structlog
from ..api.schemas.models import AgentResponse, CognitiveTrace, CognitivePhaseName, TraceMetadata
from ..config import settings
from ..llm import get_llm, get_phase_llm
from ..llm.base import BaseLLM
from ..memory.episodic_memory import EpisodicMemory
from ..memory.working_memory import WorkingMemory
from ..tools.filesystem import WorkspaceToolkit
from .contradiction import ContradictionAnalyzer
from .decision import DecisionEngine
from .dev_tracer import DevTracer, TracingLLMWrapper
from .investigation import InvestigationModule
from .loop_controller import (
    LoopController, get_controller, release_controller,
    register_conv_controller, release_conv_controller,
)
from .perspectives import MultiPerspectiveReview
from .practice import PracticeModule
from .question_preprocessing import QuestionPreprocessing
from .rational import RationalCognitionModule
from .reflection import ReflectionEngine
from .skill_manager import SkillManager

log = structlog.get_logger(__name__)

# 智能体主动提问时，最多阻塞等待用户回答的秒数（超时后按现有信息继续）
_CLARIFICATION_TIMEOUT = 900


def _extract_keywords(text: str) -> list[str]:
    """从文本中提取核心关键词（汉字相邻2-gram + 英文词）
    
    对中文使用滑动二字组（2-gram）而非整段连续汉字取为一个关键词，
    因为后者会捕获整句（如"如何判断一个数是质数"）作为一个关键词，
    导致子串匹配过于严格，覆盖率检查永远为0。
    2-gram 是无需分词库的中文近似方案。
    """
    import re
    result = []
    # 对每段连续 CJK 字符提取滑动 2-gram
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        if len(run) >= 2:
            for i in range(len(run) - 1):
                result.append(run[i:i+2])
    # 英文词（>=3字符）
    en_words = re.findall(r'[a-zA-Z]{3,}', text)
    result += [w.lower() for w in en_words]
    return result


def _text_overlap(text_a: str, text_b: str, threshold: float = 0.3) -> bool:
    """两段文本的关键词重叠度是否超过阈值"""
    kw_a = set(_extract_keywords(text_a))
    kw_b = set(_extract_keywords(text_b))
    if not kw_a:
        return False
    overlap = len(kw_a & kw_b) / len(kw_a)
    return overlap >= threshold


_FINAL_ANSWER_PROMPT = """
你正在为用户提供最终回答。你不是在写工作总结，而是在直接回答用户的问题。

核心原则：用户要的是答案，不是过程日志。

你的任务：
根据下方提供的完整分析材料（调查研究事实、矛盾分析、理性认识、决策方案），
直接、完整、诚实地回答用户的原始问题。

要求：
- 直接面向用户的问题给出答案，不要说"我做了X"、"经过分析发现"这类过程性语言
- 如果有明确的结论，先说结论再用事实支撑
- 如果证据不足以给出确定答案，诚实说明不确定性，并指出还需要什么信息
- 综合调查事实作为论据，体现矛盾分析的结构，纳入理性认识的本质规律
- 回答应自然流畅，300-800字，使用中文
- 不要在回答中提及"反思阶段"、"调查阶段"等内部流程术语
- 不要写"综上所述"、"总的来说"这类套话结尾，说完就结束

CRITICAL — 严禁以下行为：
- 严禁输出任何对你自己的指令描述（如"根据要求""我被要求""按照提示""根据给定的材料"等）
- 严禁输出任何元分析（如"要点如下""我注意到""关键发现是"等分析和组织性语言）
- 严禁提及任何 prompt 内容（如"你是""你的任务""核心原则"等对系统指令的引用）
- 严禁使用"我们被要求""按要求""依据指示"等暴露存在指令的表述
- 直接对用户说，不要解释你怎么得出的结论
如果你发现自己写了上述任何模式，删除并重写。

输出：纯文本段落，不需要 JSON 包裹，不需要 Markdown 标题。
"""


_DETAILED_REPORT_PROMPT = """
你正在为用户提供一份详实的最终回答（报告形式）。你不是在写工作总结，而是在直接、完整地回答用户的问题。

核心原则：用户要的是一份信息完整、有据可依的答案，而不是过程日志。

根据下方完整分析材料（调查事实、矛盾分析、系统结构、理性认识、决策、实践检验），写一份详实的回答：
- 结论先行，再用事实与分析层层支撑
- 尽量保留有价值的信息：关键事实（可标注可信度/来源）、主要矛盾与被忽略的力量、事物的本质与规律、可行方向与风险、已被实践检验 vs 仍属推断的区分
- 允许使用 Markdown：## 小标题、- 要点列表、必要处 **加粗**，让长文更好读
- 直接面向用户的问题组织内容，该多长就多长，但不堆砌与问题无关的材料
- 诚实标注不确定性与信息缺口

严禁：过程性/元语言（如"经过分析""我做了X""根据材料"）、内部阶段术语（如"调查阶段""反思阶段"）、暴露 prompt 或指令、"综上所述"式套话结尾。
直接对用户说。输出 Markdown 正文，不要 JSON 包裹。
"""

_DETAILED_REPORT_KEYWORDS = (
    "详细报告", "详实", "详尽", "完整报告", "深入展开", "展开讲", "详细分析",
    "更详细", "详细点", "详细一点", "尽可能详细", "深度报告", "全面报告",
    "写一份报告", "出一份报告", "报告形式", "长一点的回答",
    "detailed report", "full report", "comprehensive", "in-depth", "in depth",
)


def _wants_detailed_report(text: str) -> bool:
    """从自然语言（用户问题或 steering）中检测是否在要求一份详实报告。"""
    if not text:
        return False
    low = text.lower()
    return any(k.lower() in low for k in _DETAILED_REPORT_KEYWORDS)

class CognitiveLoop:
    def __init__(self, llm=None, web_search_enabled=None, conversation_id="", review_strategy="", project_id=""):
        self.llm = llm or get_llm()
        _web = web_search_enabled if web_search_enabled is not None else settings.web_search_enabled
        self.conversation_id = conversation_id
        self.review_strategy = review_strategy or settings.review_strategy
        self.project_id = project_id
        self.episodic = EpisodicMemory()
        # 项目感知 workspace：指定 project_id 时用项目独立目录，否则用全局 workspace
        if project_id:
            _ws = settings.projects_dir / project_id / "workspace"
            _ws.mkdir(parents=True, exist_ok=True)
            self.workspace = WorkspaceToolkit(_ws)
        else:
            self.workspace = WorkspaceToolkit(settings.workspace_dir)
        self.tracer = DevTracer(enabled=settings.dev_enabled, log_dir=settings.dev_log_dir, console_output=settings.dev_console_output)
        def _llm(phase, tag=""):
            if llm is not None:
                raw = llm
            else:
                raw = get_phase_llm(phase)
            if settings.dev_enabled and not isinstance(raw, TracingLLMWrapper):
                return TracingLLMWrapper(raw, self.tracer, phase, tag=tag)
            return raw
        self.investigation = InvestigationModule(_llm("investigation"), settings.phase("investigation"), web_search_enabled=_web, workspace=self.workspace)
        self.preprocessing = QuestionPreprocessing(_llm("preprocessing"), settings.phase("preprocessing"))
        self.contradiction = ContradictionAnalyzer(_llm("contradiction"), settings.phase("contradiction"))
        self.rational = RationalCognitionModule(_llm("rational"), settings.phase("rational"))
        self.decision = DecisionEngine(_llm("decision"), settings.phase("decision"))
        self.perspectives = MultiPerspectiveReview(_llm("perspectives"), settings.phase("perspectives"))
        self.practice = PracticeModule(_llm("practice"), settings.phase("practice"), workspace=self.workspace, practice_rounds=settings.practice_rounds)
        self.reflection = ReflectionEngine(_llm("reflection"), settings.phase("reflection"))
        skills_dir = getattr(settings, "skills_dir", None)
        if skills_dir is None:
            from pathlib import Path
            skills_dir = Path("siwu/skills")
        self.skill_manager = SkillManager(skills_dir)
        self.max_iterations = settings.max_iterations
        self.convergence_threshold = 0.85
        self.enable_trajectory_logging = settings.enable_trajectory_logging




    async def _semantic_coverage_check(self, uncovered: list[str], all_text: str) -> dict:
        """对关键词未覆盖的子问题做轻量语义核实。

        关键词匹配无法识别换说法的回答（"质数"->"素数"、"遍历"->"循环查找"）。
        这一步只对关键词标记为"未覆盖"的剩余项做一轮聚焦的 LLM 核实，
        问法：这段文本有没有从实质上回答这个子问题？
        这是一个事实判断（文本到子问题的覆盖关系），不是让 LLM 自评收敛度，
        因此不构成自我评价循环。
        """
        if not uncovered or not all_text.strip():
            return {"still_uncovered": uncovered, "semantically_covered": []}

        import json
        items_json = json.dumps(
            [{"index": i, "question": q} for i, q in enumerate(uncovered)],
            ensure_ascii=False,
        )
        truncated = all_text[:3000]
        prompt = (
            "判断以下分析文本是否从实质上回答了列出的各子问题（措辞可能不同）。\n\n"
            "注意：\n"
            '- "实质性回答"指文本提供了该子问题所需的答案，即使使用的不是同样的词。\n'
            "- 如果文本完全没提相关内容，或者只顺带提到但没有回答，则视为未覆盖。\n\n"
            f"分析文本：\n{truncated}\n\n"
            f"子问题列表：\n{items_json}\n\n"
            '输出JSON：{{"results": [{{"index": 0, "covered": true/false, "reason": "简短原因"}}]}}'
        )
        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                system="你是一个客观的内容覆盖分析器。不做评价，只判断文本是否覆盖了子问题的实质性内容。",
                temperature=0.1, max_tokens=512,
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                idx = raw.find("\n")
                if idx >= 0:
                    raw = raw[idx + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip().rstrip("`")
            data = json.loads(raw)
            results = data.get("results", [])
            still_uncovered = []
            semantically_covered = []
            for r in results:
                idx = r.get("index")
                if idx is not None and idx < len(uncovered):
                    if r.get("covered", False):
                        semantically_covered.append(uncovered[idx])
                    else:
                        still_uncovered.append(uncovered[idx])
            log.info("cognitive_loop.semantic_coverage_check.done",
                     total=len(uncovered),
                     semantically_covered=len(semantically_covered),
                     still_uncovered=len(still_uncovered))
            return {"still_uncovered": still_uncovered, "semantically_covered": semantically_covered}
        except Exception as e:
            log.warning("cognitive_loop.semantic_coverage_check.error", error=str(e))
            return {"still_uncovered": uncovered, "semantically_covered": []}

    async def _gather_termination_evidence(self, trace: CognitiveTrace, working_mem: WorkingMemory) -> dict:
        """收集终止判定的结构性证据。
        
        先做关键词初筛（快速免费诚实），关键词未覆盖项再做轻量语义核实，
        因为完全换说法的回答不需要关键词重叠也能被识别。
        返回值喂入 Reflection prompt，为收敛判断提供事实基础。
        不做刚性判定——Reflection 保留最终判断权。
        """
        evidence = {
            "sub_question_coverage": None,
            "action_verification": None,
            "contradiction_status": None,
        }
        
        # ── 1. 子问题覆盖检查（关键词初筛 + 语义补充）──
        sub_questions = working_mem.get("structured_sub_questions") or []
        if sub_questions:
            covered = []
            uncovered = []
            all_text = ""
            if trace.investigation:
                all_text += trace.investigation.summary + " "
                for f in trace.investigation.facts:
                    all_text += f.content[:200] + " "
            if trace.rational_synthesis:
                all_text += trace.rational_synthesis.essence + " "
                all_text += " ".join(trace.rational_synthesis.patterns) + " "
                all_text += trace.rational_synthesis.synthesis_text
            if trace.decision:
                all_text += (trace.decision.summary or trace.decision.strategic_assessment) + " "
            if trace.reflection and trace.reflection.final_answer:
                all_text += trace.reflection.final_answer
            
            for sq in sub_questions:
                keywords = _extract_keywords(sq)
                match_count = sum(1 for kw in keywords if kw in all_text)
                if match_count >= max(1, len(keywords) * 0.25):
                    covered.append(sq)
                else:
                    uncovered.append(sq)
            
            # 关键词未覆盖项再做语义核实（处理换说法的情况）
            _semantic_covered = []
            if uncovered and all_text:
                _semantic_result = await self._semantic_coverage_check(uncovered, all_text)
                _semantic_covered = _semantic_result.get("semantically_covered", [])
                uncovered = _semantic_result.get("still_uncovered", uncovered)
                covered = covered + _semantic_covered
            
            _sq_data = {
                "covered": len(covered),
                "total": len(sub_questions),
                "uncovered_list": uncovered,
            }
            if _semantic_covered:
                _sq_data["semantically_covered"] = _semantic_covered
            evidence["sub_question_coverage"] = _sq_data
        
        # ── 2. 决策项验证覆盖检查 ──
        if trace.decision and trace.decision.action_items:
            verified = []
            unverified = []
            unverifiable = []
            
            practice_steps = trace.practice.steps_taken if trace.practice else []
            practice_summary = (trace.practice.practice_summary if trace.practice else "") + " "
            for s in practice_steps:
                practice_summary += s.description + " " + (s.observed_result or "") + " "
            
            for ai in trace.decision.action_items:
                target = ai.description
                found = False
                for s in practice_steps:
                    if _text_overlap(target, s.description, threshold=0.3):
                        found = True
                        break
                if not found and practice_summary:
                    found = _text_overlap(target, practice_summary, threshold=0.3)
                
                if found:
                    verified.append(target)
                elif ai.practice_feasibility == "direct" and ai.suggested_practice_form:
                    unverified.append(target)
                else:
                    unverifiable.append({
                        "item": target,
                        "why": ai.why_cannot_practice or "智能体无法自行执行此验证",
                    })
            
            evidence["action_verification"] = {
                "verified": len(verified),
                "total": len(trace.decision.action_items),
                "unverified_list": unverified,
                "unverifiable_list": unverifiable,
            }
        
        # ── 3. 矛盾解决状态检查 ──
        if trace.contradictions:
            all_contradictions = trace.contradictions.all_contradictions
            resolved = []
            pending = []
            
            for c in all_contradictions:
                chain = c.derivation_chain
                chain_complete = False
                needs_reval = False
                
                if chain and chain.steps:
                    chain_complete = all(s.conclusion for s in chain.steps)
                    needs_reval = any(s.needs_revalidation for s in chain.steps)
                
                if chain_complete and not needs_reval:
                    resolved.append(c.description[:80])
                else:
                    reason = ""
                    if not chain:
                        reason = "无推导链"
                    elif needs_reval:
                        reason = "推导链中有步骤待重验证"
                    else:
                        reason = "推导链不完整"
                    pending.append({"contradiction": c.description[:80], "reason": reason})
            
            evidence["contradiction_status"] = {
                "resolved": len(resolved),
                "total": len(all_contradictions),
                "pending_list": pending,
            }
        
        return evidence

    async def run(self, question, context="", mode="standard", on_phase=None, session_id=None, conversation_id="", review_strategy="", on_title=None, model_override="", files=None, project_id="", on_clarification=None):
        session_id = session_id or str(uuid.uuid4())[:8]
        conv_id = conversation_id or self.conversation_id
        effective_review = review_strategy or self.review_strategy
        log.info("cognitive_loop.review_strategy_resolved",
                 param=review_strategy, instance=self.review_strategy,
                 effective=effective_review)
        # 按模式调整最大迭代次数：深度模式给更多迭代空间
        mode_max_iter = {
            "fast": 1,
            "standard": self.max_iterations,
            "deep": max(self.max_iterations, 7),
        }.get(mode, self.max_iterations)
        self.tracer.set_session(session_id)
        controller = get_controller(session_id)
        # 建立 conversation_id -> session_id 映射，使 /control 端点能找到本次运行的控制器
        if conv_id:
            register_conv_controller(conv_id, session_id)
        log.info("cognitive_loop.start", session_id=session_id, mode=mode, q=question[:60], conversation=conv_id)
        trace = CognitiveTrace(metadata=TraceMetadata(session_id=session_id, start_time=datetime.now()))
        working_mem = WorkingMemory(session_id=session_id)
        self.skill_manager.inject_to_working_memory(working_mem)
        working_mem.set("question", question)
        working_mem.set("context", context)
        _project_id = project_id or self.project_id

        # ── 文件加载阶段（调查阶段之前）：用户上传文件 → Markdown 注入 ──
        if files:
            try:
                from ..tools.file_loader import FileLoader
                loader = FileLoader(max_total_chars=settings.file_loader_max_total_chars, workspace_dir=str(settings.workspace_dir))
                loaded_docs = loader.load(files)
                file_context = FileLoader.format_for_context(loaded_docs)
                n_ok = sum(1 for d in loaded_docs if not d.error)
                n_fail = sum(1 for d in loaded_docs if d.error)
                if file_context:
                    # 写入专用键，由 WorkingMemory.get_context_for_phase 在调查阶段 surface
                    working_mem.set("_uploaded_files_context", file_context)
                    working_mem.set("_loaded_files", [d.file_path for d in loaded_docs if not d.error])
                failed = [d for d in loaded_docs if d.error]
                if failed:
                    working_mem.set("_failed_files", [{"name": d.file_name, "error": d.error} for d in failed])
                # 通知前端：文件加载完成
                msg_parts = []
                if n_ok: msg_parts.append(str(n_ok) + " 个文件已加载（" + str(sum(d.char_count for d in loaded_docs if not d.error)) + " 字符）")
                if n_fail: msg_parts.append(str(n_fail) + " 个文件加载失败")
                if on_phase: on_phase("file_load", "、".join(msg_parts) if msg_parts else "文件加载完成")
                log.info("cognitive_loop.files_loaded",
                         n=len(loaded_docs), n_ok=n_ok, n_fail=n_fail)
            except Exception:
                log.warning("cognitive_loop.file_load_failed", exc_info=True)

        # ── 第零阶段：问题预处理 ──
        # 在进入调查之前，对用户问题进行深度解析——
        # 矛盾分析、广泛联系、揣度意图、结构化扩展
        # 关键：必须在预处理之前加载对话历史，否则短追问（"继续"等）LLM 无法理解指代
        conv_context = self.episodic.build_conversation_context(
            conversation_id=conv_id, current_question=question,
            max_turns=5, exclude_session=session_id,
        )
        # 此时才保存 initial episode —— 必须在 build_conversation_context 之后，
        # 否则标题生成条件 `not conv_context` 对新对话永远不成立。
        if conv_id:
            self._save_initial_episode(session_id, question, conv_id, _project_id)
            # 确保 conversation_meta 行存在并标记项目归属（项目分组的权威来源）
            self.episodic.set_conversation_project(conv_id, _project_id)
        if conv_context:
            working_mem.set("conversation_history", conv_context)
            log.info("cognitive_loop.conversation_context",
                     conversation=conv_id, len=len(conv_context))
        # 注入 Agent 角色设定和用户自定义知识（写入专用键，由 get_context_for_phase 全阶段 surface）
        persona_ctx = _load_persona_context()
        if persona_ctx:
            working_mem.set("_persona_context", persona_ctx)

        if on_phase: on_phase("preprocessing", "正在分析问题意图与结构")
        preprocessed = await self.preprocessing.preprocess(
            question, conversation_history=conv_context,
        )
        working_mem.set("preprocessed_question", preprocessed)
        working_mem.set("original_question", preprocessed.original_question)
        working_mem.set("expanded_question", preprocessed.expanded_question)
        working_mem.set("question_intent", preprocessed.question_intent)
        working_mem.set("question_domains", preprocessed.question_domains)
        working_mem.set("contradiction_in_question", preprocessed.contradiction_in_question)
        working_mem.set("core_anxiety", preprocessed.core_anxiety)
        working_mem.set("questionable_premises", preprocessed.questionable_premises)
        working_mem.set("overlooked_factors", preprocessed.overlooked_factors)
        working_mem.set("structured_sub_questions", preprocessed.structured_sub_questions)
        working_mem.set("report_requested", bool(getattr(preprocessed, "wants_detailed_report", False) or _wants_detailed_report(question)))

        # ── 根据预处理判断的阶段必要性调整执行计划 ──
        _phase_necessity = {}
        for _phase in ["investigation", "contradiction", "rational",
                       "decision", "practice", "reflection"]:
            _nec = getattr(preprocessed, f"{_phase}_necessity", "required")
            _phase_necessity[_phase] = _nec

        # skip_phases：完全跳过的阶段（现有循环体已支持 skip_phases 检查）
        _skips = [_p for _p, _n in _phase_necessity.items() if _n == "skip"]
        working_mem.set("skip_phases", _skips)

        # light_phases：轻量执行（提示各阶段模块可用更短的 prompt）
        _lights = [_p for _p, _n in _phase_necessity.items() if _n == "light"]
        working_mem.set("light_phases", _lights)

        # 记录任务性质供各阶段 prompt 自行适配
        working_mem.set("task_nature", getattr(preprocessed, "task_nature", ""))
        working_mem.set("task_complexity", getattr(preprocessed, "task_complexity", "standard"))

        if on_phase:
            _nature_display = {
                "code_generation": "代码生成", "fact_lookup": "事实查询",
                "causal_explanation": "因果解释", "comparison_decision": "比较决策",
                "exploration_understanding": "探索理解", "creative_design": "创造设计",
            }.get(getattr(preprocessed, "task_nature", ""), "")
            _nature_str = " [" + _nature_display + "]" if _nature_display else ""
            _needs_inv = getattr(preprocessed, "needs_investigation", True)
            _skip_info = ""
            if _skips:
                if not _needs_inv:
                    _skip_info = " → 无需调查，直接回答"
                elif len(_skips) >= 4:
                    _skip_info = " (跳" + "、".join(_skips) + ")"
                else:
                    _skip_info = " (跳" + "、".join(_skips) + ")"
            on_phase("preprocessing",
                     "意图：" + (preprocessed.question_intent[:60] or "未识别") +
                     " 领域：" + "、".join(preprocessed.question_domains[:3]) +
                     _nature_str + _skip_info,
                     data=preprocessed)

        # 整个认知循环使用扩展后的结构化问题作为有效问题
        effective_question = preprocessed.expanded_question or question
        log.info("cognitive_loop.preprocessed",
                 original=question[:80],
                 expanded=effective_question[:120],
                 intent=preprocessed.question_intent[:60],
                 domains=preprocessed.question_domains)

        # 简单任务直接切 fast 模式，只跑一轮
        if (getattr(preprocessed, "task_complexity", "standard") == "simple"
                and mode != "deep"):
            mode = "fast"
            mode_max_iter = 1

        log.info("cognitive_loop.phase_plan",
                 task_nature=getattr(preprocessed, "task_nature", ""),
                 task_complexity=getattr(preprocessed, "task_complexity", ""),
                 skips=_skips, lights=_lights, mode=mode)

        # ── 首轮对话自动生成标题 ──
        # 当 conv_context 为空且 conv_id 有效时，说明是新对话的第一轮，
        # 用 LLM 快速生成一个简洁的会话标题，并通过 on_title 推送到前端。
        log.info("cognitive_loop.title_check",
                 conv_id=conv_id,
                 has_conv_context=bool(conv_context),
                 has_on_title=on_title is not None)
        if conv_id and not conv_context and on_title:
            try:
                log.info("cognitive_loop.title_generating", conv_id=conv_id)
                title = await self._generate_title(question, preprocessed)
                log.info("cognitive_loop.title_result", conv_id=conv_id, title=title or "(empty)")
                if title:
                    self.episodic.set_conversation_name(conv_id, title)
                    on_title(title)
                    log.info("cognitive_loop.title_generated", conversation=conv_id, title=title)
            except Exception:
                log.warning("cognitive_loop.title_generation_failed", exc_info=True)

        # ── 智能体主动提问（即时）──
        # 预处理阶段若判定必须先问用户才能答好，此刻立即阻塞发问，不拖到调查/决策之后。
        # 仅流式/交互场景提供 on_clarification；一轮对话只问一次。
        _cq = getattr(preprocessed, "clarifying_questions", None) or []
        if _cq and on_clarification is not None and not working_mem.get("clarification_asked", False):
            working_mem.set("clarification_asked", True)
            log.info("cognitive_loop.clarification_ask", n=len(_cq), where="post_preprocessing")
            on_clarification(_cq)
            controller.begin_clarification(_cq)
            _ans = await controller.await_clarification(timeout=_CLARIFICATION_TIMEOUT)
            if controller.should_stop:
                if on_phase: on_phase("preprocessing", "已终止")
                # 交由下方迭代循环首个 _check("investigation") 捕获 stop 并跳出，走正常收尾
            elif _ans:
                working_mem.set("user_clarification", _ans)
                effective_question = effective_question + "\n\n[用户对智能体提问的补充回答]：" + _ans
                if on_phase: on_phase("preprocessing", "已收到你的补充，开始调查")
            else:
                if on_phase: on_phase("preprocessing", "未收到补充（超时/跳过），按现有信息继续")

        for iteration in range(1, mode_max_iter + 1):
            log.info("cognitive_loop.iteration", i=iteration)
            trace.metadata.iterations = iteration
            t0 = datetime.now()
            if trace.contradictions and not working_mem.get_contradiction_graph():
                working_mem.set_contradiction(trace.contradictions)
            skip_phases = working_mem.get("skip_phases") or []
            focus_hints = working_mem.get("focus_hints") or {}
            def _hint(phase):
                h = focus_hints.get(phase, "")
                return "\n[反思专项提示] " + h if h else ""
            def _steer(phase):
                s = controller.collect_steers(phase)
                if s and _wants_detailed_report(s):
                    working_mem.set("report_requested", True)
                return s
            async def _check(phase):
                sig = await controller.check_phase_boundary(phase)
                if sig == "stop":
                    if on_phase: on_phase(phase, "已终止")
                    return True
                if sig == "resumed":
                    if on_phase: on_phase(phase, "已恢复，继续执行")
                return False
            # 初始化阶段输出变量——被跳过的阶段下游仍需引用
            fact_report = None
            contradiction_graph = None
            rational_synthesis = None
            if await _check("investigation"): break
            if "investigation" in skip_phases:
                log.info("cognitive_loop.skip_phase", phase="investigation")
                if on_phase: on_phase(CognitivePhaseName.INVESTIGATION, "已跳过：上一轮结论充分")
            else:
                if on_phase: on_phase(CognitivePhaseName.INVESTIGATION, "正在调查研究")
                self.skill_manager.inject_phase_skills("investigation", working_mem)
                extra_ctx = working_mem.get_context_for_phase("investigation") + _hint("investigation") + _steer("investigation")
                hist = working_mem.get("conversation_history", "")
                if hist: extra_ctx = hist + "\n" + extra_ctx if extra_ctx else hist
                fact_report = await self.investigation.investigate(question=effective_question, additional_context=context + extra_ctx)
                trace.investigation = fact_report
                working_mem.set("last_investigation", fact_report.summary)
                _record_duration(trace, "investigation", t0)
                if on_phase:
                    n_facts = len(fact_report.facts)
                    n_gaps = len(fact_report.gaps)
                    on_phase(CognitivePhaseName.INVESTIGATION, "发现 " + str(n_facts) + " 条事实，" + str(n_gaps) + " 个信息缺口",
                             data=fact_report)
            if mode == "fast":
                if on_phase: on_phase(CognitivePhaseName.CONTRADICTION, "正在矛盾分析（快速模式）")
                contradiction_graph = await self.contradiction.analyze(fact_report, effective_question, additional_context=working_mem.get_context_for_phase("contradiction"))
                trace.contradictions = contradiction_graph
                if on_phase: on_phase(CognitivePhaseName.RATIONAL, "正在形成理性认识（快速模式）")
                rational = await self.rational.synthesize(effective_question, fact_report, contradiction_graph,
                    system_model=contradiction_graph.system_model)
                trace.rational_synthesis = rational
                if on_phase: on_phase(CognitivePhaseName.DECISION, "正在形成决策（快速模式）")
                decision = await self.decision.decide(effective_question, fact_report, contradiction_graph, rational,
                    system_model=contradiction_graph.system_model)
                trace.decision = decision
                if on_phase: on_phase("done", "快速模式完成")
                break
            if await _check("contradiction"): break
            if "contradiction" in skip_phases:
                log.info("cognitive_loop.skip_phase", phase="contradiction")
                if on_phase: on_phase(CognitivePhaseName.CONTRADICTION, "已跳过：上一轮矛盾结构稳固")
                contradiction_graph = trace.contradictions
            else:
                self.skill_manager.inject_phase_skills("contradiction", working_mem)
                if on_phase: on_phase(CognitivePhaseName.CONTRADICTION, "正在进行矛盾分析")
                t1 = datetime.now()
                contradiction_graph = await self.contradiction.analyze(fact_report=fact_report, question=effective_question + _hint("contradiction") + _steer("contradiction"), additional_context=working_mem.get_context_for_phase("contradiction"))
                trace.contradictions = contradiction_graph
                working_mem.set_contradiction(contradiction_graph)
                # 存储系统模型供下游阶段使用
                if contradiction_graph.system_model:
                    working_mem.set("_system_model", contradiction_graph.system_model)
                    log.info("cognitive_loop.system_model",
                             n_elements=len(contradiction_graph.system_model.elements),
                             n_relationships=len(contradiction_graph.system_model.relationships),
                             n_feedback_loops=len(contradiction_graph.system_model.feedback_loops),
                             n_emergent=len(contradiction_graph.system_model.emergent_properties))
                _record_duration(trace, "contradiction", t1)
                if on_phase:
                    pc = contradiction_graph.principal_contradiction
                    desc = ""
                    if pc:
                        desc = pc.description[:60]
                        if len(pc.description) > 60: desc = desc + "..."
                    on_phase(CognitivePhaseName.CONTRADICTION, "主要矛盾：" + (desc if desc else "未识别"),
                             data=contradiction_graph)
            if await _check("rational"): break
            if "rational" in skip_phases:
                if on_phase: on_phase(CognitivePhaseName.RATIONAL, "已跳过：理性认识复用上一轮")
                rational_synthesis = trace.rational_synthesis
            else:
                self.skill_manager.inject_phase_skills("rational", working_mem)
                if on_phase: on_phase(CognitivePhaseName.RATIONAL, "正在形成理性认识")
                t2 = datetime.now()
                rational_synthesis = await self.rational.synthesize(question=effective_question + _hint("rational") + _steer("rational"), fact_report=fact_report, contradiction_graph=contradiction_graph, system_model=contradiction_graph.system_model)
                trace.rational_synthesis = rational_synthesis
                _record_duration(trace, "rational", t2)
                if on_phase: on_phase(CognitivePhaseName.RATIONAL, "本质：" + rational_synthesis.essence[:80],
                                       data=rational_synthesis)
            if await _check("decision"): break
            self.skill_manager.inject_phase_skills("decision", working_mem)
            if on_phase: on_phase(CognitivePhaseName.DECISION, "正在形成决策")
            t3 = datetime.now()
            dec_question = effective_question + _steer("decision")
            hist = working_mem.get("conversation_history", "")
            if hist: dec_question = dec_question + "\n\n" + hist
            decision_report = await self.decision.decide(question=dec_question, fact_report=fact_report, contradiction_graph=contradiction_graph, rational_synthesis=rational_synthesis, system_model=getattr(contradiction_graph, 'system_model', None))
            trace.decision = decision_report
            _record_duration(trace, "decision", t3)
            if on_phase: on_phase(CognitivePhaseName.DECISION, decision_report.summary,
                                   data=decision_report)
            if effective_review == "iterative" or (effective_review == "once" and iteration == 1):
                if await _check("perspectives"): break
                log.info("cognitive_loop.perspectives_running",
                         effective_review=effective_review, iteration=iteration)
                if on_phase: on_phase("perspectives", "正在进行多视角审查")
                tp = datetime.now()
                rounds = 2 if effective_review == "iterative" else 1
                perspective_synthesis = await self.perspectives.review(question=effective_question, contradiction_graph=contradiction_graph, decision_report=decision_report, rounds=rounds)
                trace.perspectives = perspective_synthesis
                _record_duration(trace, "perspectives", tp)
                if on_phase and perspective_synthesis.critical_warnings:
                    nw = len(perspective_synthesis.critical_warnings)
                    on_phase("perspectives", "关键警告 " + str(nw) + " 条",
                             data=perspective_synthesis)
            else:
                log.info("cognitive_loop.perspectives_skipped",
                         effective_review=effective_review, iteration=iteration)
            if await _check("practice"): break
            if "practice" in skip_phases:
                if on_phase: on_phase("practice", "已跳过：实践结论复用上一轮")
            else:
                if on_phase:
                    on_phase("practice", "正在执行实践检验")
                self.skill_manager.inject_phase_skills("practice", working_mem)
                t_prac = datetime.now()
                practice_report = await self.practice.practice(
                    question=effective_question + _hint("practice") + _steer("practice"),
                    decision_report=decision_report, trace=trace,
                    wm=working_mem,
                )
                if practice_report is not None:
                    trace.practice = practice_report
                    _record_duration(trace, "practice", t_prac)
                    # ── 统一处理（不再区分 exec/boundary）──
                    if on_phase:
                        n_findings = len(practice_report.unexpected_findings) + len(practice_report.unexpected_insights)
                        n_real_tasks = len(practice_report.real_world_practice_needed)
                        parts = []
                        if practice_report.mode == "executed":
                            parts.append("实践检验完成")
                        elif practice_report.mode == "partial":
                            parts.append("实践检验完成（部分行动项为知性分析，可信度上限 V2）")
                        else:
                            parts.append("知性分析完成（无直接实践，可信度上限 V2）")
                        if n_findings:
                            parts.append(str(n_findings) + " 个意外发现")
                        if n_real_tasks:
                            parts.append("生成 " + str(n_real_tasks) + " 项现实世界验证任务")
                        on_phase("practice", "，".join(parts), data=practice_report)
                    # 统一的 surprise/contradiction 处理
                    if practice_report.unexpected_findings:
                        working_mem.set("practice_surprises", "\n".join("- " + f for f in practice_report.unexpected_findings))
                    if practice_report.unexpected_insights:
                        existing = working_mem.get("practice_surprises", "")
                        new_insights = "\n".join("- " + s for s in practice_report.unexpected_insights)
                        working_mem.set("practice_surprises", existing + "\n" + new_insights if existing else new_insights)
                    if trace.contradictions and practice_report.unexpected_findings:
                        conflicts = []
                        pc = trace.contradictions.principal_contradiction
                        for f in practice_report.unexpected_findings:
                            if pc and any(pole in f for pole in pc.tension_poles):
                                conflicts.append(f)
                        if conflicts:
                            working_mem.set("_practice_contradiction_conflict", "实践发现以下与矛盾分析冲突：" + "; ".join(conflicts))
                            log.warning("cognitive_loop.practice_contradiction_conflict", n_conflicts=len(conflicts))
                    challenged_claims = [
                        c.get("claim", "") for c in practice_report.claim_assessments
                        if c.get("assessment") == "challenged"
                    ]
                    if challenged_claims:
                        working_mem.set("_boundary_challenged_claims",
                            "知性分析发现以下主张受到证据挑战：" + "; ".join(challenged_claims))
            if await _check("reflection"): break
            self.skill_manager.inject_phase_skills("reflection", working_mem)
            if on_phase: on_phase(CognitivePhaseName.REFLECTION, "正在对全流程进行反思")
            t4 = datetime.now()
            # ── 收集终止判定证据 ──
            termination_evidence = await self._gather_termination_evidence(trace, working_mem)
            working_mem.set("_termination_evidence", termination_evidence)
            reflection_report = await self.reflection.reflect(question=effective_question, trace=trace, termination_evidence=termination_evidence)
            trace.reflection = reflection_report
            _record_duration(trace, "reflection", t4)
            if on_phase:
                summary = getattr(reflection_report, 'quality_assessment', '') or "反思完成"
                conv = getattr(reflection_report, 'convergence_score', 0)
                on_phase(CognitivePhaseName.REFLECTION,
                         "收敛度 " + str(round(conv, 2)) + " . " + summary[:60],
                         data=reflection_report)

            # ── 技能蒸馏：从反思结果中提取可复用操作模式 ──
            if (hasattr(reflection_report, "skill_draft_candidates")
                    and reflection_report.skill_draft_candidates
                    and getattr(settings, "skill_auto_distill", True)):
                for raw in reflection_report.skill_draft_candidates:
                    try:
                        from ..api.schemas.models import SkillDraftCandidate
                        candidate = SkillDraftCandidate(**raw) if isinstance(raw, dict) else raw
                        candidate.extracted_from_session = session_id
                        self.skill_manager.create_draft(candidate)
                    except Exception as e:
                        log.warning("cognitive_loop.skill_draft_failed", error=str(e))

            contradiction_stable = getattr(reflection_report, 'contradiction_stability', 0.5) >= 0.8
            understanding_sufficient = getattr(reflection_report, 'understanding_level', '') == "理性"
            had_leap = getattr(reflection_report, 'qualitative_leap', True)
            # V2 场景（边界/纯知性分析）：实践阶段无法产生可执行代码，
            # 收敛度是 LLM 对自身产出的自我评估，数值门槛无独立意义——
            # 反思的定性判断（should_reinvestigate）说了算。
            if trace.practice and trace.practice.mode in ("epistemic_only", "partial"):
                should_stop = (iteration >= mode_max_iter
                               or not reflection_report.should_reinvestigate)
            else:
                # V3 场景：有可执行实践检验，收敛度门槛有意义
                should_stop = (iteration >= mode_max_iter
                               or (not reflection_report.should_reinvestigate
                                   and reflection_report.convergence_score >= self.convergence_threshold
                                   and contradiction_stable
                                   and (understanding_sufficient or not had_leap)))
            # 停滞检测：连续两轮无认识质变且反思不要求继续调查时，降级为快速模式
            # 核心原则：尊重反思阶段的判断——反思拥有最完整的认知图景。
            # 如果反思明确要求继续调查，停滞检测不触发降级。
            if not should_stop and iteration >= 2 and not reflection_report.should_reinvestigate:
                prev_leap = working_mem.get("_prev_qualitative_leap")
                if prev_leap is False and not had_leap and not understanding_sufficient:
                    if mode == "standard":
                        log.warning("cognitive_loop.stagnation", iteration=iteration,
                                    message="连续两轮无认识质变且反思不要求继续，降级为快速模式")
                        mode = "fast"
                        # 更新 mode_max_iter：快速模式最多再跑一轮
                        mode_max_iter = iteration + 1
                    else:
                        log.info("cognitive_loop.stagnation_ignored", iteration=iteration,
                                 mode=mode, message="深度模式不停滞降级，继续迭代")
            working_mem.set("_prev_qualitative_leap", had_leap)
            try:
                from .credibility_chain import CredibilityChain
                chain = CredibilityChain.from_trace(fact_report=fact_report, contradiction_graph=contradiction_graph, rational_synthesis=rational_synthesis, decision_report=decision_report)
                trace.metadata.credibility_chain_summary = chain.summary()
                working_mem.set("_credibility_chain", chain.summary())
                log.info("cognitive_loop.credibility_chain", chain=chain.summary())
            except Exception: pass
            log.info("cognitive_loop.iteration_done", convergence=reflection_report.convergence_score, reinvestigate=reflection_report.should_reinvestigate, contradiction_stability=getattr(reflection_report, 'contradiction_stability', 0.5), understanding=getattr(reflection_report, 'understanding_level', ''), skip_phases=reflection_report.skip_phases, recommended_mode=reflection_report.recommended_mode, stop=should_stop)
            if should_stop: break
            # 如果反思要求重新调查，实践结论必然不稳固——
            # 新事实新矛盾意味着新一轮决策行动项需要新的实践检验
            next_skip = list(reflection_report.skip_phases)
            if reflection_report.should_reinvestigate and "practice" in next_skip:
                log.info("cognitive_loop.force_practice",
                         reason="should_reinvestigate=true，移除 skip_phases 中的 practice")
                next_skip.remove("practice")
            working_mem.set("skip_phases", next_skip)
            if reflection_report.focus_hints: working_mem.set("focus_hints", reflection_report.focus_hints)
            if reflection_report.recommended_mode and reflection_report.recommended_mode != mode:
                log.info("cognitive_loop.mode_switch", old=mode, new=reflection_report.recommended_mode)
                mode = reflection_report.recommended_mode
            focus = reflection_report.reinvestigation_focus
            if focus:
                working_mem.set("reinvestigation_focus", focus)
                effective_question = effective_question + "\n\n[上一轮反思提示：" + focus + "]"
        if conv_id:
            release_conv_controller(conv_id)
        else:
            release_controller(session_id)
        trace.metadata.end_time = datetime.now()
        # 使用原始问题（未经迭代追加污染的纯净版本）构建响应
        original_question = working_mem.get("original_question", question)
        _detailed = bool(working_mem.get("report_requested", False)
                         or (trace.reflection and getattr(trace.reflection, "recommend_detailed_report", False)))
        response = await self._build_response(original_question, trace, session_id, conv_id, detailed=_detailed)
        self._save_episode(response, conv_id, _project_id)
        log.info("cognitive_loop.done", session_id=session_id, conversation=conv_id, iterations=trace.metadata.iterations, summary=response.summary[:80])
        return response

    async def stream_run(self, question, context="", mode="standard", conversation_id="", review_strategy="", model_override="", files=None, project_id=""):
        # model_override 为空或"智能路由"时，各阶段用各自配置的模型（get_phase_llm）
        # 指定具体模型名时，全流程统一使用该模型
        effective_override = model_override if (model_override and model_override != "智能路由") else ""
        if effective_override:
            from ..llm import get_llm as _get_llm_override
            override_loop = CognitiveLoop(
                llm=_get_llm_override(model=effective_override),
                web_search_enabled=self.investigation.can_search_web if hasattr(self, 'investigation') else None,
                conversation_id=conversation_id,
                review_strategy=review_strategy,
                project_id=project_id,
            )
            async for event in override_loop.stream_run(
                question, context, mode,
                conversation_id=conversation_id,
                review_strategy=review_strategy,
                files=files, project_id=project_id,
            ):
                yield event
            return

        queue = asyncio.Queue()
        async def _work():
            def _push(phase, summary, data=None):
                evt = {"type": "phase", "phase": phase, "summary": summary}
                if data is not None:
                    evt["data"] = data
                try:
                    queue.put_nowait(evt)
                except Exception:
                    pass
            def _push_clarification(questions):
                try:
                    queue.put_nowait({"type": "clarification", "questions": list(questions or [])})
                except Exception:
                    pass
            def _push_title(title):
                try:
                    queue.put_nowait({"type": "title", "title": title,
                                       "conversation_id": conversation_id})
                    log.info("cognitive_loop.title_queued", title=title, conv_id=conversation_id)
                except Exception:
                    log.warning("cognitive_loop.title_queue_failed", exc_info=True)
            try:
                result = await self.run(question, context, mode, on_phase=_push,
                    conversation_id=conversation_id, review_strategy=review_strategy,
                    on_title=_push_title, files=files, project_id=project_id,
                    on_clarification=_push_clarification)
                queue.put_nowait({"type": "result", "data": result})
            except Exception as exc:
                log.error("cognitive_loop.stream_error", error=str(exc), exc_info=True)
                # 即使 run() 崩溃也保存最小 episode，保证后续消息的对话上下文可用
                try:
                    import uuid as _uuid
                    sid = _uuid.uuid4().hex[:8]
                    self.episodic.save_episode(
                        session_id=sid, question=question,
                        summary=f"[处理异常] {str(exc)[:200]}",
                        conversation_id=conversation_id,
                        project_id=project_id,
                    )
                except Exception:
                    pass
                try:
                    queue.put_nowait({"type": "result", "data": None, "error": str(exc)})
                except Exception:
                    pass
            except BaseException as be:
                # CancelledError (SSE disconnect, user stop) — finalize the stuck [...] episode
                if isinstance(be, KeyboardInterrupt):
                    raise
                try:
                    import uuid as _uuid
                    sid = _uuid.uuid4().hex[:8]
                    self.episodic.save_episode(
                        session_id=sid, question=question,
                        summary="已终止",
                        conversation_id=conversation_id,
                        project_id=project_id,
                    )
                except Exception:
                    pass
                try:
                    queue.put_nowait({"type": "result", "data": None, "error": "已终止"})
                except Exception:
                    pass
        task = asyncio.create_task(_work())
        while True:
            try:
                event = queue.get_nowait()
                yield event
                if event["type"] == "result": break
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)
        await task


    async def _build_response(self, question, trace, session_id, conversation_id="", detailed=False):
        """构建最终用户回复。

        核心改动：不再使用反思阶段的 final_answer（常为过程总结），
        而是将全部累积分析材料发给 LLM，要求其直接回答用户的问题。
        """
        decision = trace.decision
        if decision is None:
            return AgentResponse(
                summary="认知循环未完成，请重试",
                session_id=session_id, question=question,
                full_trace=trace, conversation_id=conversation_id,
            )

        # ── 构建分析上下文 ──
        analysis_context = self._summarize_for_answer(question, trace, detailed=detailed)

        # ── 调用 LLM 生成直接回答 ──
        summary = ""
        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": analysis_context}],
                system=_DETAILED_REPORT_PROMPT if detailed else _FINAL_ANSWER_PROMPT,
                temperature=0.5,
                max_tokens=(settings.detailed_report_max_tokens if detailed else settings.final_answer_max_tokens),
            )
            summary = response.content.strip()
            if len(summary) < 20:
                summary = ""
            else:
                log.info("build_response.llm_answer", len=len(summary))
        except Exception as e:
            log.warning("build_response.llm_answer_failed", error=str(e))

        # ── 回退链：reflection.final_answer → practice → decision ──
        if not summary:
            if trace.reflection and trace.reflection.final_answer:
                summary = trace.reflection.final_answer.strip()
                log.info("build_response.fallback_final_answer", len=len(summary))
            if not summary and trace.practice and trace.practice.analysis_summary:
                summary = trace.practice.analysis_summary.strip()
                log.info("build_response.fallback_analysis_summary", len=len(summary))
            if not summary:
                summary = decision.summary or decision.strategic_assessment[:200]
                if trace.practice and trace.practice.practice_summary:
                    ps = trace.practice.practice_summary
                    if ps and len(ps) > 10:
                        summary = ps[:300] + "\n" + summary
                log.info("build_response.fallback_decision_summary", len=len(summary))

        # ── 行动项（action_items）：来自决策输出 ──
        action_texts = []
        if decision.action_items:
            action_texts = [a.description for a in decision.action_items[:6]]

        if not action_texts and trace.practice and trace.practice.real_world_practice_needed:
            if trace.practice.real_world_practice_needed:
                action_texts = [
                    "[建议实践] " + (t.practice_method[:120] if hasattr(t, 'practice_method') else str(t)[:120])
                    for t in trace.practice.real_world_practice_needed[:5]
                ]

        # ── 生成文件 ──
        from ..api.schemas.models import GeneratedFile
        generated_files = []
        seen_paths = set()
        if trace.practice and trace.practice.steps_taken:
            for step in trace.practice.steps_taken:
                desc = step.description or ""
                if desc.startswith("创建："):
                    path = desc[3:].strip()
                    for sep in [" ->", "（", "("]:
                        idx2 = path.find(sep)
                        if idx2 > 0:
                            path = path[:idx2].strip()
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        ws = self.workspace.workspace
                        fpath = ws / path
                        size = fpath.stat().st_size if fpath.exists() else 0
                        generated_files.append(GeneratedFile(
                            path=path,
                            description=step.action_taken or "",
                            size_bytes=size,
                        ))

        return AgentResponse(
            summary=summary, action_items=action_texts,
            full_trace=trace, session_id=session_id,
            question=question, conversation_id=conversation_id,
            generated_files=generated_files,
        )

    def _summarize_for_answer(self, question: str, trace, detailed: bool = False) -> str:
        """将全部认知轨迹压缩为结构化上下文，供回答阶段 LLM 使用。
        detailed=True 时放宽各处截断、保留更多信息，供详实报告使用。"""
        n_facts, cap_sum, n_gaps = (30, 800, 10) if detailed else (15, 300, 5)
        n_sec, cap_sec = (8, 300) if detailed else (5, 150)
        n_list, cap_item, cap_ess = (10, 200, 600) if detailed else (5, 100, 300)
        parts = [f"# 用户原始问题\n{question}\n"]

        if trace.investigation:
            inv = trace.investigation
            parts.append("## 调查发现的事实")
            for i, f in enumerate(inv.facts[:n_facts]):
                src = f.source_type or "未知来源"
                cred = f"{f.credibility:.0%}" if f.credibility else "?"
                parts.append(f"{i+1}. [{src}] (可信度 {cred}) {f.content}")
            if inv.gaps:
                parts.append(f"\n信息缺口 ({len(inv.gaps)}个)：{'；'.join(g.description[:80] for g in inv.gaps[:n_gaps])}")
            if inv.summary:
                parts.append(f"\n调查摘要：{inv.summary[:cap_sum]}")

        if trace.contradictions:
            cg = trace.contradictions
            if cg.principal_contradiction:
                pc = cg.principal_contradiction
                parts.append(f"\n## 主要矛盾\n{pc.description}")
                if pc.tension_poles:
                    parts.append(f"对立面：{' vs '.join(pc.tension_poles)}")
            if detailed and cg.system_model and cg.system_model.elements:
                parts.append("系统要素：" + "、".join(e.name for e in cg.system_model.elements))
            if cg.secondary_contradictions:
                parts.append(f"\n次要矛盾：")
                for c in cg.secondary_contradictions[:n_sec]:
                    parts.append(f"- {c.description[:cap_sec]}")

        if trace.rational_synthesis:
            r = trace.rational_synthesis
            parts.append(f"\n## 理性认识")
            parts.append(f"本质：{r.essence[:cap_ess]}")
            if r.patterns:
                parts.append(f"规律：{'；'.join(r.patterns[:n_list])}")
            if r.hypotheses:
                parts.append(f"假设：{'；'.join(r.hypotheses[:n_list])}")

        if trace.decision:
            d = trace.decision
            parts.append(f"\n## 决策方向")
            parts.append(d.summary or d.strategic_assessment[:cap_ess])
            if d.action_items:
                parts.append(f"行动项：{'；'.join(a.description[:cap_item] for a in d.action_items[:n_list])}")
            if d.risks:
                parts.append(f"风险：{'；'.join(d.risks[:n_list])}")

        if trace.practice:
            p = trace.practice
            mode_label = {"executed": "直接实践", "partial": "部分实践", "epistemic_only": "知性分析"}
            mode_display = mode_label.get(p.mode, p.mode)
            parts.append(f"\n## 实践检验（模式：{mode_display}，可信度上限：{p.confidence_ceiling or 'V3'}）")
            if p.practice_summary:
                parts.append(p.practice_summary[:cap_ess])
            if p.unexpected_findings:
                parts.append(f"意外发现：{'；'.join(p.unexpected_findings[:n_list])}")
            if p.analysis_summary:
                parts.append(f"知性分析：{p.analysis_summary[:cap_ess]}")

        if trace.perspectives and trace.perspectives.synthesized_insight:
            parts.append(f"\n## 多视角综合\n{trace.perspectives.synthesized_insight[:cap_ess]}")

        parts.append("\n---\n请根据以上材料，直接回答用户的原始问题。")
        return "\n".join(parts)

    def _save_episode(self, response, conversation_id="", project_id=""):
        try:
            trace = response.full_trace
            pc = ""
            if trace and trace.contradictions and trace.contradictions.principal_contradiction:
                pc = trace.contradictions.principal_contradiction.description[:200]
            lessons = []
            if trace and trace.reflection and trace.reflection.lessons_learned:
                lessons = trace.reflection.lessons_learned

            # 改进四：提取所有矛盾的推导链
            derivation_chains = []
            if trace and trace.contradictions:
                for c in trace.contradictions.all_contradictions:
                    if c.derivation_chain:
                        derivation_chains.append({
                            "chain_id": c.derivation_chain.chain_id,
                            "contradiction": c.description[:200],
                            "summary": c.derivation_chain.summary,
                            "steps": [
                                {
                                    "step_id": s.step_id,
                                    "fact_basis": s.fact_basis,
                                    "inference": s.inference,
                                    "conclusion": s.conclusion,
                                    "confidence": s.confidence,
                                    "reversible": s.reversible,
                                }
                                for s in c.derivation_chain.steps
                            ],
                            "factual_foundation": c.derivation_chain.factual_foundation,
                        })
                # 也保存矛盾分析阶段的整体推导链
                if trace.contradictions.contradiction_derivation:
                    cg_dc = trace.contradictions.contradiction_derivation
                    derivation_chains.append({
                        "chain_id": cg_dc.chain_id,
                        "contradiction": "[矛盾分析整体推导]",
                        "summary": cg_dc.summary,
                        "steps": [
                            {
                                "step_id": s.step_id,
                                "fact_basis": s.fact_basis,
                                "inference": s.inference,
                                "conclusion": s.conclusion,
                                "confidence": s.confidence,
                                "reversible": s.reversible,
                            }
                            for s in cg_dc.steps
                        ],
                        "factual_foundation": cg_dc.factual_foundation,
                    })

            self.episodic.save_episode(
                session_id=response.session_id,
                question=response.question,
                summary=response.summary[:500],
                action_items=response.action_items,
                principal_contradiction=pc,
                lessons=lessons,
                conversation_id=conversation_id or response.conversation_id,
                derivation_chains=derivation_chains if derivation_chains else None,
                project_id=project_id,
            )
        except Exception:
            log.warning("episodic_memory.save_failed", exc_info=True)

    async def _generate_title(self, question: str, preprocessed) -> str:
        """Generate a concise session title using structured JSON output."""
        intent = (preprocessed.question_intent or "").split("\u2014\u2014")[0].strip()[:40]
        domains = preprocessed.question_domains[:2] if preprocessed.question_domains else []
        prompt = (
            "Generate a conversation title.\n\n"
            + "Topic: " + intent + "\n"
            + ("Domains: " + ", ".join(domains) + "\n" if domains else "")
            + "Question: " + question[:80] + "\n\n"
            + "Return JSON: {\"title\": \"<6-12 Chinese characters, concise, no punctuation>\"}"
        )
        try:
            resp = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                system="You name conversations. Output ONLY the JSON object, nothing else. No preamble.",
                temperature=0.2, max_tokens=128,
                response_format={"type": "json_object"},
            )
            import json, re
            raw = resp.content.strip()
            # If the model echoed preamble text, extract just the JSON part
            json_match = re.search(r'\{[^{}]*"title"\s*:\s*"[^"]*"[^{}]*\}', raw)
            if json_match:
                raw = json_match.group(0)
            data = json.loads(raw)
            title = str(data.get("title", "")).strip()
            # Cleanup: strip stray quotes, brackets, punctuation
            title = title.strip("'\"\u201c\u201d\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\"'`\u3001\u3002\uff0c\u3002").strip()
            title = re.sub(r'^[\s\d\.\-\*\#\:\uff1a]+', '', title)
            if len(title) > 12:
                title = title[:12]
            if title:
                return title
            return question[:12]
        except Exception:
            log.warning("cognitive_loop.title_generation_failed", exc_info=True)
            return question[:12]

    def _save_initial_episode(self, session_id: str, question: str, conversation_id: str, project_id: str = ""):
        """Save a minimal episode immediately during preprocessing,
        so the session is recoverable even if the page is closed."""
        try:
            self.episodic.save_episode(
                session_id=session_id,
                question=question,
                summary="[...]",
                conversation_id=conversation_id,
                project_id=project_id,
            )
        except Exception:
            pass


def _record_duration(trace, phase, t0):
    elapsed = (datetime.now() - t0).total_seconds()
    trace.metadata.phase_durations[phase] = elapsed


def _load_persona_context() -> str:
    """Load agent persona and custom knowledge from ui-settings.json."""
    import json
    from ..config import settings
    path = settings.data_dir / "ui-settings.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        persona = data.get("agent_persona", "").strip()
        if persona:
            parts.append("## Agent role (user config)\n" + persona)
        knowledge = data.get("custom_knowledge", "").strip()
        if knowledge:
            parts.append("## User knowledge / experience\n" + knowledge)
        return "\n\n".join(parts)
    except Exception:
        return ""
