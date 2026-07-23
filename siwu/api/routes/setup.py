"""
思悟 Agent —— 设置 / 引导路由
POST /api/v1/setup        保存 LLM 配置到 config.toml + .env
GET  /api/v1/setup/status 检查是否已完成配置
GET  /api/v1/settings     读取 UI 设置（字体、角色、知识）
PUT  /api/v1/settings     保存 UI 设置
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...config import settings, CONFIG_TOML, save_config_section, save_config_key, _toml_value
from .agent import init_agent_resources

router = APIRouter(prefix="/api/v1", tags=["setup", "settings"])
log = logging.getLogger(__name__)

PROVIDER_PRESETS = {
    "deepseek": {"label": "DeepSeek", "url": "https://api.deepseek.com", "model": "deepseek-v4-pro"},
    "openai": {"label": "OpenAI", "url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "anthropic": {"label": "Anthropic Claude", "url": "", "model": "claude-sonnet-4-5"},
    "ollama": {"label": "Ollama 本地", "url": "http://localhost:11434/v1", "model": "llama3"},
    "custom": {"label": "自定义 / 中转站", "url": "", "model": ""},
}

PROVIDER_MODELS = {
    "deepseek": ["deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3-mini", "o4-mini"],
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5", "claude-sonnet-4"],
    "ollama": ["llama3", "llama3.1", "mistral", "qwen2.5"],
    "custom": [],
}

# ── UI Settings persistence ───────────────────────────────────

def _ui_settings_path() -> Path:
    p = settings.data_dir / "ui-settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_ui_settings() -> dict:
    path = _ui_settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _save_ui_settings(data: dict) -> None:
    path = _ui_settings_path()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_runtime_overrides() -> dict:
    """Read runtime parameter overrides from ui-settings.
    Callers should merge with their own defaults.
    Returns a flat dict of known runtime keys (no nesting)."""
    data = _load_ui_settings()
    known_keys = {
        "max_iterations", "practice_rounds",
        "web_search_max_results", "web_fetch_max_urls",
        "web_fetch_max_chars_per_page", "web_fetch_max_total_chars",
    }
    return {k: data[k] for k in known_keys if k in data}


class UiSettings(BaseModel):
    font_family: str = "system"       # system | serif | mono
    font_size: str = "medium"         # small | medium | large
    page_zoom: int = 100              # 页面缩放百分比
    show_thinking_after_answer: bool = True  # 回答后是否显示思维过程
    agent_persona: str = ""           # Agent 角色设定
    custom_knowledge: str = ""        # 用户自定义知识 / 重要经验
    phase_models: dict[str, str] = {}  # 智能路由：各阶段模型选择
    # ── 运行时高级参数 ──
    max_iterations: int = 0           # 0 = 使用 config.toml 默认值
    practice_rounds: int = 0          # 实践阶段实验轮数
    web_search_max_results: int = 0   # 搜索最大结果数
    web_fetch_max_urls: int = 0       # 网页抓取最大URL数
    web_fetch_max_chars_per_page: int = 0  # 单页最大字符数
    web_fetch_max_total_chars: int = 0     # 总抓取最大字符数


def _get_default_ui_settings() -> dict:
    return {
        "font_family": "system",
        "font_size": "medium",
        "page_zoom": 100,
        "show_thinking_after_answer": True,
        "agent_persona": "",
        "custom_knowledge": "",
        "phase_models": {},
        # Runtime params — 0 means "use config.toml default"
        "max_iterations": 0,
        "practice_rounds": 0,
        "web_search_max_results": 0,
        "web_fetch_max_urls": 0,
        "web_fetch_max_chars_per_page": 0,
        "web_fetch_max_total_chars": 0,
    }


@router.get("/settings")
async def get_settings():
    """读取 UI 设置（config.toml [ui] 优先，回退到 ui-settings.json）。"""
    # Read from config.toml's [ui] section first
    data = {
        "font_family": settings.ui_font_family,
        "font_size": settings.ui_font_size,
        "page_zoom": settings.ui_page_zoom,
        "show_thinking_after_answer": settings.ui_show_thinking,
        "agent_persona": settings.ui_agent_persona,
        "custom_knowledge": settings.ui_custom_knowledge,
        "phase_models": json.loads(settings.ui_phase_models) if settings.ui_phase_models else {},
    }
    # Fallback to ui-settings.json for any missing non-empty values, and for runtime params
    legacy = _load_ui_settings()
    for k in ["font_family", "font_size", "page_zoom", "show_thinking_after_answer",
               "agent_persona", "custom_knowledge", "phase_models"]:
        if not data.get(k) and legacy.get(k):
            data[k] = legacy[k]
    # Runtime params still come from ui-settings.json (not in config.toml)
    for k in ["max_iterations", "practice_rounds", "web_search_max_results",
               "web_fetch_max_urls", "web_fetch_max_chars_per_page", "web_fetch_max_total_chars"]:
        data[k] = legacy.get(k, _get_default_ui_settings().get(k, 0))
    return data


@router.put("/settings")
async def save_settings(req: UiSettings):
    """保存 UI 设置到 config.toml 和 ui-settings.json。"""
    # Write [ui] section to config.toml
    ui_lines = [
        f"font_family = {_toml_value(req.font_family)}",
        f"font_size = {_toml_value(req.font_size)}",
        f"page_zoom = {req.page_zoom}",
        f"show_thinking = {'true' if req.show_thinking_after_answer else 'false'}",
    ]
    if req.agent_persona:
        ui_lines.append(f"agent_persona = {_toml_value(req.agent_persona)}")
    if req.custom_knowledge:
        ui_lines.append(f"custom_knowledge = {_toml_value(req.custom_knowledge)}")
    if req.phase_models:
        ui_lines.append(f"phase_models = {_toml_value(json.dumps(req.phase_models, ensure_ascii=False))}")

    save_config_section("ui", ui_lines)
    # Also write runtime params to [runtime] section in config.toml
    if req.max_iterations > 0:
        save_config_key("runtime", "max_iterations", req.max_iterations)
    if req.practice_rounds > 0:
        save_config_key("runtime", "practice_rounds", req.practice_rounds)

    # Still write ui-settings.json for legacy compatibility (runtime params)
    data = req.model_dump()
    _save_ui_settings(data)

    # Update in-memory settings
    settings.ui_font_family = req.font_family
    settings.ui_font_size = req.font_size
    settings.ui_page_zoom = req.page_zoom
    settings.ui_show_thinking = req.show_thinking_after_answer
    settings.ui_agent_persona = req.agent_persona or ""
    settings.ui_custom_knowledge = req.custom_knowledge or ""
    settings.ui_phase_models = json.dumps(req.phase_models, ensure_ascii=False) if req.phase_models else ""
    if req.max_iterations > 0:
        settings.max_iterations = req.max_iterations

    log.info("ui_settings_saved")
    return {"ok": True}


# ── Dev mode toggle ──

class DevModeRequest(BaseModel):
    enabled: bool = False


@router.put("/settings/dev")
async def set_dev_mode(req: DevModeRequest):
    """Enable or disable developer mode."""
    save_config_key("developer", "enabled", req.enabled)
    settings.dev_enabled = req.enabled
    log.info("dev_mode_toggled | enabled=%s", req.enabled)
    return {"ok": True, "dev_enabled": req.enabled}


# ── Models endpoint ──────────────────────────────────────────

@router.get("/models")
async def list_models(provider_key: str = ""):
    """返回当前 provider 可用的模型列表。可传入 provider_key 查询特定服务商的模型。"""
    pk = provider_key.strip().lower() if provider_key else ""
    if not pk:
        # Auto-detect from current settings
        pk = "deepseek"
        if settings.llm_provider == "anthropic":
            pk = "anthropic"
        elif settings.llm_base_url:
            url = settings.llm_base_url.lower()
            if "deepseek" in url:
                pk = "deepseek"
            elif "openai" in url:
                pk = "openai"
            elif "ollama" in url or "localhost" in url:
                pk = "ollama"
            else:
                pk = "custom"
    models = PROVIDER_MODELS.get(pk, [])
    current = settings.default_model
    return {"models": ["智能路由"] + models, "current": current, "provider_key": pk}


# ── Setup routes ─────────────────────────────────────────────

class SetupRequest(BaseModel):
    provider: str = "deepseek"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class TestConnectionRequest(BaseModel):
    provider: str = "deepseek"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class TestConnectionResponse(BaseModel):
    ok: bool
    latency_ms: float = 0.0
    model: str = ""
    error: str = ""


class SetupStatusResponse(BaseModel):
    configured: bool
    provider: str = ""
    model: str = ""
    base_url: str = ""
    provider_key: str = ""  # deepseek / openai / anthropic / ollama / custom
    dev_enabled: bool = False


class LlmConfigResponse(BaseModel):
    provider_key: str = ""     # deepseek / openai / anthropic / ollama / custom
    base_url: str = ""
    model: str = ""
    api_key: str = ""          # current saved key (local-only, safe for localhost)
    configured: bool = False


@router.get("/setup/status", response_model=SetupStatusResponse)
async def setup_status():
    """检查是否已配置 API Key。"""
    has_key = bool(settings.llm_api_key or settings.anthropic_api_key or settings.deepseek_api_key)
    # Determine provider_key from current settings
    provider_key = "deepseek"
    if settings.llm_provider == "anthropic":
        provider_key = "anthropic"
    elif settings.llm_base_url:
        url = settings.llm_base_url.lower()
        if "deepseek" in url:
            provider_key = "deepseek"
        elif "openai" in url:
            provider_key = "openai"
        elif "ollama" in url or "localhost" in url:
            provider_key = "ollama"
        else:
            provider_key = "custom"
    return SetupStatusResponse(
        configured=has_key,
        provider=settings.llm_provider,
        model=settings.default_model,
        base_url=settings.llm_base_url,
        provider_key=provider_key,
        dev_enabled=settings.dev_enabled,
    )


@router.get("/setup/config", response_model=LlmConfigResponse)
async def get_llm_config():
    """获取当前 LLM 配置（用于设置界面的账户 tab）。"""
    provider_key = "deepseek"
    if settings.llm_provider == "anthropic":
        provider_key = "anthropic"
    elif settings.llm_base_url:
        url = settings.llm_base_url.lower()
        if "deepseek" in url:
            provider_key = "deepseek"
        elif "openai" in url:
            provider_key = "openai"
        elif "ollama" in url or "localhost" in url:
            provider_key = "ollama"
        else:
            provider_key = "custom"
    has_key = bool(settings.llm_api_key or settings.anthropic_api_key or settings.deepseek_api_key)
    return LlmConfigResponse(
        provider_key=provider_key,
        base_url=settings.llm_base_url,
        model=settings.default_model,
        api_key=settings.llm_api_key or settings.anthropic_api_key or settings.deepseek_api_key,
        configured=has_key,
    )


@router.post("/setup")
async def save_setup(req: SetupRequest):
    """保存 LLM 配置到 config.toml 和 .env，并重新加载。"""
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API Key 不能为空")

    is_anthropic = req.provider.lower() == "anthropic"

    # Patch the [llm] section in config.toml (preserves all other sections)
    if is_anthropic:
        save_config_section("llm", [
            'provider = "anthropic"',
            f"anthropic_api_key = {_toml_value(req.api_key)}",
            f"model = {_toml_value(req.model)}",
        ])
    else:
        save_config_section("llm", [
            'provider = "openai_compatible"',
            f"base_url = {_toml_value(req.base_url)}",
            f"api_key = {_toml_value(req.api_key)}",
            f"model = {_toml_value(req.model)}",
        ])
    log.info("config_patched | section=llm")

    # ── Also write to .env for dotenv compatibility ──
    # Variable names must match what siwu/config.py _build() reads
    try:
        env_path = Path.cwd() / ".env"
        env_lines = [
            f"# 思悟 Agent —— API 配置（由 Web UI 自动生成）",
            f"SIWU_LLM_PROVIDER={'anthropic' if is_anthropic else 'openai_compatible'}",
        ]
        if is_anthropic:
            env_lines.append(f"ANTHROPIC_API_KEY={req.api_key.strip()}")
        else:
            env_lines.append(f"OPENAI_API_KEY={req.api_key.strip()}")
            env_lines.append(f"SIWU_LLM_BASE_URL={req.base_url.strip()}")
        env_lines.append(f"SIWU_LLM_MODEL={req.model.strip()}")
        env_lines.append("")
        env_path.write_text("\n".join(env_lines), encoding="utf-8")
        log.info("env_written | path=%s", str(env_path))
    except Exception as e:
        log.warning("env_write_failed | %s", str(e))

    # Update in-memory settings so the next request uses the new config
    if is_anthropic:
        settings.anthropic_api_key = req.api_key.strip()
        settings.llm_api_key = ""
        settings.llm_base_url = ""
        settings.llm_provider = "anthropic"
    else:
        settings.llm_api_key = req.api_key.strip()
        settings.llm_base_url = req.base_url.strip()
        settings.anthropic_api_key = ""
        settings.llm_provider = "openai_compatible"
    settings.default_model = req.model.strip()

    # Reinitialize agent resources with the new config
    try:
        init_agent_resources()
        log.info("agent_reinitialized_after_setup")
    except Exception as e:
        log.warning("agent_reinit_failed | %s", str(e))

    return {"ok": True, "message": "配置已保存", "path": str(CONFIG_TOML)}


@router.post("/setup/test")
async def test_connection(req: TestConnectionRequest):
    """测试 LLM 连接 —— 发送一条简短的测试消息并返回延迟。"""
    import time

    if not req.api_key.strip():
        return {"ok": False, "latency_ms": 0, "model": "", "error": "API Key 不能为空"}

    provider_key = req.provider.lower()
    is_openai_compat = provider_key in ("deepseek", "openai", "ollama", "custom")
    is_anthropic = provider_key == "anthropic"

    if not is_openai_compat and not is_anthropic:
        return {"ok": False, "latency_ms": 0, "model": "", "error": f"不支持的服务商: {req.provider}"}

    t0 = time.perf_counter()
    try:
        if is_openai_compat:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=req.api_key.strip(), base_url=req.base_url.strip(), timeout=30.0, max_retries=1)
            resp = await client.chat.completions.create(
                model=req.model.strip(),
                messages=[{"role": "user", "content": "reply 'ok'"}],
                max_tokens=20,
                temperature=0.0,
            )
            latency = round((time.perf_counter() - t0) * 1000, 1)
            return {"ok": True, "latency_ms": latency, "model": str(resp.model or req.model)}
        else:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=req.api_key.strip(), timeout=30.0, max_retries=1)
            resp = await client.messages.create(
                model=req.model.strip(),
                messages=[{"role": "user", "content": "reply 'ok'"}],
                max_tokens=20,
            )
            latency = round((time.perf_counter() - t0) * 1000, 1)
            return {"ok": True, "latency_ms": latency, "model": str(resp.model)}
    except Exception as e:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        err_msg = str(e)
        if len(err_msg) > 500:
            err_msg = err_msg[:500] + "…"
        log.warning("llm_test_connection_failed | provider=%s | %s", req.provider, err_msg[:120])
        return {"ok": False, "latency_ms": latency, "model": "", "error": err_msg}


# ── Version / Update routes ──────────────────────────────────────

class VersionInfoResponse(BaseModel):
    python_version: str = "0.0.3"
    electron_version: str = "0.0.3"
    latest_version: str = ""
    update_available: bool = False
    release_url: str = ""
    release_date: str = ""
    release_notes: str = ""
    download_urls: list[str] = []
    error: str = ""


def _get_pyproject_version() -> str:
    try:
        ppt = Path.cwd() / "pyproject.toml"
        if ppt.exists():
            text = ppt.read_text(encoding="utf-8")
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("version") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "0.0.3"


def _get_package_json_version() -> str:
    try:
        pj = Path.cwd() / "package.json"
        if pj.exists():
            data = json.loads(pj.read_text(encoding="utf-8"))
            return data.get("version", "0.0.3")
    except Exception:
        pass
    try:
        pj2 = Path.cwd() / "resources" / "app" / "package.json"
        if pj2.exists():
            data = json.loads(pj2.read_text(encoding="utf-8"))
            return data.get("version", "0.0.3")
    except Exception:
        pass
    return "0.0.3"


@router.get("/setup/version", response_model=VersionInfoResponse)
async def check_version():
    """Check version + latest GitHub release."""
    py_ver = _get_pyproject_version()
    el_ver = _get_package_json_version()

    result = VersionInfoResponse(python_version=py_ver, electron_version=el_ver)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://api.github.com/repos/Mourpheuon/Siwu-agent/releases/latest",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "Siwu-Agent"},
            )
            if r.status_code == 200:
                release = r.json()
                tag = release.get("tag_name", "").lstrip("v")
                result.latest_version = tag
                result.release_url = release.get("html_url", "")
                result.release_date = release.get("published_at", "")[:10]
                result.release_notes = (release.get("body") or "")[:3000]
                for asset in release.get("assets", []):
                    url = asset.get("browser_download_url", "")
                    if url:
                        result.download_urls.append(url)
                if tag and el_ver:
                    try:
                        from packaging.version import Version
                        if Version(tag) > Version(el_ver):
                            result.update_available = True
                    except Exception:
                        result.update_available = tag != el_ver
            elif r.status_code == 404:
                result.error = "未找到 GitHub Release"
            elif r.status_code == 403:
                result.error = "GitHub API 限流，请稍后重试"
            else:
                result.error = f"GitHub API 返回 {r.status_code}"
    except Exception as e:
        result.error = f"获取更新信息失败: {e}"

    return result


# ── Version bump helpers ─────────────────────────────────────────────

import re as _re

_VERSION_FILES = [
    "package.json",
    "siwu/web/package.json",
    "pyproject.toml",
    "siwu/__init__.py",
]

_VERSION_PATTERNS = {
    # file_basename: (regex, replacement_template)
    "package.json": (_re.compile(r'"version"\s*:\s*"[^"]*"'), '"version": "{version}"'),
    "pyproject.toml": (_re.compile(r'^version\s*=\s*"[^"]*"', _re.MULTILINE), 'version = "{version}"'),
    "__init__.py": (_re.compile(r'^__version__\s*=\s*"[^"]*"', _re.MULTILINE), '__version__ = "{version}"'),
}


def _bump_versions(version: str, cwd: Path) -> dict:
    """Write *version* into all version files. Returns a backup dict {path: old_content}."""
    backup = {}
    for rel in _VERSION_FILES:
        fp = cwd / rel
        if not fp.exists():
            continue
        old = fp.read_text(encoding="utf-8")
        backup[str(fp)] = old
        name = fp.name
        if name in _VERSION_PATTERNS:
            rx, tmpl = _VERSION_PATTERNS[name]
            new = rx.sub(tmpl.format(version=version), old)
        else:
            new = old
        if new != old:
            fp.write_text(new, encoding="utf-8")
            log.info("version_bumped | file=%s | version=%s", rel, version)
    return backup


def _restore_versions(backup: dict):
    """Restore original content from backup."""
    for path, content in backup.items():
        Path(path).write_text(content, encoding="utf-8")
    if backup:
        log.info("versions_restored | count=%s", len(backup))


# ── Developer / Build routes ────────────────────────────────────────

class BuildRequest(BaseModel):
    release_version: str = ""  # e.g. "0.0.4". Empty = skip release, just build.
    skip_build: bool = False   # skip npm build, go straight to release (existing .exe must be present)


async def _run_cmd(*args, cwd: str = None, check: bool = False, **kwargs) -> asyncio.subprocess.Process:
    """Spawn a subprocess. On Windows, use shell mode so .cmd wrappers are resolved."""
    if sys.platform == "win32":
        # Shell mode: quote args with spaces, join into one string
        cmd = " ".join(f'"{a}"' if " " in a else a for a in args)
        return await asyncio.create_subprocess_shell(cmd, cwd=cwd, **kwargs)
    else:
        return await asyncio.create_subprocess_exec(*args, cwd=cwd, **kwargs)


def _resolve_gh() -> str | None:
    """Return the path to gh CLI, or None if not installed."""
    import shutil
    found = shutil.which("gh")
    if found:
        return found
    if sys.platform == "win32":
        for candidate in [
            os.path.expandvars(r"%ProgramFiles%\GitHub CLI\gh.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\GitHub CLI\gh.exe"),
            os.path.expandvars(r"%LocalAppData%\Programs\GitHub CLI\gh.exe"),
            os.path.expandvars(r"%LocalAppData%\GitHubCLI\gh.exe"),
        ]:
            if os.path.isfile(candidate):
                return candidate
        found = shutil.which("gh.cmd")
        if found:
            return found
    return None


async def _gh_release_create(version: str, dist_dir: Path, *, gh: str) -> str:
    """Create a GitHub release using the gh CLI (uses local SSH keys).
    *gh* must be a resolved path to the gh executable."""
    tag = f"v{version}"
    cwd = str(dist_dir.parent)  # project root

    # Check gh CLI
    which = await _run_cmd(gh, "--version",
                           stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    sout, serr = await which.communicate()
    if which.returncode != 0:
        err = ((serr or sout).decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace").strip())
        raise Exception(f"GitHub CLI 不可用（{err[:200]}）")

    # Delete existing release if present (same version, overwrite)
    check = await _run_cmd(gh, "release", "view", tag, cwd=cwd,
                           stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await check.wait()
    if check.returncode == 0:
        log.info("github_release_exists | tag=%s — 删除后重建", tag)
        del_r = await _run_cmd(gh, "release", "delete", tag, "--yes", cwd=cwd,
                               stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        sout, serr = await del_r.communicate()
        if del_r.returncode != 0:
            err = (serr or sout).decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace").strip()
            log.warning("gh_release_delete_failed | %s", err[:200])

    # Create / update tag on remote
    # Strategy: push tag → create release (avoids interactive prompts)
    import shutil as _shutil2
    _git = _shutil2.which("git")
    if _git:
        # Delete stale remote tag if it exists (ignore errors)
        await _run_cmd(_git, "push", "origin", f":refs/tags/{tag}", cwd=cwd,
                       stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        # Force-update local tag to current HEAD
        await _run_cmd(_git, "tag", "-f", tag, cwd=cwd,
                       stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        # Push the tag to remote
        await _run_cmd(_git, "push", "origin", tag, cwd=cwd,
                       stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)

    body = f"自动构建发布 {tag}\n\n构建时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    args = [gh, "release", "create", tag,
            "--title", f"思悟 v{version}", "--notes", body]
    if version.startswith("0.0"):
        args.append("--prerelease")
    create = await _run_cmd(*args, cwd=cwd,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await create.communicate()
    if create.returncode != 0:
        err = stderr.decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace").strip() or \
              stdout.decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace").strip()
        log.error("gh_release_create_failed | %s", err[:300])
        raise Exception(f"创建 Release 失败: {err[:300]}")

    # Upload only the installer .exe matching this version
    assets = []
    import fnmatch as _fnmatch
    for p in sorted(dist_dir.iterdir()):
        if p.is_file() and _fnmatch.fnmatch(p.name, "*.exe") and version in p.name:
            assets.append(str(p))
    if assets:
        log.info("gh_upload_assets | count=%s", len(assets))
        up = await _run_cmd(gh, "release", "upload", tag, *assets, "--clobber", cwd=cwd,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        sout, serr = await up.communicate()
        if up.returncode != 0:
            log.warning("gh_upload_failed | %s", (serr or sout).decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace")[:200])

    release_url = f"https://github.com/Mourpheuon/Siwu-agent/releases/tag/{tag}"
    log.info("github_release_published | url=%s", release_url)
    return release_url


@router.get("/setup/release-check")
async def release_check(version: str = ""):
    """Check if a matching .exe already exists for this version."""
    if not version.strip():
        return {"exists": False, "files": []}
    dist_dir = Path.cwd() / "dist-electron"
    if not dist_dir.exists():
        return {"exists": False, "files": []}
    import fnmatch as _fnmatch
    files = []
    for p in sorted(dist_dir.iterdir()):
        if p.is_file() and _fnmatch.fnmatch(p.name, "*.exe") and version.strip() in p.name:
            files.append({"name": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 1)})
    return {"exists": len(files) > 0, "files": files}


@router.post("/setup/build-electron")
async def build_electron(req: BuildRequest = BuildRequest()):
    """Run electron-builder, stream output via SSE. Optionally auto-publish."""

    async def event_generator():
        cwd = Path.cwd()
        release_ver = req.release_version.strip()
        skip_build = req.skip_build and bool(release_ver)  # only meaningful with a version
        version_backup = None

        if release_ver:
            version_backup = _bump_versions(release_ver, cwd)
            yield _sse({"type": "log", "line": f"🔖 已将版本号同步为 {release_ver}"})

        dist_dir = cwd / "dist-electron"

        if not skip_build:
            log.info("electron_build_started | cwd=%s", str(cwd))

            # Clean up old build artifacts so only current version remains
            if dist_dir.exists():
                import fnmatch as _fnmatch
                old_count = 0
                for f in list(dist_dir.iterdir()):
                    try:
                        if f.is_file() and (_fnmatch.fnmatch(f.name, "*.exe") or
                                            _fnmatch.fnmatch(f.name, "*.blockmap") or
                                            _fnmatch.fnmatch(f.name, "*.yml") or
                                            _fnmatch.fnmatch(f.name, "*.yaml")):
                            f.unlink(); old_count += 1
                    except OSError:
                        pass
                if old_count:
                    yield _sse({"type": "log", "line": f"🧹 已清理 {old_count} 个旧构建文件"})

            yield _sse({"type": "log", "line": "📦 开始构建 Electron 包壳…"})
            yield _sse({"type": "log", "line": f"📂 工作目录: {cwd}"})
            yield _sse({"type": "status", "phase": "building"})

            try:
                process = await _run_cmd("npm", "run", "electron:build",
                                         cwd=str(cwd),
                                         stdout=asyncio.subprocess.PIPE,
                                         stderr=asyncio.subprocess.STDOUT)
            except FileNotFoundError:
                if version_backup: _restore_versions(version_backup)
                yield _sse({"type": "error", "message": "未找到 npm 命令，请确认 Node.js 已安装"})
                yield _sse({"type": "result", "ok": False, "error": "npm 未安装"})
                return
            except Exception as e:
                if version_backup: _restore_versions(version_backup)
                yield _sse({"type": "error", "message": f"无法启动构建进程: {e}"})
                yield _sse({"type": "result", "ok": False, "error": str(e)})
                return

            # Single loop: read stdout lines with timeout, send heartbeat when idle
            last_beat = 0
            yield _sse({"type": "heartbeat", "elapsed": 0})
            while True:
                try:
                    raw = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                except asyncio.TimeoutError:
                    last_beat += 5
                    yield _sse({"type": "heartbeat", "elapsed": last_beat})
                    continue
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    yield _sse({"type": "log", "line": text})

            await process.wait()
            returncode = process.returncode

            if version_backup:
                _restore_versions(version_backup)
                yield _sse({"type": "log", "line": "🔖 已恢复版本文件"})

            if returncode != 0:
                yield _sse({"type": "error", "message": f"构建失败，退出码 {process.returncode}"})
                yield _sse({"type": "result", "ok": False, "error": f"构建失败 (code={process.returncode})"})
                return

            built_files = sorted(p for p in dist_dir.iterdir() if p.is_file())
            yield _sse({"type": "log", "line": f"✅ 构建完成！输出 {len(built_files)} 个文件:"})
            for bf in built_files:
                size_mb = bf.stat().st_size / (1024 * 1024)
                yield _sse({"type": "log", "line": f"   {bf.name}  ({size_mb:.1f} MB)"})
        else:
            # skip_build: go straight to release
            yield _sse({"type": "log", "line": f"⏩ 跳过构建，使用已有 {release_ver} 产物"})

        # ── Common release path ──
        if release_ver:
            if skip_build:
                # dist_dir might have been shadowed by not-skip block; re-resolve
                dist_dir = cwd / "dist-electron"
            if not dist_dir.exists():
                dist_dir = cwd / "dist"
            if not dist_dir.exists():
                yield _sse({"type": "result", "ok": True, "message": "未找到输出目录"})
                return

            yield _sse({"type": "log", "line": ""})
            yield _sse({"type": "status", "phase": "releasing"})

            gh = _resolve_gh()
            if not gh:
                if sys.platform == "win32":
                    yield _sse({"type": "log", "line": "⬇️ 正在安装 GitHub CLI…"})
                    inst = await _run_cmd("winget", "install", "GitHub.cli",
                                          "--accept-package-agreements", "--accept-source-agreements",
                                          stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                    async for raw in inst.stdout:
                        txt = raw.decode("gbk" if sys.platform == "win32" else "utf-8", errors="replace").rstrip()
                        if txt: yield _sse({"type": "log", "line": f"   {txt}"})
                    await inst.wait()
                    gh = _resolve_gh()
                else:
                    yield _sse({"type": "log", "line": "未安装 GitHub CLI，macOS 请手动: brew install gh"})

            if not gh:
                yield _sse({"type": "log", "line": "❌ 自动安装失败，请手动安装 GitHub CLI 后重试"})
                yield _sse({"type": "result", "ok": True, "message": "GitHub CLI 未安装", "error": "请运行: winget install GitHub.cli"})
            else:
                yield _sse({"type": "log", "line": f"🚀 正在创建 GitHub Release v{release_ver}…"})
                try:
                    release_url = await _gh_release_create(release_ver, dist_dir, gh=gh)
                    yield _sse({"type": "log", "line": f"🎉 Release 已发布: {release_url}"})
                    yield _sse({"type": "result", "ok": True, "release_url": release_url, "version": release_ver})
                except Exception as e:
                    yield _sse({"type": "log", "line": f"❌ 发布失败: {e}"})
                    yield _sse({"type": "result", "ok": True, "message": "自动发布失败", "error": str(e)})
        elif not skip_build:
            yield _sse({"type": "result", "ok": True, "message": "构建完成"})
        else:
            yield _sse({"type": "result", "ok": False, "error": "跳过构建但没有版本号"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
