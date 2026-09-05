from __future__ import annotations

import re
from typing import Any

from jinja2 import BaseLoader


_NAV_ITEMS = [
    ("/", "总览", "root"),
    ("/batches", "数据批次", "exact"),
    ("/batches/upload", "数据导入", "prefix"),
    ("/review/bulk", "快速审核", "prefix"),
    ("/review", "单张审核", "exact"),
    ("/species", "鱼种管理", "prefix"),
    ("/fish-knowledge", "鱼鉴内容", "prefix"),
    ("/feedback", "用户反馈", "prefix"),
    ("/datasets", "数据集", "prefix"),
    ("/crop-datasets", "Crop Dataset", "prefix"),
    ("/training", "模型训练", "prefix"),
    ("/inference", "模型实测", "prefix"),
    ("/intelligence", "模型智能分析", "prefix"),
    ("/crop-qa", "Crop QA", "prefix"),
    ("/debug/detector-parity", "Detector Parity", "prefix"),
]
_NAV_HREFS = {href for href, _label, _mode in _NAV_ITEMS}

_HEADER_RE = re.compile(
    r"<header\b[^>]*>(?P<body>.*?)</header>",
    re.IGNORECASE | re.DOTALL,
)
_ANCHOR_RE = re.compile(
    r"<a\b[^>]*\bhref=(?P<quote>[\"'])(?P<href>[^\"']+)(?P=quote)[^>]*>.*?</a>",
    re.IGNORECASE | re.DOTALL,
)


def _active_expr(path: str, mode: str) -> str:
    if mode == "root":
        return "nav_path == '/'"
    if mode == "exact":
        return f"nav_path == '{path}'"
    return f"nav_path.startswith('{path}')"


def _canonical_nav() -> str:
    links = []
    for href, label, mode in _NAV_ITEMS:
        expr = _active_expr(href, mode)
        links.append(
            f'<a href="{href}" class="{{{{ \'active\' if {expr} else \'\' }}}}" '
            f'{{% if {expr} %}}aria-current="page"{{% endif %}}>{label}</a>'
        )
    return """
<style id="yujian-unified-nav-style">
  header.app-nav{background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 20px;display:flex;gap:6px;align-items:center;position:sticky;top:0;z-index:50;overflow-x:auto;white-space:nowrap;scrollbar-width:thin}
  header.app-nav a{display:inline-flex;align-items:center;min-height:34px;padding:0 10px;border-radius:8px;text-decoration:none;color:#4b5563;font-weight:650;font-size:14px;flex:0 0 auto}
  header.app-nav a:hover{background:#f3f4f6;color:#111827}
  header.app-nav a.active{background:#111827;color:#fff}
  header.app-nav .filters{flex:0 0 auto;margin-left:auto}
  @media(max-width:760px){header.app-nav{padding:8px 12px;gap:4px}header.app-nav a{padding:0 9px;font-size:13px}}
</style>
{% set nav_path = request.url.path %}
<header class="app-nav" aria-label="主导航">
""" + "\n".join(links) + "\n</header>"


CANONICAL_NAV = _canonical_nav()


def _preserve_header_controls(body: str) -> str:
    """Remove legacy primary-nav links while retaining page-specific header controls."""

    def replace_anchor(match: re.Match[str]) -> str:
        if match.group("href") in _NAV_HREFS:
            return ""
        return match.group(0)

    return _ANCHOR_RE.sub(replace_anchor, body).strip()


def _replace_header(match: re.Match[str]) -> str:
    controls = _preserve_header_controls(match.group("body"))
    if not controls:
        return CANONICAL_NAV
    return CANONICAL_NAV.replace("\n</header>", f"\n{controls}\n</header>")


class UnifiedNavLoader(BaseLoader):
    """Wrap an existing Jinja loader and normalize the first page header."""

    def __init__(self, delegate: BaseLoader):
        self.delegate = delegate

    def get_source(self, environment: Any, template: str):
        source, filename, uptodate = self.delegate.get_source(environment, template)
        if template.endswith(".html") and _HEADER_RE.search(source):
            source = _HEADER_RE.sub(_replace_header, source, count=1)
        return source, filename, uptodate

    def list_templates(self):
        if hasattr(self.delegate, "list_templates"):
            return self.delegate.list_templates()
        return []


def install_unified_nav(templates: Any) -> None:
    """Install the shared navigation exactly once on a Jinja2Templates instance."""
    loader = templates.env.loader
    if loader is None or isinstance(loader, UnifiedNavLoader):
        return
    templates.env.loader = UnifiedNavLoader(loader)
    templates.env.cache.clear()
