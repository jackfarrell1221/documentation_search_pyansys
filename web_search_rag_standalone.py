from __future__ import annotations

from typing import TypedDict, List, Dict, Optional

import warnings
import requests
import urllib3
from ddgs import DDGS
import trafilatura
from langgraph.graph import StateGraph, END
import ollama


class RAGState(TypedDict):
    query: str
    search_results: List[Dict[str, str]]
    fetched_sources: List[Dict[str, str]]
    answer: str
    error: Optional[str]
    num_results: int


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


def _safe_get(url: str, timeout: int = 15) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
        if resp.status_code >= 400:
            return None
        return resp.text
    except requests.RequestException:
        return None


def _normalize_result(item: Dict[str, str]) -> Dict[str, str]:
    return {
        "title": item.get("title") or item.get("heading") or "",
        "url": item.get("href") or item.get("url") or "",
        "snippet": item.get("body") or item.get("snippet") or "",
    }


def _augment_query(query: str) -> str:
    q = query.strip()
    if not q:
        return q
    lower = q.lower()
    if "pyansys" in lower or "ansys" in lower:
        return q
    return f"PyAnsys {q}"


def search_web(state: RAGState) -> RAGState:
    query = _augment_query(state.get("query", ""))
    num_results = state.get("num_results", 5)
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=num_results))
        results = [_normalize_result(item) for item in raw_results]
        return {
            **state,
            "search_results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            **state,
            "error": f"search_web failed: {exc}",
        }


def fetch_sources(state: RAGState) -> RAGState:
    results = state.get("search_results", [])
    sources: List[Dict[str, str]] = []
    try:
        for item in results[:2]:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            content = ""
            from_snippet = "false"
            if url:
                html = _safe_get(url)
                if html:
                    extracted = trafilatura.extract(html)
                    if extracted and len(extracted.strip()) >= 200:
                        content = extracted.strip()
                    else:
                        content = snippet
                        from_snippet = "true"
                else:
                    content = snippet
                    from_snippet = "true"
            else:
                content = snippet
                from_snippet = "true"
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "from_snippet": from_snippet,
                }
            )

        if not sources:
            return {
                **state,
                "fetched_sources": [],
                "error": "No sources fetched.",
            }

        return {
            **state,
            "fetched_sources": sources,
            "error": None,
        }
    except Exception as exc:
        return {
            **state,
            "error": f"fetch_sources failed: {exc}",
        }


def generate_answer(state: RAGState) -> RAGState:
    query = state.get("query", "")
    sources = state.get("fetched_sources", [])
    try:
        context_blocks = []
        for i, src in enumerate(sources, start=1):
            title = src.get("title", "").strip()
            url = src.get("url", "").strip()
            content = src.get("content", "").strip()
            context_blocks.append(
                f"Source {i}: {title}\nURL: {url}\nContent:\n{content}\n"
            )
        context_text = "\n".join(context_blocks)

        system_prompt = (
            "You are a PyAnsys troubleshooting assistant. "
            "Only answer PyAnsys error troubleshooting and problem-solving questions. "
            "If the question is general or unrelated, say you can only help with PyAnsys issues. "
            "Use the provided sources to craft a concise, accurate solution. "
            "If information is missing, say what is unknown and suggest safe next steps."
        )
        user_prompt = (
            f"Question: {query}\n\n"
            f"Sources:\n{context_text}\n\nAnswer:"
        )

        client = ollama.Client()
        response = client.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if hasattr(client, "close"):
            client.close()
        answer = response.get("message", {}).get("content", "")
        if answer:
            answer = f"{answer.strip()}\n\nThis is the fix."
        return {
            **state,
            "answer": answer.strip(),
            "error": None,
        }
    except Exception as exc:
        try:
            if "client" in locals() and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
        return {
            **state,
            "error": f"generate_answer failed: {exc}",
        }


def handle_error(state: RAGState) -> RAGState:
    error = state.get("error") or "Unknown error"
    fallback = (
        "I ran into an issue while processing the request. "
        "Please try again or provide a more specific PyAnsys question. "
        f"Details: {error}\n\nThis is the fix."
    )
    return {
        **state,
        "answer": fallback,
    }


def _route_after_node(state: RAGState) -> str:
    return "handle_error" if state.get("error") else "continue"


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("search_web", search_web)
    graph.add_node("fetch_sources", fetch_sources)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("search_web")

    graph.add_conditional_edges(
        "search_web",
        _route_after_node,
        {"continue": "fetch_sources", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "fetch_sources",
        _route_after_node,
        {"continue": "generate_answer", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "generate_answer",
        _route_after_node,
        {"continue": END, "handle_error": "handle_error"},
    )
    graph.add_edge("handle_error", END)

    return graph.compile()


_APP = build_graph()


def run(query: str, num_results: int = 5) -> RAGState:
    state: RAGState = {
        "query": query,
        "search_results": [],
        "fetched_sources": [],
        "answer": "",
        "error": None,
        "num_results": num_results,
    }
    return _APP.invoke(state)


def run_stream(query: str, num_results: int = 5):
    state: RAGState = {
        "query": query,
        "search_results": [],
        "fetched_sources": [],
        "answer": "",
        "error": None,
        "num_results": num_results,
    }
    return _APP.stream(state)