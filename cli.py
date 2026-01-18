from __future__ import annotations

import sys

import web_search_rag_standalone as rag


def _print_sources(state):
    sources = state.get("fetched_sources", [])
    if not sources:
        print("Sources: none")
        return
    print("Sources:")
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "").strip() or "(untitled)"
        url = src.get("url", "").strip() or "(no url)"
        print(f"{i}. {title} - {url}")


def main():
    print("PyAnsys Troubleshooting RAG")
    print("Type a question, or 'quit'/'exit' to end.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Exiting.")
            return

        print(f"Query: {query}")
        final_state = None
        for event in rag.run_stream(query, num_results=5):
            for node_name, state in event.items():
                if node_name == "search_web":
                    print("Search complete.")
                elif node_name == "fetch_sources":
                    print("Fetch complete.")
                elif node_name == "generate_answer":
                    print("Generate complete.")
                elif node_name == "handle_error":
                    print("Handled error.")
                final_state = state

        if not final_state:
            print("No output.")
            continue

        answer = final_state.get("answer", "").strip()
        if answer:
            print("\nAnswer:\n")
            print(answer)
        else:
            print("\nAnswer:\n")
            print("No answer generated.")

        print("")
        _print_sources(final_state)


if __name__ == "__main__":
    sys.exit(main())
