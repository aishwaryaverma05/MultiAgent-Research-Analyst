from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str, progress_callback=None) -> dict:
    """
    Runs the 4-step research pipeline.

    progress_callback (optional): a function with the signature
        callback(step_index: int, label: str, output: str | None)
    - called with output=None when a step STARTS
    - called with the step's output string when a step FINISHES
    """

    def notify(step, label, output=None):
        if progress_callback:
            progress_callback(step, label, output)

    state = {}

    # step 1: search agent
    print("\n" + "=" * 50)
    print("Step-1: Search agent is working...")
    print("=" * 50)
    notify(0, "Search Agent")

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [{"role": "user", "content": f"Find recent, reliable and detailed information about: {topic}"}]
    })
    state["search_results"] = search_result["messages"][-1].content

    print("\n search result are:", state["search_results"])
    notify(0, "Search Agent", state["search_results"])

    # step 2: reader agent
    print("\n" + "=" * 50)
    print("Step-2: Reader agent is fetching top resources...")
    print("=" * 50)
    notify(1, "Reader Agent")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [(
            "user", f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped content\n", state["scraped_content"])
    notify(1, "Reader Agent", state["scraped_content"])

    # step 3: writer chain
    print("\n" + "=" * 50)
    print("Step-3: Writer chain is generating the research report...")
    print("=" * 50)
    notify(2, "Writer Chain")

    research_combined_results = (
        f"SEARCH RESULTS:\n {state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined_results
    })

    print("\nResearch report generated:\n", state["report"])
    notify(2, "Writer Chain", str(state["report"]))

    # step 4: critic chain
    print("\n" + "=" * 50)
    print("Step-4: Critic chain is evaluating the research report...")
    print("=" * 50)
    notify(3, "Critic Chain")

    state["Feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCritic feedback:\n", state["Feedback"])
    notify(3, "Critic Chain", str(state["Feedback"]))

    return state


if __name__ == "__main__":
    topic = input("Enter the research topic: ")
    run_research_pipeline(topic)