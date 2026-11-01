from uuid import uuid4

from backend.graphs.main_graph import graph


def run_workflow(question: str) -> dict:
    """
    Executes the Talk2SQL LangGraph workflow.

    Args:
        question: Natural-language question submitted by the user.

    Returns:
        Final state returned by the compiled LangGraph.
    """

    # Generate a unique thread identifier.
    thread_id = str(uuid4())

    # Prepare the initial graph state.
    initial_state = {
        "user_question": question
    }
    
    # Configure the LangGraph execution.
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Execute the graph.
    result = graph.invoke(
        initial_state,
        config=config
    )

    return result


def main():
    """
    Main application entry point.
    """

    print("=" * 55)
    print("                 TALK2SQL")
    print("       Natural Language to SQL Analytics")
    print("=" * 55)

    question = input("\nEnter your question: ").strip()

    # Validate user input.
    if not question:
        print("\nError: Question cannot be empty.")
        return

    print("\nProcessing your question...")

    try:
        result = run_workflow(question)

        print("\n" + "=" * 55)
        print("             WORKFLOW RESULT")
        print("=" * 55)

        print(f"Status: {result.get('status')}")
        print(f"Request ID: {result.get('request_id')}")
        print(f"Conversation ID: {result.get('conversation_id')}")
        print(f"Intent: {result.get('intent')}")
        print(f"Next step: {result.get('next_step')}")

        print("\nMessages:")

        for message in result.get("messages", []):
            print(f"- {message.content}")

        print("\n" + "=" * 55)

    except Exception as exc:
        print("\nAn error occurred while executing Talk2SQL.")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()



