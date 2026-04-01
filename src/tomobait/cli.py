import argparse
import sys

from tomobait.agents import LLMNotConfiguredError, get_llm_status, run_agent_chat


def main():
    parser = argparse.ArgumentParser(description="TomoBait CLI")
    parser.add_argument("question", type=str, help="The question to ask the agent")
    parser.add_argument(
        "--status", action="store_true", help="Show LLM configuration status"
    )
    args = parser.parse_args()

    # Show status if requested
    if args.status:
        status = get_llm_status()
        print(f"Provider: {status['provider']}")
        print(f"Model: {status['model']}")
        print(f"API Key Env: {status['api_key_env']}")
        print(f"Available: {status['available']}")
        if status['error']:
            print(f"Error: {status['error']}")
        return

    try:
        run_agent_chat(args.question)
    except LLMNotConfiguredError as e:
        status = get_llm_status()
        print(f"❌ LLM not configured: {e}", file=sys.stderr)
        env_var = status['api_key_env']
        print(f"   Set {env_var} or use different provider", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
