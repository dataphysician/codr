"""
Agentic Workflow Demo with Thin LiteLLM Wrapper

This demonstrates how agents can use the thin wrapper functions to easily
make litellm.completion() calls while switching between providers.
"""

from llm_client import (
    openai_completion, anthropic_completion, gemini_completion,
    completion, chat, get_response_text
)


class SimpleAgent:
    """
    Simple agent that can use any provider through direct litellm.completion() calls.
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.conversation_history: list[dict[str, str]] = []
    
    def chat_openai(self, user_message: str, **kwargs) -> str:
        """Chat using OpenAI (direct litellm.completion call)."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = openai_completion(self.conversation_history, **kwargs)
        agent_response = get_response_text(response)
        
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        return agent_response
    
    def chat_anthropic(self, user_message: str, **kwargs) -> str:
        """Chat using Anthropic (direct litellm.completion call)."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = anthropic_completion(self.conversation_history, **kwargs)
        agent_response = get_response_text(response)
        
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        return agent_response
    
    def chat_gemini(self, user_message: str, **kwargs) -> str:
        """Chat using Gemini (direct litellm.completion call)."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = gemini_completion(self.conversation_history, **kwargs)
        agent_response = get_response_text(response)
        
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        return agent_response
    
    
    def chat_any(self, user_message: str, provider: str = "openai", **kwargs) -> str:
        """Chat using any provider via generic completion function."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = completion(self.conversation_history, provider=provider, **kwargs)
        agent_response = get_response_text(response)
        
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        return agent_response


def demo_direct_calls():
    """Demonstrate the direct litellm.completion() pattern for agents."""
    print("=== Direct LiteLLM Calls Demo ===\n")
    
    messages = [{"role": "user", "content": "What is 2+2? Be very brief."}]
    
    providers = [
        ("OpenAI", lambda: openai_completion(messages)),
        ("Anthropic", lambda: anthropic_completion(messages)),
        ("Gemini", lambda: gemini_completion(messages))
    ]
    
    for provider_name, completion_func in providers:
        print(f"Testing {provider_name}:")
        try:
            response = completion_func()
            print(f"  Response: {get_response_text(response)}")
        except Exception as e:
            print(f"  Error (expected if not configured): {e}")
        print()


def demo_agent_provider_switching():
    """Demonstrate an agent switching between providers."""
    print("=== Agent Provider Switching Demo ===\n")
    
    agent = SimpleAgent("Multi-Provider Agent")
    
    test_cases = [
        ("OpenAI", agent.chat_openai, "Hello! What's the capital of France?"),
        ("Anthropic", agent.chat_anthropic, "That's correct. What about Germany?"), 
        ("Gemini", agent.chat_gemini, "Good. Now what about Japan?")
    ]
    
    for provider_name, chat_method, message in test_cases:
        print(f"Using {provider_name}: {message}")
        try:
            response = chat_method(message)
            print(f"  Response: {response[:100]}...")
        except Exception as e:
            print(f"  Error (expected if not configured): {e}")
        print()


def demo_workflow_with_different_providers():
    """Demonstrate a workflow using different providers for different tasks."""
    print("=== Workflow with Different Providers Demo ===\n")
    
    # Workflow: Research (OpenAI) -> Analysis (OpenAI) -> Creative (Anthropic) -> Summary (Gemini)
    workflow_steps = [
        {
            "step": "Research",
            "provider": "openai",
            "prompt": "List 3 key benefits of renewable energy sources."
        },
        {
            "step": "Analysis", 
            "provider": "openai",
            "prompt": "Analyze the economic implications of the previous points."
        },
        {
            "step": "Creative",
            "provider": "anthropic", 
            "prompt": "Write a compelling headline about renewable energy benefits."
        },
        {
            "step": "Summary",
            "provider": "gemini",
            "prompt": "Summarize all the above information in 2 sentences."
        }
    ]
    
    agent = SimpleAgent("Workflow Agent")
    
    for step_info in workflow_steps:
        print(f"Step: {step_info['step']} (using {step_info['provider']})")
        print(f"Prompt: {step_info['prompt']}")
        
        try:
            response = agent.chat_any(
                step_info['prompt'], 
                provider=step_info['provider']
            )
            print(f"Response: {response[:150]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)


def demo_convenience_functions():
    """Demonstrate the convenience chat function for quick interactions."""
    print("=== Convenience Functions Demo ===\n")
    
    test_message = "What is machine learning in one sentence?"
    
    providers = ["openai", "anthropic", "gemini"]
    
    for provider in providers:
        print(f"Quick chat with {provider}:")
        try:
            response = chat(test_message, provider=provider)
            print(f"  Response: {response}")
        except Exception as e:
            print(f"  Error (expected if not configured): {e}")
        print()


def demo_raw_litellm_patterns():
    """Show the raw litellm.completion() patterns that agents can use directly."""
    print("=== Raw LiteLLM Patterns ===\n")
    print("Agents can also use these direct patterns:\n")
    
    patterns = {
        "OpenAI": '''
import litellm
response = litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60
)
print(response.choices[0].message.content)
        ''',
        
        "Anthropic": '''
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=60
)
        ''',
        
        "Gemini": '''
response = litellm.completion(
    model="gemini/gemini-2.0-flash-exp", 
    messages=[{"role": "user", "content": "Hello!"}],
    api_key=os.getenv("GEMINI_API_KEY"),
    timeout=60
)
        ''',
        
    }
    
    for provider, code in patterns.items():
        print(f"{provider} Pattern:")
        print(code)
        print()


def main():
    """Main demo runner."""
    print("🤖 Thin LiteLLM Wrapper Demo\n")
    print("This shows how agents can easily use litellm.completion() calls")
    print("with different providers through simple wrapper functions.\n")
    
    print("📝 Configuration:")
    print("• Copy .env.example to .env and add your API keys")
    print("• Defaults are loaded from environment variables")
    print("• All parameters can be overridden at call time\n")
    
    # Run demos
    demo_direct_calls()
    demo_agent_provider_switching()
    demo_workflow_with_different_providers()
    demo_convenience_functions()
    demo_raw_litellm_patterns()
    
    print("=== Key Benefits ===")
    print("• Thin wrapper over direct litellm.completion() calls")
    print("• Easy provider switching: openai_completion() → anthropic_completion()")
    print("• Environment variable defaults with runtime overrides")
    print("• Direct access to raw LiteLLM patterns when needed")
    print("• Provider-specific parameter handling for different models")
    print("• OpenAI message format compatibility across all providers")


if __name__ == "__main__":
    main()