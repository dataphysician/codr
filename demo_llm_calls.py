"""
Demo LLM Calls using Unified OpenAI API Completion

This demonstrates various ways to use the unified completion interface
with different providers, focusing on Keywell (default) and Claude.
"""

import litellm
from llm_client import (
    keywell_completion, anthropic_completion, openai_completion, gemini_completion,
    completion, chat, get_response_text
)


def demo_keywell_default_calls():
    """Demonstrate Keywell completion calls (default provider)."""
    print("=== Keywell Default Calls Demo ===\n")
    
    test_cases = [
        {
            "name": "Simple Question",
            "messages": [{"role": "user", "content": "What is machine learning?"}]
        },
        {
            "name": "Multi-turn Conversation",
            "messages": [
                {"role": "user", "content": "Hello! Can you help me understand Python?"},
                {"role": "assistant", "content": "Of course! Python is a versatile programming language. What specifically would you like to know?"},
                {"role": "user", "content": "How do I create a function in Python?"}
            ]
        },
        {
            "name": "Technical Question",
            "messages": [{"role": "user", "content": "Explain the difference between supervised and unsupervised learning in 2 sentences."}]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}:")
        try:
            response = keywell_completion(test_case["messages"])
            answer = get_response_text(response)
            print(f"   Response: {answer[:150]}...")
        except Exception as e:
            print(f"   Error: {e}")
        print()


def demo_claude_calls():
    """Demonstrate Claude (Anthropic) completion calls."""
    print("=== Claude (Anthropic) Calls Demo ===\n")
    
    test_cases = [
        {
            "name": "Creative Writing",
            "messages": [{"role": "user", "content": "Write a short haiku about coding."}]
        },
        {
            "name": "Analysis Task",
            "messages": [{"role": "user", "content": "Analyze the pros and cons of remote work in 3 bullet points each."}]
        },
        {
            "name": "Code Review",
            "messages": [{"role": "user", "content": "Review this Python code: def add(a, b): return a + b"}]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}:")
        try:
            response = anthropic_completion(test_case["messages"])
            answer = get_response_text(response)
            print(f"   Response: {answer[:150]}...")
        except Exception as e:
            print(f"   Error: {e}")
        print()


def demo_provider_comparison():
    """Compare responses from different providers for the same question."""
    print("=== Provider Comparison Demo ===\n")
    
    question = "Explain recursion in programming in simple terms."
    messages = [{"role": "user", "content": question}]
    
    providers = [
        ("Keywell", lambda: keywell_completion(messages)),
        ("Claude", lambda: anthropic_completion(messages)),
        ("OpenAI", lambda: openai_completion(messages)),
        # ("Gemini", lambda: gemini_completion(messages))
    ]
    
    print(f"Question: {question}\n")
    
    for provider_name, completion_func in providers:
        print(f"> {provider_name} Response:")
        try:
            response = completion_func()
            answer = get_response_text(response)
            print(f"   {answer[:200]}...")
        except Exception as e:
            print(f"   Error (expected if not configured): {e}")
        print()


def demo_generic_completion_interface():
    """Demonstrate the generic completion interface for easy provider switching."""
    print("=== Generic Completion Interface Demo ===\n")
    
    question = "What are the benefits of using type hints in Python?"
    
    # Test different providers using the same interface
    providers = ["keywell", "anthropic", "openai", "gemini"]
    
    for provider in providers:
        print(f"Using {provider} provider:")
        try:
            response = completion(
                messages=[{"role": "user", "content": question}],
                provider=provider
            )
            answer = get_response_text(response)
            print(f"   Response: {answer[:120]}...")
        except Exception as e:
            print(f"   Error (expected if not configured): {e}")
        print()


def demo_convenience_chat_function():
    """Demonstrate the convenience chat function for quick interactions."""
    print("=== Convenience Chat Function Demo ===\n")
    
    quick_questions = [
        "What is Docker?",
        "Explain REST APIs briefly.",
        "What's the difference between Git and GitHub?"
    ]
    
    for i, question in enumerate(quick_questions, 1):
        print(f"{i}. Quick question: {question}")
        
        # Try Keywell first (default), then Claude as fallback
        for provider in ["keywell", "anthropic"]:
            try:
                answer = chat(question, provider=provider)
                print(f"   {provider.title()}: {answer[:100]}...")
                break  # Stop on first successful response
            except Exception as e:
                if provider == "anthropic":  # Last provider
                    print(f"   All providers failed: {e}")
        print()


def demo_conversation_context():
    """Demonstrate maintaining conversation context across calls."""
    print("=== Conversation Context Demo ===\n")
    
    # Start a conversation
    conversation = []
    
    # User's first message
    user_msg1 = "I'm learning Python. Can you help me understand variables?"
    conversation.append({"role": "user", "content": user_msg1})
    
    print("User:", user_msg1)
    
    try:
        # Get response from Keywell
        response1 = keywell_completion(conversation)
        ai_response1 = get_response_text(response1)
        conversation.append({"role": "assistant", "content": ai_response1})
        
        print("Keywell:", ai_response1[:100], "...")
        print()
        
        # User's follow-up question
        user_msg2 = "Can you show me an example of variable assignment?"
        conversation.append({"role": "user", "content": user_msg2})
        
        print("User:", user_msg2)
        
        # Get contextual response
        response2 = keywell_completion(conversation)
        ai_response2 = get_response_text(response2)
        
        print("Keywell:", ai_response2[:150], "...")
        
    except Exception as e:
        print(f"Conversation demo error: {e}")
    
    print()


def demo_different_model_parameters():
    """Demonstrate using different model parameters."""
    print("=== Model Parameters Demo ===\n")
    
    base_message = [{"role": "user", "content": "Write a very short story about a robot."}]
    
    # Test with different providers and parameters
    test_configs = [
        {
            "name": "Keywell with default model",
            "func": lambda: keywell_completion(base_message)
        },
        {
            "name": "Claude with temperature",
            "func": lambda: anthropic_completion(base_message, temperature=0.8)
        },
        {
            "name": "Generic completion (Keywell default)",
            "func": lambda: completion(base_message, provider="keywell")
        },
        {
            "name": "OpenAI with max_tokens",
            "func": lambda: completion(base_message, provider="openai", max_tokens=100)
        }
    ]
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        try:
            response = config["func"]()
            answer = get_response_text(response)
            print(f"   Response: {answer[:120]}...")
        except Exception as e:
            print(f"   Error (expected if not configured): {e}")
        print()


def demo_raw_litellm_pattern():
    """Show the raw litellm.completion() pattern for direct usage."""
    print("=== Raw LiteLLM Pattern Demo ===\n")
    print("For direct litellm.completion() usage in agent code:\n")
    
    patterns = {
        "Keywell (Default)": '''
import litellm
from keywell import setup_keywell_handler

# Setup Keywell handler once
setup_keywell_handler()

# Use directly in agent code
response = litellm.completion(
    model=f"mydbx/{os.getenv('KEYWELL_MODEL_ID', '')}",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base=os.getenv("KEYWELL_ENDPOINT"),
    api_key=os.getenv("DATABRICKS_API_KEY"),
    optional_params={
        "sid": os.getenv("KEYWELL_SID", "")
    },
    timeout=60
)
        ''',
        
        "Claude": '''
response = litellm.completion(
    model="anthropic/claude-opus-4-1-20250805",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=60
)
        ''',
        
        "OpenAI": '''
response = litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60
)
        '''
    }
    
    for provider, code in patterns.items():
        print(f"{provider} Pattern:")
        print(code)
        print()


def main():
    """Run all demo functions."""
    print("> Unified OpenAI API Completion Demos\n")
    print("These demos show how to use the unified completion interface")
    print("with different providers, focusing on Keywell (default) and Claude.\n")
    
    print("=== Setup Notes:")
    print("• Copy .env.example to .env and add your API credentials")
    print("• Keywell is configured as the default provider")
    print("• Claude (Anthropic) is the main alternative shown")
    print("• All functions use the same OpenAI message format\n")
    
    # Run all demos
    demo_keywell_default_calls()
    demo_claude_calls()
    demo_provider_comparison()
    demo_generic_completion_interface()
    demo_convenience_chat_function()
    demo_conversation_context()
    demo_different_model_parameters()
    demo_raw_litellm_pattern()
    
    print("=== Demo Summary ===")
    print("• keywell_completion() - Direct Keywell calls")
    print("• anthropic_completion() - Direct Claude calls") 
    print("• completion(provider='...') - Generic interface")
    print("• chat('question', provider='...') - Convenience function")
    print("• Raw litellm.completion() - Direct LiteLLM usage")
    print("• All maintain OpenAI message format compatibility")


if __name__ == "__main__":
    main()