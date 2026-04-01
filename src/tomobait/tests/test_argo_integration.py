"""
Integration tests for ANL Argo OpenAI-compatible endpoint.

These tests require ANL Argo access and credentials.
Set ARGO_USER environment variable or configure api_key in config.yaml.

Run tests:
    export ARGO_USER=your_anl_username
    uv run pytest src/tomobait/tests/test_argo_integration.py -v

Skip tests if no credentials:
    uv run pytest src/tomobait/tests/ -v -m "not integration"
"""

import os

import pytest

# Configuration
ARGO_BASE_URL = "https://apps-dev.inside.anl.gov/argoapi/v1/"
ARGO_USER = os.getenv("ARGO_USER")  # ANL domain username as API key
ARGO_MODEL = os.getenv("ARGO_MODEL", "gpt4o")  # Default to gpt4o


def requires_argo_credentials():
    """Skip decorator for tests requiring Argo credentials."""
    return pytest.mark.skipif(
        not ARGO_USER,
        reason="ARGO_USER environment variable not set"
    )


@pytest.fixture
def argo_client():
    """Create OpenAI client configured for Argo."""
    if not ARGO_USER:
        pytest.skip("ARGO_USER environment variable not set")

    from openai import OpenAI

    return OpenAI(
        base_url=ARGO_BASE_URL,
        api_key=ARGO_USER,
    )


@pytest.fixture
def argo_model():
    """Return the model to use for tests."""
    return ARGO_MODEL


@requires_argo_credentials()
class TestArgoConnection:
    """Test basic connectivity to Argo endpoint."""

    def test_basic_completion(self, argo_client, argo_model):
        """Test basic chat completion works."""
        response = argo_client.chat.completions.create(
            model=argo_model,
            messages=[{"role": "user", "content": "Say hello in exactly 3 words"}],
            max_tokens=50,
        )

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_response_structure(self, argo_client, argo_model):
        """Test that response has expected OpenAI-compatible structure."""
        response = argo_client.chat.completions.create(
            model=argo_model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

        # Verify OpenAI-compatible response structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert response.choices[0].message.role == "assistant"


@requires_argo_credentials()
class TestArgoToolCalling:
    """Test tool/function calling support (critical for TomoBait)."""

    def test_tool_definition(self, argo_client, argo_model):
        """Test that tools can be defined and processed."""
        tools = [{
            "type": "function",
            "function": {
                "name": "query_documentation",
                "description": "Query the documentation database for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for the documentation"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

        response = argo_client.chat.completions.create(
            model=argo_model,
            messages=[{
                "role": "user",
                "content": "Please search the documentation for tomography setup instructions"
            }],
            tools=tools,
            max_tokens=200,
        )

        # Model should either call the tool or respond with content
        message = response.choices[0].message
        assert message.tool_calls is not None or message.content is not None

    def test_tool_call_format(self, argo_client, argo_model):
        """Test that tool calls have correct format when triggered."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }]

        response = argo_client.chat.completions.create(
            model=argo_model,
            messages=[{
                "role": "user",
                "content": "What's the weather in Chicago? Use the get_weather tool."
            }],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )

        message = response.choices[0].message

        # If tool was called, verify structure
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            assert tool_call.type == "function"
            assert tool_call.function.name == "get_weather"
            assert tool_call.function.arguments is not None


@requires_argo_credentials()
class TestArgoMultiTurn:
    """Test multi-turn conversation support."""

    def test_conversation_context(self, argo_client, argo_model):
        """Test that conversation context is preserved across turns."""
        messages = [
            {"role": "user", "content": "My name is TestUser."},
        ]

        # First turn
        response1 = argo_client.chat.completions.create(
            model=argo_model,
            messages=messages,
            max_tokens=50,
        )

        # Add assistant response and follow-up
        messages.append({
            "role": "assistant",
            "content": response1.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "What is my name?"
        })

        # Second turn - should remember the name
        response2 = argo_client.chat.completions.create(
            model=argo_model,
            messages=messages,
            max_tokens=50,
        )

        # The response should contain "TestUser"
        assert "TestUser" in response2.choices[0].message.content


@requires_argo_credentials()
class TestArgoAutogenIntegration:
    """Test Argo with Autogen agents."""

    def test_autogen_config_creation(self, argo_model):
        """Test creating an Autogen LLM config for Argo."""
        llm_config = {
            "config_list": [{
                "model": argo_model,
                "base_url": ARGO_BASE_URL,
                "api_key": ARGO_USER,
            }]
        }

        # Verify config structure
        assert "config_list" in llm_config
        assert len(llm_config["config_list"]) == 1
        assert llm_config["config_list"][0]["model"] == argo_model
        assert llm_config["config_list"][0]["base_url"] == ARGO_BASE_URL

    def test_assistant_agent_creation(self, argo_model):
        """Test creating an Autogen AssistantAgent with Argo config."""
        from autogen import AssistantAgent

        llm_config = {
            "config_list": [{
                "model": argo_model,
                "base_url": ARGO_BASE_URL,
                "api_key": ARGO_USER,
            }]
        }

        agent = AssistantAgent(
            name="test_agent",
            llm_config=llm_config,
            system_message="You are a helpful assistant.",
        )

        assert agent is not None
        assert agent.name == "test_agent"


@requires_argo_credentials()
class TestArgoErrorHandling:
    """Test error handling for Argo endpoint."""

    def test_invalid_model(self, argo_client):
        """Test error handling for invalid model name."""
        with pytest.raises(Exception):
            argo_client.chat.completions.create(
                model="invalid-model-name-xyz",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )


class TestArgoConfigValidation:
    """Test configuration validation (no credentials required)."""

    def test_base_url_format(self):
        """Test that base URL is properly formatted."""
        assert ARGO_BASE_URL.endswith("/")
        assert ARGO_BASE_URL.startswith("https://")

    def test_default_model(self):
        """Test that default model is set."""
        assert ARGO_MODEL is not None
        assert len(ARGO_MODEL) > 0
