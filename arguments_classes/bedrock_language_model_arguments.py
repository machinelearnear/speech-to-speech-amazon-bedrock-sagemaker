from dataclasses import dataclass, field

@dataclass
class BedrockLanguageModelHandlerArguments:
    model_id: str = field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        metadata={
            "help": "The Amazon Bedrock model ID to use. Default is 'anthropic.claude-3-sonnet-20240229-v1:0'."
        },
    )
    temperature: float = field(
        default=0.5,
        metadata={
            "help": "Controls randomness in the model's output. Lower values make the output more focused and deterministic. Default is 0.5."
        },
    )
    top_k: int = field(
        default=200,
        metadata={
            "help": "Limits the number of top tokens considered for each step of text generation. Default is 200."
        },
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    chat_size: int = field(
        default=10,
        metadata={
            "help": "Number of interactions to keep in the chat history. Default is 10."
        },
    )
    init_chat_role: str = field(
        default=None,
        metadata={
            "help": "Initial role for setting up the chat context. Default is None."
        },
    )
    init_chat_prompt: str = field(
        default="You are a helpful AI assistant.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    aws_region: str = field(
        default="us-east-1",
        metadata={
            "help": "The AWS region where the Bedrock service is located. Default is 'us-east-1'."
        },
    )
    aws_access_key_id: str = field(
        default=None,
        metadata={
            "help": "AWS access key ID for authentication. If not provided, will use the default AWS configuration."
        },
    )
    aws_secret_access_key: str = field(
        default=None,
        metadata={
            "help": "AWS secret access key for authentication. If not provided, will use the default AWS configuration."
        },
    )