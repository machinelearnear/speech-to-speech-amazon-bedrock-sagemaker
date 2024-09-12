import boto3
from botocore.exceptions import ClientError
from baseHandler import BaseHandler
from threading import Thread
from queue import Queue
import logging
from nltk import sent_tokenize

logger = logging.getLogger(__name__)

class BedrockModelHandler(BaseHandler):
    def setup(
        self,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.5,
        top_k=200,
        user_role="user",
        chat_size=10,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.user_role = user_role
        self.chat = []
        self.chat_size = chat_size
        
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.append({"role": init_chat_role, "content": [{"text": init_chat_prompt}]})
        
        self.bedrock_client = boto3.client(service_name='bedrock-runtime')
        
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input_text = "Repeat the word 'home'."
        self.process(dummy_input_text)
        logger.info(f"{self.__class__.__name__}: warmed up!")

    def process(self, prompt):
        logger.debug("inferring with language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            prompt = f"Please reply to my message in {self._get_language_name(language_code)}. " + prompt

        self.chat.append({"role": self.user_role, "content": [{"text": prompt}]})
        
        # Trim chat history if it exceeds chat_size
        if len(self.chat) > self.chat_size:
            self.chat = self.chat[-self.chat_size:]

        system_prompts = [{"text": "You are a helpful AI assistant."}]
        inference_config = {"temperature": self.temperature}
        additional_model_fields = {"top_k": self.top_k}

        try:
            response = self.bedrock_client.converse_stream(
                modelId=self.model_id,
                messages=self.chat,
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            stream = response.get('stream')
            if stream:
                generated_text = ""
                for event in stream:
                    if 'contentBlockDelta' in event:
                        new_text = event['contentBlockDelta']['delta']['text']
                        generated_text += new_text
                        sentences = sent_tokenize(generated_text)
                        if len(sentences) > 1:
                            yield (sentences[0], language_code)
                            generated_text = new_text

                # Don't forget the last sentence
                if generated_text:
                    yield (generated_text, language_code)

                self.chat.append({"role": "assistant", "content": [{"text": generated_text}]})

        except ClientError as err:
            message = err.response['Error']['Message']
            logger.error("A client error occurred: %s", message)
            yield (f"An error occurred: {message}", language_code)

    def _get_language_name(self, language_code):
        language_map = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
        }
        return language_map.get(language_code, "the same language as the input")