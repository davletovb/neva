from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai
from models import AIAgent
import logging


class TransformerAIAgent(AIAgent):
    """
    An AI agent that uses a Hugging Face transformer model.
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {e}")

    def process_input(self, input):
        attributes_summary = self.generate_attribute_summary()
        tools_summary = " The available tools are: " + \
            ', '.join(
                [f"{tool.name} which {tool.description}" for tool in self.tools]) + "."
        full_input = f"{attributes_summary}{tools_summary} {input}"
        inputs = self.tokenizer.encode(full_input, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=200)
        answer = self.tokenizer.decode(outputs[0])
        return answer

    def step(self):
        logging.info(f"{self.model_name} is taking a step.")


class ChatGPTAgent(AIAgent):
    """
    An AI agent that uses OpenAI's GPT-3 model for processing input.
    """

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.load_model()

    def load_model(self):
        # For GPT-3, the model is loaded remotely, so there's no need to load it here
        pass

    def process_input(self, input):
        attributes_summary = self.generate_attribute_summary()
        tools_summary = " The available tools are: " + \
            ', '.join(
                [f"{tool.name} which {tool.description}" for tool in self.tools]) + "."
        full_input = f"{attributes_summary}{tools_summary} {input}"
        response = 'Placeholder response for: ' + full_input
        return response

    def step(self):
        logging.info(f"{self.model_name} is taking a step.")
