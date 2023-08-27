from models import AIAgent
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai
import logging

logging.basicConfig(level=logging.INFO)

class TransformerAgent(AIAgent):
    """
    An AI agent that uses a Hugging Face transformer model for processing input.
    """
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Load the transformer model and tokenizer based on the provided model name.
        """
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

    def process_input(self, input):
        """
        Process input using the loaded transformer model.
        """
        attributes_summary = self.generate_attribute_summary()
        tools_summary = " The available tools are: " + \
            ', '.join([f"{tool.name} which {tool.description}" for tool in self.tools]) + "."
        full_input = f"{attributes_summary}{tools_summary} {input}"
        inputs = self.tokenizer.encode(full_input, return_tensors='pt')
        
        try:
            outputs = self.model.generate(inputs, max_length=200)
        except Exception as e:
            logging.error(f"Failed to generate output: {e}")
            return "Error in generating output."
        
        answer = self.tokenizer.decode(outputs[0])
        return answer

    def step(self):
        """
        Log the action of the transformer AI agent taking a step.
        """
        logging.info(f"{self.model_name} is taking a step.")


class GPTAgent(AIAgent):
    """
    An AI agent that uses OpenAI's GPT model for processing input.
    """

    def __init__(self, api_key):
        super().__init__()
        openai.api_key = api_key
        self.load_model()

    def load_model(self):
        """
        For GPT, the model is loaded remotely, so there's no need to load it here.
        """
        pass

    def process_input(self, input):
        """
        Process input using the GPT model.
        """
        attributes_summary = self.generate_attribute_summary()
        tools_summary = " The available tools are: " + \
            ', '.join([f"{tool.name} which {tool.description}" for tool in self.tools]) + "."
        full_input = f"{attributes_summary}{tools_summary} {input}"
        
        try:
            response = 'Placeholder response for: ' + full_input
        except Exception as e:
            logging.error(f"Failed to generate output: {e}")
            return "Error in generating output."
        
        return response

    def step(self):
        """
        Log the action of the GPT-3 AI agent taking a step.
        """
        logging.info(f"{self.model_name} is taking a step.")
