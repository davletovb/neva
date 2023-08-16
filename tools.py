import math
import wikipedia
from google import Translator
from summarizer import Summarizer
from models import Tool
import logging


class MathTool(Tool):
    """
    A tool for performing mathematical operations.
    """

    def use(self, task):
        try:
            # Use Python's built-in math functions to perform the task
            return eval(task)
        except Exception as e:
            raise RuntimeError(f"Failed to use MathTool: {e}")


class WikipediaTool(Tool):
    """
    A tool for searching Wikipedia.
    """

    def use(self, task):
        try:
            # Use the Wikipedia library to search for the task
            return wikipedia.summary(task, sentences=2)
        except Exception as e:
            raise RuntimeError(f"Failed to use WikipediaTool: {e}")


class TranslatorTool(Tool):
    """
    A tool for translating text.
    """

    def use(self, task):
        try:
            # Use the Googletrans library to translate the task
            translator = Translator()
            return translator.translate(task).text
        except Exception as e:
            raise RuntimeError(f"Failed to use TranslatorTool: {e}")


class SummarizerTool(Tool):
    """
    A tool for summarizing text.
    """

    def use(self, task):
        try:
            # Use the BERT extractive summarizer to summarize the task
            summarizer = Summarizer()
            return summarizer(task, min_length=60, max_length=100)
        except Exception as e:
            raise RuntimeError(f"Failed to use SummarizerTool: {e}")
