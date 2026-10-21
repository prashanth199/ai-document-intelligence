import os

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()


class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
        )

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "groq/openai/gpt-oss-20b"

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.model.with_structured_output(schema, method="json_schema").invoke(prompt)

        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self.model.with_structured_output(schema,method="json_schema").ainvoke(prompt)

        return (await self.model.ainvoke(prompt)).content

