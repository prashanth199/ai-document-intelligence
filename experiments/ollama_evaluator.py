from deepeval.models import DeepEvalBaseLLM

from langchain_ollama import ChatOllama

class OllamaEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOllama(
            model="qwen3:latest",
            temperature=0
        )

    def get_model_name(self):
        return "qwen3:latest"

    def generate(self, prompt: str, schema=None):

        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return structured_model.invoke(prompt)
        
        response = self.model.invoke(prompt)
        return response.content
    
    def load_model(self):
        return self.model

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return await structured_model.ainvoke(prompt)

        response = await self.model.ainvoke(prompt)
        return response.content



# evaluator = OllamaEvaluator()

# result = evaluator.generate("What is 2 + 2? Reply with only the number.")

# print(result)

# from pydantic import BaseModel

# class MathAnswer(BaseModel):
#     answer: int

# evaluator = OllamaEvaluator()

# result = evaluator.generate(
#     "What is 2 + 2?",
#     schema=MathAnswer
# )

# print(result)
# print(type(result))

# import asyncio

# async def main():
#     evaluator = OllamaEvaluator()

#     result = await evaluator.a_generate(
#         "What is 5 + 5?",
#         schema=MathAnswer
#     )

#     print(result)
#     print(type(result))

# asyncio.run(main())