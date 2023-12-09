from typing import Any

from langchain.tools import BaseTool
import retirevalqa
class PdfKnowledge(BaseTool):
    name = "My Pdf Knowledge"
    description = ("use this tool when you need to get the answer from the question. "
                   "This is the first tool before using any other tool")


    def _run(
        self,
        question: str
    ) -> Any:
        print("My Pdf Knowledge tools")
        qa, memory = retirevalqa.qa_memory()
        result = qa(question)
        return result["answer"]

    def _arun(
        self,
    ) -> Any:
        raise('This tool is not implemented async')