from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

import gradio as gr
import requests
import xmltodict
from PyPDF2 import PdfReader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers.pipelines.question_answering import QuestionAnsweringPipeline

QA_MODEL_NAME = "ixa-ehu/SciBERT-SQuAD-QuAC"
TEMP_PDF_PATH = "/tmp/arxiv_paper.pdf"
ARXIV_URL_PATTERN = r"(http|https)://(arxiv.org/pdf/)+([0-9]+\.[0-9]+)\.pdf"


def is_valid_url(url: str) -> bool:
    return re.fullmatch(ARXIV_URL_PATTERN, url) is not None


@dataclass
class PaperMetaData:
    arxiv_id: str
    title: str
    summary: str
    text: str

    @staticmethod
    def _clean_field(text: str) -> str:
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def from_api(cls, arxiv_id: str, text: str) -> PaperMetaData:
        paper_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(paper_url)
        paper_dict = xmltodict.parse(response.content)["feed"]["entry"]
        return PaperMetaData(
            arxiv_id=arxiv_id,
            title=cls._clean_field(paper_dict["title"]),
            summary=cls._clean_field(paper_dict["summary"]),
            text=text,
        )


def clean_text(text: str) -> str:
    text = re.sub(r"\x03|\x02", "", text)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\n", " ", text)
    return text


class PDFPaper:
    def __init__(self, url: str):
        if not is_valid_url(url):
            raise ValueError("The URL provided is not a valid arxiv PDF url.")
        self.url = url
        self.arxiv_id = re.fullmatch(ARXIV_URL_PATTERN, url).group(3)

    def _download(self, download_path: str = TEMP_PDF_PATH) -> None:
        pdf_r = requests.get(self.url)
        pdf_r.raise_for_status()
        with open(download_path, "wb") as pdf_file:
            pdf_file.write(pdf_r.content)

    def read_text(self, pdf_path: str = TEMP_PDF_PATH) -> str:
        self._download(pdf_path)
        reader = PdfReader(pdf_path)
        pdf_text = " ".join([page.extract_text() for page in reader.pages])
        return clean_text(pdf_text)

    def get_paper_full_data(self) -> PaperMetaData:
        return PaperMetaData.from_api(arxiv_id=self.arxiv_id, text=self.read_text())


def get_paper_data(url: str) -> Tuple[str, str, str]:
    paper_data = PDFPaper(url=url).get_paper_full_data()
    return paper_data.title, paper_data.summary, paper_data.text


def get_qa_pipeline(qa_model_name: str = QA_MODEL_NAME) -> QuestionAnsweringPipeline:
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline


def get_answer(question: str, context: str) -> str:
    qa_pipeline = get_qa_pipeline()
    prediction = qa_pipeline(question=question, context=context)
    return prediction["answer"]


demo = gr.Blocks()

with demo:
    gr.Markdown("# arXiv Paper Q&A\nImport an arXiv paper and ask questions about it!")

    gr.Markdown("## 📄 Import the paper on arXiv")
    arxiv_url = gr.Textbox(
        label="arXiv Paper URL", placeholder="Insert here the URL of a paper on arXiv"
    )
    fetch_document_button = gr.Button("Import Paper")
    paper_title = gr.Textbox(label="Paper Title")
    paper_summary = gr.Textbox(label="Paper Summary")
    paper_text = gr.Textbox(label="Paper Text")
    fetch_document_button.click(
        fn=get_paper_data,
        inputs=arxiv_url,
        outputs=[paper_title, paper_summary, paper_text],
    )

    gr.Markdown("## 🤨 Ask a question about the paper")
    question = gr.Textbox(label="Ask a question about the paper:")
    ask_button = gr.Button("Ask me 🤖")
    answer = gr.Textbox(label="Answer:")
    ask_button.click(fn=get_answer, inputs=[question, paper_summary], outputs=answer)


demo.launch()
