import re

import gradio as gr
import requests
import xmltodict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers.pipelines.question_answering import QuestionAnsweringPipeline

QA_MODEL_NAME = "ixa-ehu/SciBERT-SQuAD-QuAC"


def clean_text(text: str) -> str:
    text = re.sub("\n", " ", text)
    return text


def get_paper_summary(arxiv_id: str) -> str:
    paper_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(paper_url)
    paper_dict = xmltodict.parse(response.content)["feed"]["entry"]
    return clean_text(paper_dict["summary"])


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
    gr.Markdown("# Document QA")

    # Retrieve paper
    arxiv_id = gr.Textbox(
        label="arXiv Paper ID", placeholder="Insert here the ID of a paper on arXiv"
    )
    paper_summary = gr.Textbox(label="Paper summary")
    fetch_document_button = gr.Button("Get Summary")
    fetch_document_button.click(
        fn=get_paper_summary, inputs=arxiv_id, outputs=paper_summary
    )

    # QA on paper
    question = gr.Textbox(label="Ask a question about the paper:")
    answer = gr.Textbox("Answer:")
    ask_button = gr.Button("Ask me 🤖")
    ask_button.click(fn=get_answer, inputs=[question, paper_summary], outputs=answer)


demo.launch()
