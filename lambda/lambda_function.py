# -*- coding: utf-8 -*-
"""
Alexa Skill: Hugging Face Papers Summary (Alexa-Hosted Version with LangChain)
Fetches latest ML papers from Hugging Face and summarizes in Brazilian Portuguese
using ChatGPT (GPT-4o) via LangChain.

Para usar no Alexa Developer Console (Alexa-hosted):
1. Crie uma skill Alexa-hosted (Python)
2. Cole este código no lambda_function.py
3. Adicione as dependências no requirements.txt
4. Configure OPENAI_API_KEY em Settings > Environment Variables
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional

import ask_sdk_core.utils as ask_utils
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import (
    AbstractRequestHandler,
    AbstractExceptionHandler
)
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variable - configure in Alexa Developer Console
# Settings > Environment Variables > Add OPENAI_API_KEY
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Initialize LangChain OpenAI model
llm = None
if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        max_tokens=1024,
        timeout=25,
    )

# Mapeamento de números por extenso para dígitos
NUMERO_MAP = {
    "um": 1, "uma": 1, "primeiro": 1, "primeira": 1,
    "dois": 2, "duas": 2, "segundo": 2, "segunda": 2,
    "três": 3, "tres": 3, "terceiro": 3, "terceira": 3,
    "quatro": 4, "quarto": 4, "quarta": 4,
    "cinco": 5, "quinto": 5, "quinta": 5,
}


def parse_paper_number(value: str) -> Optional[int]:
    """Parse paper number from slot value (handles words and digits)."""
    if not value:
        return None
    value = value.lower().strip()
    if value in NUMERO_MAP:
        return NUMERO_MAP[value]
    try:
        return int(value)
    except ValueError:
        return None


def fetch_huggingface_papers(limit: int = 5) -> list:
    """
    Fetch latest papers from Hugging Face Hub using their public API.
    """
    url = "https://huggingface.co/api/daily_papers"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AlexaSkill/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        result = []
        for paper in data[:limit]:
            paper_info = paper.get("paper", {})
            result.append({
                "title": paper_info.get("title", "Sem título"),
                "summary": paper_info.get("summary", ""),
                "authors": [a.get("name", "") for a in paper_info.get("authors", [])][:5],
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return []


def call_llm(prompt: str) -> str:
    """
    Call ChatGPT via LangChain.
    """
    if not llm:
        return "Erro: A chave da API do OpenAI não está configurada. Configure a variável OPENAI_API_KEY nas configurações da skill."

    try:
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"LangChain OpenAI error: {e}")
        return "Desculpe, tive um problema ao gerar o resumo."


def summarize_papers_with_llm(papers: list) -> str:
    """
    Use GPT-4o via LangChain to summarize papers in Brazilian Portuguese.
    """
    if not papers:
        return "Não encontrei artigos recentes para resumir. Tente novamente mais tarde."

    papers_text = ""
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper["authors"])
        papers_text += f"\nArtigo {i}: {paper['title']}\nAutores: {authors}\nResumo: {paper['summary'][:500]}\n"

    prompt = f"""Você é um assistente de voz da Alexa especializado em inteligência artificial.
Resuma os seguintes artigos científicos do Hugging Face de forma natural e conversacional em Português Brasileiro.

REGRAS IMPORTANTES:
- O resumo será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 150 palavras no total
- Use linguagem simples e acessível
- Não use siglas sem explicar
- Não use formatação como asteriscos ou marcadores
- NUMERE os artigos (primeiro, segundo, terceiro...) para que o usuário possa pedir detalhes
- Termine dizendo que o usuário pode pedir mais detalhes sobre qualquer artigo

{papers_text}

Gere um resumo natural e fluido em português brasileiro."""

    return call_llm(prompt)


def get_paper_details_with_llm(paper: dict, paper_number: int) -> str:
    """
    Use GPT-4o via LangChain to provide detailed explanation of a specific paper.
    """
    authors = ", ".join(paper["authors"])

    prompt = f"""Você é um assistente de voz da Alexa especializado em inteligência artificial.
Explique em detalhes o seguinte artigo científico em Português Brasileiro de forma natural e conversacional.

Título: {paper['title']}
Autores: {authors}
Resumo completo: {paper['summary']}

REGRAS IMPORTANTES:
- O texto será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 200 palavras
- Explique o que o artigo propõe e por que é importante
- Use linguagem acessível, explicando termos técnicos
- Não use formatação como asteriscos ou marcadores
- Comece dizendo "O artigo número {paper_number}..." ou similar

Gere uma explicação detalhada e natural em português brasileiro."""

    return call_llm(prompt)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = (
            "Olá! Sou sua assistente de artigos do Hugging Face. "
            "Diga resumir artigos para ouvir as novidades em inteligência artificial."
        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Diga resumir artigos para começar.")
                .response
        )


class GetPapersSummaryIntentHandler(AbstractRequestHandler):
    """Handler for Get Papers Summary Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetPapersSummaryIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Fetching and summarizing Hugging Face papers")

        papers = fetch_huggingface_papers(limit=4)

        if not papers:
            speak_output = (
                "Desculpe, não consegui buscar os artigos no momento. "
                "Por favor, tente novamente mais tarde."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .response
            )

        # Store papers in session for later reference
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["papers"] = papers

        speak_output = summarize_papers_with_llm(papers)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer mais detalhes sobre algum artigo? Diga o número.")
                .response
        )


class GetLatestNewsIntentHandler(AbstractRequestHandler):
    """Handler for getting latest AI news/papers."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetLatestNewsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Fetching latest AI news from Hugging Face")

        papers = fetch_huggingface_papers(limit=3)

        if not papers:
            speak_output = "Desculpe, não consegui buscar as novidades. Tente novamente."
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .response
            )

        # Store papers in session for later reference
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["papers"] = papers

        speak_output = summarize_papers_with_llm(papers)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer mais detalhes sobre algum artigo? Diga o número.")
                .response
        )


class GetPaperDetailsIntentHandler(AbstractRequestHandler):
    """Handler for getting details about a specific paper."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetPaperDetailsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Getting paper details")

        # Get paper number from slot
        slots = handler_input.request_envelope.request.intent.slots
        paper_number = None

        if slots and "paperNumber" in slots and slots["paperNumber"].value:
            paper_number = parse_paper_number(slots["paperNumber"].value)

        # Get papers from session
        session_attr = handler_input.attributes_manager.session_attributes
        papers = session_attr.get("papers", [])

        if not papers:
            speak_output = (
                "Ainda não busquei os artigos. "
                "Diga resumir artigos primeiro, e depois peça detalhes."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Diga resumir artigos para começar.")
                    .response
            )

        if not paper_number or paper_number < 1 or paper_number > len(papers):
            speak_output = (
                f"Por favor, diga um número de 1 a {len(papers)}. "
                f"Por exemplo, diga: detalhes do artigo 1."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Qual artigo você quer saber mais?")
                    .response
            )

        # Get the specific paper
        paper = papers[paper_number - 1]
        speak_output = get_paper_details_with_llm(paper, paper_number)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer saber sobre outro artigo?")
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = (
            "Eu resumo os artigos mais recentes de inteligência artificial do Hugging Face. "
            "Diga resumir artigos para ouvir as novidades. "
            "Depois, você pode pedir detalhes sobre um artigo específico dizendo, "
            "por exemplo, detalhes do artigo dois."
        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Handler for Cancel and Stop Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
            ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        return (
            handler_input.response_builder
                .speak("Até mais!")
                .response
        )


class FallbackIntentHandler(AbstractRequestHandler):
    """Handler for Fallback Intent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = "Não entendi. Diga resumir artigos ou quais são as novidades."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling."""

    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        return True

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        logger.error(exception, exc_info=True)

        return (
            handler_input.response_builder
                .speak("Desculpe, ocorreu um erro. Tente novamente.")
                .ask("Tente novamente.")
                .response
        )


# Skill Builder
sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GetPapersSummaryIntentHandler())
sb.add_request_handler(GetLatestNewsIntentHandler())
sb.add_request_handler(GetPaperDetailsIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
