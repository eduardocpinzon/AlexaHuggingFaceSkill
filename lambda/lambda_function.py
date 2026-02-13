"""
Alexa Skill: Hugging Face Papers Summary (Alexa-Hosted Version)
Fetches latest ML papers from Hugging Face and summarizes in Brazilian Portuguese
using ChatGPT (GPT-4o) via direct OpenAI API calls.

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variable - configure in Alexa Developer Console
# Settings > Environment Variables > Add OPENAI_API_KEY
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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
    Call ChatGPT via direct HTTP request to OpenAI API.
    """
    if not OPENAI_API_KEY:
        return "Erro: A chave da API do OpenAI não está configurada. Configure a variável OPENAI_API_KEY nas configurações da skill."

    try:
        payload = json.dumps({
            "model": "gpt-5.2-2025-12-11",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1024,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )

        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "Desculpe, tive um problema ao gerar o resumo."


def summarize_papers_with_llm(papers: list) -> str:
    """
    Use GPT-4o to summarize papers in Brazilian Portuguese.
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
- Use no máximo 200 palavras no total
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
    Use GPT-4o to provide detailed explanation of a specific paper.
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
- Comece dizendo "O artigo número {paper_number} de titulo {paper['title']}..." ou similar

Gere uma explicação detalhada e natural em português brasileiro."""

    return call_llm(prompt)


def get_practical_usage_with_llm(paper: dict, paper_number: int) -> str:
    """
    Use LLM to generate practical usage ideas for a specific paper.
    """
    authors = ", ".join(paper["authors"])

    prompt = f"""Você é um assistente de voz da Alexa especializado em inteligência artificial.
Com base no seguinte artigo científico, sugira ideias práticas de uso e aplicações reais da proposta apresentada.

Título: {paper['title']}
Autores: {authors}
Resumo completo: {paper['summary']}

REGRAS IMPORTANTES:
- O texto será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 250 palavras
- Sugira de 3 a 5 aplicações práticas e reais
- Explique como empresas, desenvolvedores ou pesquisadores poderiam usar essa tecnologia
- Dê exemplos concretos de setores ou problemas que poderiam se beneficiar
- Use linguagem acessível e conversacional
- Não use formatação como asteriscos ou marcadores
- Comece dizendo "O artigo número {paper_number}, sobre {paper['title']}, pode ser aplicado de várias formas..." ou similar

Gere sugestões práticas e criativas em português brasileiro."""

    return call_llm(prompt)


def compare_papers_with_llm(paper1: dict, num1: int, paper2: dict, num2: int) -> str:
    """
    Use LLM to compare two papers.
    """
    authors1 = ", ".join(paper1["authors"])
    authors2 = ", ".join(paper2["authors"])

    prompt = f"""Você é um assistente de voz da Alexa especializado em inteligência artificial.
Compare os dois artigos científicos abaixo, destacando semelhanças, diferenças e como se complementam.

Artigo {num1}: {paper1['title']}
Autores: {authors1}
Resumo: {paper1['summary'][:500]}

Artigo {num2}: {paper2['title']}
Autores: {authors2}
Resumo: {paper2['summary'][:500]}

REGRAS IMPORTANTES:
- O texto será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 250 palavras
- Compare os objetivos, métodos e contribuições de cada artigo
- Destaque o que há em comum e o que difere
- Mencione se os artigos se complementam
- Use linguagem acessível e conversacional
- Não use formatação como asteriscos ou marcadores

Gere uma comparação natural e fluida em português brasileiro."""

    return call_llm(prompt)


def get_simplified_explanation_with_llm(paper: dict, paper_number: int) -> str:
    """
    Use LLM to generate a simplified (ELI5) explanation of a paper.
    """
    authors = ", ".join(paper["authors"])

    prompt = f"""Você é um assistente de voz da Alexa especializado em explicar ciência de forma simples.
Explique o seguinte artigo científico como se estivesse explicando para alguém que não tem nenhum conhecimento técnico.

Título: {paper['title']}
Autores: {authors}
Resumo completo: {paper['summary']}

REGRAS IMPORTANTES:
- O texto será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 200 palavras
- Explique como se fosse para uma criança de 10 anos ou alguém completamente leigo
- Use analogias do dia a dia para explicar conceitos técnicos
- Evite completamente jargão técnico, use palavras simples
- Não use formatação como asteriscos ou marcadores
- Comece dizendo "Imagine que..." ou "Pense assim..." ou algo que crie uma analogia fácil

Gere uma explicação extremamente simples e acessível em português brasileiro."""

    return call_llm(prompt)


def get_key_findings_with_llm(paper: dict, paper_number: int) -> str:
    """
    Use LLM to extract key findings and contributions from a paper.
    """
    authors = ", ".join(paper["authors"])

    prompt = f"""Você é um assistente de voz da Alexa especializado em inteligência artificial.
Extraia e apresente as principais descobertas e contribuições do seguinte artigo científico.

Título: {paper['title']}
Autores: {authors}
Resumo completo: {paper['summary']}

REGRAS IMPORTANTES:
- O texto será LIDO EM VOZ ALTA pela Alexa
- Use no máximo 200 palavras
- Foque nos resultados principais e nas contribuições mais importantes
- Mencione métricas ou melhorias quantitativas se disponíveis
- Explique por que essas descobertas são relevantes para a área
- Use linguagem acessível, explicando termos técnicos
- Não use formatação como asteriscos ou marcadores
- Comece dizendo "As principais descobertas do artigo número {paper_number}..." ou similar

Gere um resumo das descobertas chave em português brasileiro."""

    return call_llm(prompt)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        speak_output = (
            "Olá! Sou sua assistente de artigos do Hugging Face. "
            "Diga resumir artigos para ouvir as novidades em inteligência artificial. "
            "Depois, você pode pedir detalhes, usos práticos, explicação simples, "
            "descobertas ou comparar artigos."
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


class GetPracticalUsageIntentHandler(AbstractRequestHandler):
    """Handler for getting practical usage ideas for a paper."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetPracticalUsageIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Getting practical usage ideas for a paper")

        slots = handler_input.request_envelope.request.intent.slots
        paper_number = None

        if slots and "paperNumber" in slots and slots["paperNumber"].value:
            paper_number = parse_paper_number(slots["paperNumber"].value)

        session_attr = handler_input.attributes_manager.session_attributes
        papers = session_attr.get("papers", [])

        if not papers:
            speak_output = (
                "Ainda não busquei os artigos. "
                "Diga resumir artigos primeiro, e depois peça os usos práticos."
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
                f"Por exemplo, diga: usos práticos do artigo 1."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("De qual artigo você quer saber os usos práticos?")
                    .response
            )

        paper = papers[paper_number - 1]
        speak_output = get_practical_usage_with_llm(paper, paper_number)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer saber os usos práticos de outro artigo?")
                .response
        )


class ComparePapersIntentHandler(AbstractRequestHandler):
    """Handler for comparing two papers."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("ComparePapersIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Comparing two papers")

        slots = handler_input.request_envelope.request.intent.slots
        first_number = None
        second_number = None

        if slots:
            if "firstPaper" in slots and slots["firstPaper"].value:
                first_number = parse_paper_number(slots["firstPaper"].value)
            if "secondPaper" in slots and slots["secondPaper"].value:
                second_number = parse_paper_number(slots["secondPaper"].value)

        session_attr = handler_input.attributes_manager.session_attributes
        papers = session_attr.get("papers", [])

        if not papers:
            speak_output = (
                "Ainda não busquei os artigos. "
                "Diga resumir artigos primeiro, e depois peça para comparar."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Diga resumir artigos para começar.")
                    .response
            )

        if (not first_number or not second_number or
                first_number < 1 or first_number > len(papers) or
                second_number < 1 or second_number > len(papers)):
            speak_output = (
                f"Por favor, diga dois números de 1 a {len(papers)}. "
                f"Por exemplo, diga: comparar artigo 1 com artigo 2."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Quais dois artigos você quer comparar?")
                    .response
            )

        if first_number == second_number:
            speak_output = "Você precisa escolher dois artigos diferentes para comparar."
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Quais dois artigos você quer comparar?")
                    .response
            )

        paper1 = papers[first_number - 1]
        paper2 = papers[second_number - 1]
        speak_output = compare_papers_with_llm(paper1, first_number, paper2, second_number)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer comparar outros artigos?")
                .response
        )


class GetSimplifiedExplanationIntentHandler(AbstractRequestHandler):
    """Handler for getting a simplified (ELI5) explanation of a paper."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetSimplifiedExplanationIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Getting simplified explanation for a paper")

        slots = handler_input.request_envelope.request.intent.slots
        paper_number = None

        if slots and "paperNumber" in slots and slots["paperNumber"].value:
            paper_number = parse_paper_number(slots["paperNumber"].value)

        session_attr = handler_input.attributes_manager.session_attributes
        papers = session_attr.get("papers", [])

        if not papers:
            speak_output = (
                "Ainda não busquei os artigos. "
                "Diga resumir artigos primeiro, e depois peça uma explicação simples."
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
                f"Por exemplo, diga: explica de forma simples o artigo 1."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Qual artigo você quer que eu explique de forma simples?")
                    .response
            )

        paper = papers[paper_number - 1]
        speak_output = get_simplified_explanation_with_llm(paper, paper_number)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer uma explicação simples de outro artigo?")
                .response
        )


class GetKeyFindingsIntentHandler(AbstractRequestHandler):
    """Handler for getting key findings and contributions of a paper."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return ask_utils.is_intent_name("GetKeyFindingsIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Getting key findings for a paper")

        slots = handler_input.request_envelope.request.intent.slots
        paper_number = None

        if slots and "paperNumber" in slots and slots["paperNumber"].value:
            paper_number = parse_paper_number(slots["paperNumber"].value)

        session_attr = handler_input.attributes_manager.session_attributes
        papers = session_attr.get("papers", [])

        if not papers:
            speak_output = (
                "Ainda não busquei os artigos. "
                "Diga resumir artigos primeiro, e depois peça as descobertas."
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
                f"Por exemplo, diga: descobertas do artigo 1."
            )
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("De qual artigo você quer saber as descobertas?")
                    .response
            )

        paper = papers[paper_number - 1]
        speak_output = get_key_findings_with_llm(paper, paper_number)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Quer saber as descobertas de outro artigo?")
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
            "Depois, você pode pedir detalhes, usos práticos, uma explicação simples, "
            "ou as principais descobertas de um artigo. "
            "Você também pode comparar dois artigos entre si. "
            "Por exemplo, diga: usos práticos do artigo dois."
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
        speak_output = (
            "Não entendi. Diga resumir artigos para começar, "
            "ou peça usos práticos, explicação simples, descobertas ou comparar artigos."
        )

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
sb.add_request_handler(GetPracticalUsageIntentHandler())
sb.add_request_handler(ComparePapersIntentHandler())
sb.add_request_handler(GetSimplifiedExplanationIntentHandler())
sb.add_request_handler(GetKeyFindingsIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
