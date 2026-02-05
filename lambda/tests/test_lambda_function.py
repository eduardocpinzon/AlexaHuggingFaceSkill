# -*- coding: utf-8 -*-
"""
Unit tests for lambda_function.py – Alexa Skill: Hugging Face Papers Summary.

Execução:
    pip install -r lambda/requirements-test.txt
    pytest lambda/tests/ -v
"""

import json
import unittest
from unittest.mock import patch, MagicMock, call

from lambda_function import (
    parse_paper_number,
    fetch_huggingface_papers,
    call_llm,
    summarize_papers_with_llm,
    get_paper_details_with_llm,
    NUMERO_MAP,
    LaunchRequestHandler,
    GetPapersSummaryIntentHandler,
    GetLatestNewsIntentHandler,
    GetPaperDetailsIntentHandler,
    HelpIntentHandler,
    CancelOrStopIntentHandler,
    FallbackIntentHandler,
    SessionEndedRequestHandler,
    CatchAllExceptionHandler,
)


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "title": "Paper One Title",
        "summary": "Summary of paper one about transformers.",
        "authors": ["Author A", "Author B"],
    },
    {
        "title": "Paper Two Title",
        "summary": "Summary of paper two about diffusion models.",
        "authors": ["Author C"],
    },
    {
        "title": "Paper Three Title",
        "summary": "Summary of paper three about reinforcement learning.",
        "authors": ["Author D", "Author E", "Author F"],
    },
]

# Resposta no formato da API pública do Hugging Face
SAMPLE_API_RESPONSE = [
    {
        "paper": {
            "title": p["title"],
            "summary": p["summary"],
            "authors": [{"name": n} for n in p["authors"]],
        }
    }
    for p in SAMPLE_PAPERS
]


def _mock_urlopen(data):
    """Retorna um mock compatível com context manager para urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_handler_input(intent_name=None, request_type=None, slots=None, session_attrs=None):
    """
    Cria um mock de HandlerInput do Alexa SDK.

    O response_builder é fluent: speak() e ask() retornam self,
    permitindo encadeamento como handler_input.response_builder.speak(...).ask(...).response
    """
    hi = MagicMock()

    # Response builder fluent ---------------------------------------------------
    rb = MagicMock()
    rb.speak.return_value = rb
    rb.ask.return_value = rb
    rb.response = MagicMock(name="Response")
    hi.response_builder = rb

    # Atributos de sessão -------------------------------------------------------
    hi.attributes_manager.session_attributes = session_attrs if session_attrs is not None else {}

    # Envelope de request -------------------------------------------------------
    if intent_name:
        hi.request_envelope.request.intent.name = intent_name
        hi.request_envelope.request.intent.slots = slots if slots is not None else {}
        hi.request_envelope.request.object_type = "IntentRequest"
    if request_type:
        hi.request_envelope.request.object_type = request_type

    return hi


def _slot(value):
    """Cria um mock de slot com o atributo .value."""
    s = MagicMock()
    s.value = value
    return s


# ===========================================================================
# 1. parse_paper_number
# ===========================================================================


class TestParsePaperNumber(unittest.TestCase):
    """Testes unitários para parse_paper_number()."""

    # --- Entradas None / vazias ------------------------------------------------

    def test_none_retorna_none(self):
        self.assertIsNone(parse_paper_number(None))

    def test_string_vazia_retorna_none(self):
        self.assertIsNone(parse_paper_number(""))

    def test_apenas_espacos_retorna_none(self):
        self.assertIsNone(parse_paper_number("   "))

    # --- Strings numéricas -----------------------------------------------------

    def test_string_numerica_valida(self):
        self.assertEqual(parse_paper_number("3"), 3)

    def test_string_numerica_com_espacos(self):
        self.assertEqual(parse_paper_number("  2  "), 2)

    def test_string_numerica_um(self):
        self.assertEqual(parse_paper_number("1"), 1)

    # --- Palavras em português (NUMERO_MAP) ------------------------------------

    def test_todas_as_palavras_mapeadas(self):
        """Cada chave de NUMERO_MAP deve resolver para o valor esperado."""
        for word, expected in NUMERO_MAP.items():
            with self.subTest(word=word):
                self.assertEqual(parse_paper_number(word), expected)

    def test_case_insensitive_primeiro(self):
        self.assertEqual(parse_paper_number("Primeiro"), 1)

    def test_case_insensitive_segundo_maiusculo(self):
        self.assertEqual(parse_paper_number("SEGUNDO"), 2)

    # --- Entradas inválidas ----------------------------------------------------

    def test_string_nao_numerica_retorna_none(self):
        self.assertIsNone(parse_paper_number("banana"))

    def test_float_string_retorna_none(self):
        self.assertIsNone(parse_paper_number("1.5"))

    def test_string_aleatoria_retorna_none(self):
        self.assertIsNone(parse_paper_number("xyz123"))


# ===========================================================================
# 2. fetch_huggingface_papers
# ===========================================================================


class TestFetchHuggingFacePapers(unittest.TestCase):
    """Testes unitários para fetch_huggingface_papers()."""

    @patch("lambda_function.urllib.request.urlopen")
    def test_retorna_papers_ate_o_limit(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(SAMPLE_API_RESPONSE).encode("utf-8")
        )

        papers = fetch_huggingface_papers(limit=2)

        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["title"], "Paper One Title")
        self.assertEqual(papers[1]["title"], "Paper Two Title")

    @patch("lambda_function.urllib.request.urlopen")
    def test_retorna_todos_quando_limit_excede_disponivel(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(SAMPLE_API_RESPONSE).encode("utf-8")
        )

        papers = fetch_huggingface_papers(limit=10)
        self.assertEqual(len(papers), 3)

    @patch("lambda_function.urllib.request.urlopen")
    def test_limit_zero_retorna_lista_vazia(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(SAMPLE_API_RESPONSE).encode("utf-8")
        )

        self.assertEqual(fetch_huggingface_papers(limit=0), [])

    @patch("lambda_function.urllib.request.urlopen")
    def test_resposta_api_vazia_retorna_lista_vazia(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(b"[]")

        self.assertEqual(fetch_huggingface_papers(), [])

    @patch("lambda_function.urllib.request.urlopen")
    def test_erro_de_rede_retorna_lista_vazia(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")

        self.assertEqual(fetch_huggingface_papers(), [])

    @patch("lambda_function.urllib.request.urlopen")
    def test_campos_ausentes_usam_valores_padrao(self, mock_urlopen):
        """Se title/summary/authors estiverem ausentes, defaults são usados."""
        api_data = [{"paper": {}}]
        mock_urlopen.return_value = _mock_urlopen(json.dumps(api_data).encode("utf-8"))

        papers = fetch_huggingface_papers(limit=1)

        self.assertEqual(papers[0]["title"], "Sem título")
        self.assertEqual(papers[0]["summary"], "")
        self.assertEqual(papers[0]["authors"], [])

    @patch("lambda_function.urllib.request.urlopen")
    def test_autores_limitados_a_cinco(self, mock_urlopen):
        authors = [{"name": f"Author {i}"} for i in range(8)]
        api_data = [{"paper": {"title": "T", "summary": "S", "authors": authors}}]
        mock_urlopen.return_value = _mock_urlopen(json.dumps(api_data).encode("utf-8"))

        papers = fetch_huggingface_papers(limit=1)
        self.assertEqual(len(papers[0]["authors"]), 5)

    @patch("lambda_function.urllib.request.urlopen")
    def test_extrai_dados_corretos_do_formato_api(self, mock_urlopen):
        """Verifica que title, summary e authors são extraídos da estrutura aninhada."""
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(SAMPLE_API_RESPONSE).encode("utf-8")
        )

        papers = fetch_huggingface_papers(limit=1)

        self.assertEqual(papers[0]["title"], "Paper One Title")
        self.assertEqual(papers[0]["summary"], "Summary of paper one about transformers.")
        self.assertEqual(papers[0]["authors"], ["Author A", "Author B"])


# ===========================================================================
# 3. call_llm
# ===========================================================================


class TestCallLlm(unittest.TestCase):
    """Testes unitários para call_llm()."""

    @patch("lambda_function.OPENAI_API_KEY", "")
    def test_chave_ausente_retorna_mensagem_de_erro(self):
        result = call_llm("any prompt")
        self.assertIn("OPENAI_API_KEY", result)

    @patch("lambda_function.OPENAI_API_KEY", "sk-test-key")
    @patch("lambda_function.urllib.request.urlopen")
    def test_chamada_bem_sucedida_retorna_conteudo(self, mock_urlopen):
        api_response = {"choices": [{"message": {"content": "Resumo gerado."}}]}
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(api_response).encode("utf-8")
        )

        result = call_llm("Resumir isto.")
        self.assertEqual(result, "Resumo gerado.")

    @patch("lambda_function.OPENAI_API_KEY", "sk-test-key")
    @patch("lambda_function.urllib.request.urlopen")
    def test_erro_na_api_retorna_mensagem_fallback(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")

        result = call_llm("prompt")
        self.assertIn("problema", result)

    @patch("lambda_function.OPENAI_API_KEY", "sk-test-key")
    @patch("lambda_function.urllib.request.urlopen")
    def test_payload_da_requisicao_correto(self, mock_urlopen):
        """Verifica que o body HTTP contém model, messages e max_tokens esperados."""
        api_response = {"choices": [{"message": {"content": "ok"}}]}
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(api_response).encode("utf-8")
        )

        call_llm("test prompt")

        # urlopen recebe o objeto Request real (não mockado)
        request_obj = mock_urlopen.call_args[0][0]
        body = json.loads(request_obj.data.decode("utf-8"))

        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(body["messages"][0]["content"], "test prompt")
        self.assertEqual(body["max_tokens"], 1024)

    @patch("lambda_function.OPENAI_API_KEY", "sk-test-key")
    @patch("lambda_function.urllib.request.urlopen")
    def test_header_authorization_contem_bearer(self, mock_urlopen):
        """Verifica que o Authorization header usa Bearer com a chave corrreta."""
        api_response = {"choices": [{"message": {"content": "ok"}}]}
        mock_urlopen.return_value = _mock_urlopen(
            json.dumps(api_response).encode("utf-8")
        )

        call_llm("prompt")

        request_obj = mock_urlopen.call_args[0][0]
        # urllib.request.Request normaliza headers com .capitalize()
        auth_header = request_obj.get_header("Authorization")
        self.assertEqual(auth_header, "Bearer sk-test-key")


# ===========================================================================
# 4. summarize_papers_with_llm
# ===========================================================================


class TestSummarizePapersWithLlm(unittest.TestCase):
    """Testes unitários para summarize_papers_with_llm()."""

    def test_lista_vazia_retorna_mensagem_nao_encontrou(self):
        result = summarize_papers_with_llm([])
        self.assertIn("Não encontrei", result)

    @patch("lambda_function.call_llm", return_value="Resumo mock.")
    def test_prompt_contem_titulos_dos_papers(self, mock_llm):
        summarize_papers_with_llm(SAMPLE_PAPERS)

        prompt_sent = mock_llm.call_args[0][0]
        for paper in SAMPLE_PAPERS:
            self.assertIn(paper["title"], prompt_sent)

    @patch("lambda_function.call_llm", return_value="Resumo mock.")
    def test_prompt_contem_autores(self, mock_llm):
        summarize_papers_with_llm(SAMPLE_PAPERS)

        prompt_sent = mock_llm.call_args[0][0]
        self.assertIn("Author A", prompt_sent)
        self.assertIn("Author C", prompt_sent)

    @patch("lambda_function.call_llm", return_value="Resumo mock.")
    def test_prompt_contem_resumos_dos_papers(self, mock_llm):
        summarize_papers_with_llm(SAMPLE_PAPERS)

        prompt_sent = mock_llm.call_args[0][0]
        for paper in SAMPLE_PAPERS:
            self.assertIn(paper["summary"][:500], prompt_sent)

    @patch("lambda_function.call_llm", return_value="Resumo retornado pelo LLM.")
    def test_retorna_output_do_llm(self, mock_llm):
        result = summarize_papers_with_llm(SAMPLE_PAPERS)
        self.assertEqual(result, "Resumo retornado pelo LLM.")

    @patch("lambda_function.call_llm", return_value="R")
    def test_call_llm_chamado_exatamente_uma_vez(self, mock_llm):
        summarize_papers_with_llm(SAMPLE_PAPERS)
        mock_llm.assert_called_once()


# ===========================================================================
# 5. get_paper_details_with_llm
# ===========================================================================


class TestGetPaperDetailsWithLlm(unittest.TestCase):
    """Testes unitários para get_paper_details_with_llm()."""

    @patch("lambda_function.call_llm", return_value="Detalhes mock.")
    def test_prompt_contem_titulo_resumo_e_autores(self, mock_llm):
        paper = SAMPLE_PAPERS[0]
        get_paper_details_with_llm(paper, paper_number=1)

        prompt_sent = mock_llm.call_args[0][0]
        self.assertIn(paper["title"], prompt_sent)
        self.assertIn(paper["summary"], prompt_sent)
        self.assertIn("Author A", prompt_sent)
        self.assertIn("Author B", prompt_sent)

    @patch("lambda_function.call_llm", return_value="Detalhes mock.")
    def test_prompt_contem_numero_do_paper(self, mock_llm):
        get_paper_details_with_llm(SAMPLE_PAPERS[1], paper_number=2)

        prompt_sent = mock_llm.call_args[0][0]
        self.assertIn("2", prompt_sent)

    @patch("lambda_function.call_llm", return_value="Detalhes retornados.")
    def test_retorna_output_do_llm(self, mock_llm):
        result = get_paper_details_with_llm(SAMPLE_PAPERS[2], paper_number=3)
        self.assertEqual(result, "Detalhes retornados.")


# ===========================================================================
# 6. LaunchRequestHandler
# ===========================================================================


class TestLaunchRequestHandler(unittest.TestCase):
    """Testes unitários para LaunchRequestHandler."""

    def setUp(self):
        self.handler = LaunchRequestHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle_delega_para_is_request_type(self, mock_utils):
        mock_utils.is_request_type.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_request_type.assert_called_with("LaunchRequest")

    @patch("lambda_function.ask_utils")
    def test_can_handle_retorna_false_para_outro_tipo(self, mock_utils):
        mock_utils.is_request_type.return_value = lambda hi: False

        self.assertFalse(self.handler.can_handle(MagicMock()))

    def test_handle_fala_saudacao_e_menciona_hugging_face(self):
        hi = _make_handler_input()

        self.handler.handle(hi)

        hi.response_builder.speak.assert_called_once()
        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("Hugging Face", speech)
        hi.response_builder.ask.assert_called_once()


# ===========================================================================
# 7. GetPapersSummaryIntentHandler
# ===========================================================================


class TestGetPapersSummaryIntentHandler(unittest.TestCase):
    """Testes unitários para GetPapersSummaryIntentHandler."""

    def setUp(self):
        self.handler = GetPapersSummaryIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_intent_name.assert_called_with("GetPapersSummaryIntent")

    @patch("lambda_function.summarize_papers_with_llm", return_value="Resumo.")
    @patch("lambda_function.fetch_huggingface_papers", return_value=SAMPLE_PAPERS)
    def test_handle_armazena_papers_na_sessao(self, mock_fetch, mock_summ):
        session = {}
        hi = _make_handler_input(
            intent_name="GetPapersSummaryIntent", session_attrs=session
        )

        self.handler.handle(hi)

        self.assertEqual(session["papers"], SAMPLE_PAPERS)

    @patch("lambda_function.summarize_papers_with_llm", return_value="Resumo do LLM.")
    @patch("lambda_function.fetch_huggingface_papers", return_value=SAMPLE_PAPERS)
    def test_handle_fala_o_resumo_gerado(self, mock_fetch, mock_summ):
        hi = _make_handler_input(intent_name="GetPapersSummaryIntent")

        self.handler.handle(hi)

        hi.response_builder.speak.assert_called_with("Resumo do LLM.")
        hi.response_builder.ask.assert_called_once()

    @patch("lambda_function.fetch_huggingface_papers", return_value=[])
    def test_handle_sem_papers_fala_mensagem_de_erro(self, mock_fetch):
        hi = _make_handler_input(intent_name="GetPapersSummaryIntent")

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("não consegui buscar", speech)
        # Não deve chamar .ask() quando erra (sessão finalizada)
        hi.response_builder.ask.assert_not_called()

    @patch("lambda_function.summarize_papers_with_llm", return_value="R")
    @patch("lambda_function.fetch_huggingface_papers", return_value=SAMPLE_PAPERS)
    def test_handle_chama_fetch_com_limit_4(self, mock_fetch, mock_summ):
        hi = _make_handler_input(intent_name="GetPapersSummaryIntent")

        self.handler.handle(hi)

        mock_fetch.assert_called_with(limit=4)


# ===========================================================================
# 8. GetLatestNewsIntentHandler
# ===========================================================================


class TestGetLatestNewsIntentHandler(unittest.TestCase):
    """Testes unitários para GetLatestNewsIntentHandler."""

    def setUp(self):
        self.handler = GetLatestNewsIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_intent_name.assert_called_with("GetLatestNewsIntent")

    @patch("lambda_function.summarize_papers_with_llm", return_value="News summary.")
    @patch("lambda_function.fetch_huggingface_papers", return_value=SAMPLE_PAPERS)
    def test_handle_armazena_papers_e_fala_resumo(self, mock_fetch, mock_summ):
        session = {}
        hi = _make_handler_input(
            intent_name="GetLatestNewsIntent", session_attrs=session
        )

        self.handler.handle(hi)

        self.assertEqual(session["papers"], SAMPLE_PAPERS)
        hi.response_builder.speak.assert_called_with("News summary.")

    @patch("lambda_function.fetch_huggingface_papers", return_value=[])
    def test_handle_sem_papers_fala_erro(self, mock_fetch):
        hi = _make_handler_input(intent_name="GetLatestNewsIntent")

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("não consegui", speech)
        hi.response_builder.ask.assert_not_called()

    @patch("lambda_function.summarize_papers_with_llm", return_value="R")
    @patch("lambda_function.fetch_huggingface_papers", return_value=SAMPLE_PAPERS)
    def test_handle_chama_fetch_com_limit_3(self, mock_fetch, mock_summ):
        hi = _make_handler_input(intent_name="GetLatestNewsIntent")

        self.handler.handle(hi)

        mock_fetch.assert_called_with(limit=3)


# ===========================================================================
# 9. GetPaperDetailsIntentHandler
# ===========================================================================


class TestGetPaperDetailsIntentHandler(unittest.TestCase):
    """Testes unitários para GetPaperDetailsIntentHandler."""

    def setUp(self):
        self.handler = GetPaperDetailsIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_intent_name.assert_called_with("GetPaperDetailsIntent")

    # --- Sem papers na sessão --------------------------------------------------

    def test_handle_sem_papers_na_sessao(self):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("1")},
            session_attrs={},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("Ainda não busquei", speech)
        hi.response_builder.ask.assert_called_once()

    # --- Número inválido / fora de intervalo -----------------------------------

    def test_handle_slot_valor_invalido(self):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("banana")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("número", speech)

    def test_handle_numero_zero(self):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("0")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("número", speech)

    def test_handle_numero_negativo(self):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("-1")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("número", speech)

    def test_handle_numero_excede_papers(self):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("99")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn(str(len(SAMPLE_PAPERS)), speech)

    def test_handle_slot_ausente(self):
        """Slots não contém paperNumber."""
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("número", speech)

    def test_handle_slot_valor_none(self):
        """Slot existe mas .value é None."""
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot(None)},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("número", speech)

    # --- Número válido ---------------------------------------------------------

    @patch("lambda_function.get_paper_details_with_llm", return_value="Detalhes artigo 2.")
    def test_handle_numero_valido_numerico(self, mock_details):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("2")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        mock_details.assert_called_with(SAMPLE_PAPERS[1], 2)
        hi.response_builder.speak.assert_called_with("Detalhes artigo 2.")
        hi.response_builder.ask.assert_called_once()

    @patch("lambda_function.get_paper_details_with_llm", return_value="Detalhes primeiro.")
    def test_handle_ordinal_portugues_primeiro(self, mock_details):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("primeiro")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        mock_details.assert_called_with(SAMPLE_PAPERS[0], 1)

    @patch("lambda_function.get_paper_details_with_llm", return_value="Detalhes terceiro.")
    def test_handle_ordinal_portugues_terceiro(self, mock_details):
        hi = _make_handler_input(
            intent_name="GetPaperDetailsIntent",
            slots={"paperNumber": _slot("terceiro")},
            session_attrs={"papers": SAMPLE_PAPERS},
        )

        self.handler.handle(hi)

        mock_details.assert_called_with(SAMPLE_PAPERS[2], 3)


# ===========================================================================
# 10. HelpIntentHandler
# ===========================================================================


class TestHelpIntentHandler(unittest.TestCase):
    """Testes unitários para HelpIntentHandler."""

    def setUp(self):
        self.handler = HelpIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_intent_name.assert_called_with("AMAZON.HelpIntent")

    def test_handle_fala_instrucoes_de_uso(self):
        hi = _make_handler_input()

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("resumir artigos", speech)
        self.assertIn("Hugging Face", speech)
        hi.response_builder.ask.assert_called_once()


# ===========================================================================
# 11. CancelOrStopIntentHandler
# ===========================================================================


class TestCancelOrStopIntentHandler(unittest.TestCase):
    """Testes unitários para CancelOrStopIntentHandler."""

    def setUp(self):
        self.handler = CancelOrStopIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle_quando_cancel_intent(self, mock_utils):
        """True quando CancelIntent case (short-circuit, StopIntent não avaliado)."""
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))

    @patch("lambda_function.ask_utils")
    def test_can_handle_quando_apenas_stop_intent(self, mock_utils):
        """True quando só StopIntent case — verifica a lógica OR."""
        mock_utils.is_intent_name.side_effect = [
            lambda hi: False,  # AMAZON.CancelIntent → False
            lambda hi: True,   # AMAZON.StopIntent  → True
        ]

        self.assertTrue(self.handler.can_handle(MagicMock()))

        # Verifica que ambos os intents foram consultados na ordem correta
        calls = [c[0][0] for c in mock_utils.is_intent_name.call_args_list]
        self.assertEqual(calls, ["AMAZON.CancelIntent", "AMAZON.StopIntent"])

    @patch("lambda_function.ask_utils")
    def test_can_handle_nenhum_intent_retorna_false(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: False

        self.assertFalse(self.handler.can_handle(MagicMock()))

    def test_handle_fala_despedida(self):
        hi = _make_handler_input()

        self.handler.handle(hi)

        hi.response_builder.speak.assert_called_with("Até mais!")
        # Não deve manter sessão aberta
        hi.response_builder.ask.assert_not_called()


# ===========================================================================
# 12. FallbackIntentHandler
# ===========================================================================


class TestFallbackIntentHandler(unittest.TestCase):
    """Testes unitários para FallbackIntentHandler."""

    def setUp(self):
        self.handler = FallbackIntentHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_intent_name.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_intent_name.assert_called_with("AMAZON.FallbackIntent")

    def test_handle_fala_mensagem_fallback(self):
        hi = _make_handler_input()

        self.handler.handle(hi)

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("Não entendi", speech)
        hi.response_builder.ask.assert_called_once()


# ===========================================================================
# 13. SessionEndedRequestHandler
# ===========================================================================


class TestSessionEndedRequestHandler(unittest.TestCase):
    """Testes unitários para SessionEndedRequestHandler."""

    def setUp(self):
        self.handler = SessionEndedRequestHandler()

    @patch("lambda_function.ask_utils")
    def test_can_handle(self, mock_utils):
        mock_utils.is_request_type.return_value = lambda hi: True

        self.assertTrue(self.handler.can_handle(MagicMock()))
        mock_utils.is_request_type.assert_called_with("SessionEndedRequest")

    def test_handle_retorna_response_sem_falar(self):
        hi = _make_handler_input()

        self.handler.handle(hi)

        # Não deve chamar speak nem ask
        hi.response_builder.speak.assert_not_called()
        hi.response_builder.ask.assert_not_called()


# ===========================================================================
# 14. CatchAllExceptionHandler
# ===========================================================================


class TestCatchAllExceptionHandler(unittest.TestCase):
    """Testes unitários para CatchAllExceptionHandler."""

    def setUp(self):
        self.handler = CatchAllExceptionHandler()

    def test_can_handle_sempre_retorna_true(self):
        self.assertTrue(self.handler.can_handle(MagicMock(), Exception("x")))
        self.assertTrue(self.handler.can_handle(MagicMock(), ValueError("y")))
        self.assertTrue(self.handler.can_handle(MagicMock(), RuntimeError("z")))

    def test_handle_fala_mensagem_de_erro_generica(self):
        hi = _make_handler_input()

        self.handler.handle(hi, RuntimeError("boom"))

        speech = hi.response_builder.speak.call_args[0][0]
        self.assertIn("erro", speech)
        hi.response_builder.ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
