# Alexa Skill: Resumo Hugging Face

Skill da Alexa que busca os artigos mais recentes do Hugging Face e resume em Portugues Brasileiro usando GPT-4o. Tudo roda no Alexa Developer Console, sem precisar de AWS.

## Como Funciona

```
Usuario fala -> Alexa reconhece -> Lambda busca papers no HuggingFace
                                          |
Alexa le <-- GPT-4o resume em PT-BR <-----
                                          |
               Usuario pede detalhes -> GPT-4o explica artigo especifico
```

1. A skill busca os papers mais recentes da API `daily_papers` do Hugging Face
2. Envia para GPT-4o com instrucoes para resumir em PT-BR
3. GPT-4o gera um resumo numerado dos artigos
4. Alexa le o resumo em voz alta
5. Usuario pode pedir detalhes de qualquer artigo pelo numero
6. GPT-4o gera explicacao detalhada do artigo escolhido

## Frases de Exemplo

| Frase | O que faz |
|-------|-----------|
| "Alexa, abrir resumo hugging face" | Inicia a skill |
| "resumir artigos" | Resume os ultimos papers |
| "quais sao as novidades" | Resume as novidades em IA |
| "detalhes do artigo 2" | Explica o segundo artigo em detalhes |
| "mais sobre o primeiro" | Detalha o primeiro artigo |
| "ajuda" | Explica como usar |
| "parar" | Encerra a skill |

## Estrutura do Projeto

```
AlexaHuggingFaceSkill/
├── lambda/
│   ├── lambda_function.py    # Codigo principal da skill
│   └── requirements.txt      # Dependencias Python (ask-sdk-core)
├── interactionModels/
│   └── custom/
│       └── pt-BR.json        # Modelo de interacao (portugues BR)
├── skill.json                # Manifesto da skill Alexa
├── deploy.sh                 # Script de deploy para AWS Lambda
├── SETUP.md                  # Guia passo a passo de configuracao
└── README.md                 # Este arquivo
```

## Tecnologias

- **Python 3** com [ASK SDK](https://developer.amazon.com/en-US/docs/alexa/alexa-skills-kit-sdk-for-python/overview.html) (ask-sdk-core)
- **Hugging Face API** — endpoint `daily_papers` (gratuito)
- **OpenAI API** — modelo GPT-4o para sumarizacao
- **Alexa Developer Console** — hospedagem e deploy

## Pre-requisitos

- Conta no [Alexa Developer Console](https://developer.amazon.com/alexa/console/ask)
- API Key do [OpenAI](https://platform.openai.com/api-keys) (comeca com `sk-`)

## Configuracao e Deploy

Consulte o guia completo em [SETUP.md](SETUP.md), que cobre:

1. Criacao da skill no Alexa Developer Console
2. Configuracao do Interaction Model (invocation name e intents)
3. Deploy do codigo Lambda
4. Configuracao da variavel de ambiente `OPENAI_API_KEY`
5. Teste da skill

### Deploy Alternativo (AWS Lambda)

Para deploy direto na AWS ao inves do Alexa-hosted, use o script:

```bash
./deploy.sh
```

Isso empacota o codigo e dependencias em `lambda_function.zip` para upload no AWS Lambda.

## Custos

| Servico | Custo |
|---------|-------|
| Alexa-hosted Lambda | **Gratuito** |
| Hugging Face API | **Gratuito** |
| GPT-4o | ~$0.005 por uso (~R$0.03) |

## Troubleshooting

| Problema | Solucao |
|----------|---------|
| "A chave da API do OpenAI nao esta configurada" | Verifique se adicionou `OPENAI_API_KEY` nas Environment Variables e faca Deploy novamente |
| Timeout ou erro de conexao | Alexa-hosted tem limite de 8s. Reduza `limit=4` para `limit=2` no codigo |
| "Skill request failed" | Veja logs em **Code** > **Logs** e verifique se o Build do modelo foi concluido |

## Licenca

Este projeto e disponibilizado como open source.
