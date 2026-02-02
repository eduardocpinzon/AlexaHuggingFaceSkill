# Alexa Skill: Resumo Hugging Face (100% no Console)

Skill da Alexa que busca os artigos mais recentes do Hugging Face e resume em Português Brasileiro usando Claude Sonnet. **Tudo roda no Alexa Developer Console, sem precisar de AWS.**

## Passo a Passo Completo

### 1. Criar a Skill no Alexa Developer Console

1. Acesse [developer.amazon.com/alexa/console/ask](https://developer.amazon.com/alexa/console/ask)
2. Clique em **Create Skill**
3. Configure:
   - **Skill name**: `Resumo Hugging Face`
   - **Primary locale**: `Portuguese (BR)`
   - **Type of experience**: Other
   - **Model**: Custom
   - **Hosting services**: **Alexa-hosted (Python)** ← IMPORTANTE!
   - **Hosting region**: US East (N. Virginia)
4. Clique em **Create Skill**
5. Escolha o template **Start from Scratch**

### 2. Configurar o Interaction Model

1. No menu lateral, vá em **Invocations** > **Skill Invocation Name**
2. Mude para: `resumo hugging face`
3. Salve

4. Vá em **Interaction Model** > **JSON Editor**
5. Cole o conteúdo abaixo e clique **Save Model**:

```json
{
  "interactionModel": {
    "languageModel": {
      "invocationName": "resumo hugging face",
      "intents": [
        {"name": "AMAZON.CancelIntent", "samples": []},
        {"name": "AMAZON.HelpIntent", "samples": []},
        {"name": "AMAZON.StopIntent", "samples": []},
        {"name": "AMAZON.NavigateHomeIntent", "samples": []},
        {"name": "AMAZON.FallbackIntent", "samples": []},
        {
          "name": "GetPapersSummaryIntent",
          "slots": [],
          "samples": [
            "resumir artigos",
            "resumir papers",
            "me conta sobre os artigos",
            "quero saber dos artigos",
            "artigos recentes",
            "últimos artigos",
            "últimos papers"
          ]
        },
        {
          "name": "GetLatestNewsIntent",
          "slots": [],
          "samples": [
            "quais são as novidades",
            "o que há de novo",
            "novidades",
            "notícias",
            "me atualiza",
            "o que está em alta",
            "destaques"
          ]
        }
      ],
      "types": []
    }
  }
}
```

6. Clique em **Build Model** (aguarde terminar)

### 3. Adicionar o Código

1. Vá em **Code** (menu superior)
2. No arquivo `lambda_function.py`, **apague tudo** e cole o conteúdo do arquivo `lambda/lambda_function.py` deste projeto
3. Clique em **Save**
4. Clique em **Deploy** (aguarde terminar)

### 4. Configurar a API Key do Anthropic

1. Ainda na aba **Code**, clique em **Settings** (ícone de engrenagem no canto inferior esquerdo)
2. Vá em **Environment Variables**
3. Clique em **Add Environment Variable**
4. Configure:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Sua API key do OpenAI (pegue em [platform.openai.com/api-keys](https://platform.openai.com/api-keys))
5. Clique em **Save**
6. Volte para o código e clique em **Deploy** novamente

### 5. Testar

1. Vá em **Test** (menu superior)
2. Mude **Skill testing is enabled in**: para **Development**
3. Digite ou fale: `abrir resumo hugging face`
4. Depois diga: `resumir artigos`

## Frases de Exemplo

| Frase | O que faz |
|-------|-----------|
| "Alexa, abrir resumo hugging face" | Inicia a skill |
| "resumir artigos" | Resume os últimos papers |
| "quais são as novidades" | Resume as novidades em IA |
| "detalhes do artigo 2" | Explica o segundo artigo em detalhes |
| "mais sobre o primeiro" | Detalha o primeiro artigo |
| "ajuda" | Explica como usar |
| "parar" | Encerra a skill |

## Como Funciona

```
Usuário fala → Alexa reconhece → Lambda busca papers no HuggingFace
                                         ↓
Alexa lê ← Claude Sonnet resume em PT-BR ←
                                         ↓
              Usuário pede detalhes → Claude explica artigo específico
```

1. A skill busca os papers mais recentes da API `daily_papers` do Hugging Face
2. Envia para Claude Sonnet com instruções para resumir em PT-BR
3. Claude gera um resumo numerado dos artigos
4. Alexa lê o resumo em voz alta
5. Usuário pode pedir detalhes de qualquer artigo pelo número
6. Claude gera explicação detalhada do artigo escolhido

## Obter API Key do OpenAI

1. Acesse [platform.openai.com](https://platform.openai.com)
2. Crie conta ou faça login
3. Vá em **API Keys** > **Create new secret key**
4. Copie a key (começa com `sk-`)

## Custos

| Serviço | Custo |
|---------|-------|
| Alexa-hosted Lambda | **Gratuito** |
| Hugging Face API | **Gratuito** |
| GPT-4o | ~$0.005 por uso (~R$0.03) |

## Troubleshooting

### "A chave da API do OpenAI não está configurada"
- Verifique se adicionou `OPENAI_API_KEY` nas Environment Variables
- Faça Deploy novamente após adicionar

### Timeout ou erro de conexão
- O Alexa-hosted tem limite de 8 segundos
- Se persistir, reduza o número de papers (mude `limit=4` para `limit=2`)

### "Skill request failed"
- Vá em **Code** > **Logs** para ver o erro detalhado
- Verifique se o Build do modelo foi concluído

## Estrutura do Projeto

```
AlexaHuggingFaceSkill/
├── lambda/
│   └── lambda_function.py    # Código principal (cole no console)
├── interactionModels/
│   └── custom/
│       └── pt-BR.json        # Modelo de interação completo
└── SETUP.md                  # Este guia
```
