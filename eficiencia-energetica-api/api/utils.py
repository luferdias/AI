"""
Utilitários transversais: logging, e ponte com o LLM (Groq).

Segue o padrão da disciplina (get_logger, execute_prompt via Groq), com um
acréscimo importante para reprodutibilidade pela banca: execute_prompt sinaliza
ausência de chave em vez de quebrar, permitindo o fallback determinístico.
"""

import logging
import os


def get_logger() -> logging.Logger:
    """Configura e retorna o logger da aplicação (nível INFO, com timestamp)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    return logging.getLogger("eficiencia-energetica")


class LLMIndisponivelError(RuntimeError):
    """Sinaliza que o LLM não pôde ser usado (sem chave ou erro de chamada)."""


def execute_prompt(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Envia um prompt ao LLM via Groq e retorna o texto gerado.

    Levanta LLMIndisponivelError se a chave GROQ_API_KEY não estiver
    configurada ou se a chamada falhar — o chamador decide o fallback.
    """
    logger = get_logger()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY ausente; LLM indisponível, acionando fallback.")
        raise LLMIndisponivelError("GROQ_API_KEY não configurada")

    try:
        from groq import Groq  # import tardio: dependência opcional
        client = Groq(api_key=api_key)
        resposta = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        conteudo = resposta.choices[0].message.content
        logger.info("Resposta do LLM gerada com sucesso.")
        return conteudo
    except LLMIndisponivelError:
        raise
    except Exception as exc:  # falha de rede, cota, etc.
        logger.error(f"Falha na chamada ao LLM: {exc}")
        raise LLMIndisponivelError(str(exc)) from exc
