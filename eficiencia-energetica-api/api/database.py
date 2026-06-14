"""
Armazenamento em memória das faturas.

Mantido deliberadamente simples (dicionário em memória) para que a banca execute
a API em qualquer máquina sem provisionar banco de dados. A separação em módulo
próprio permite, no futuro, trocar por SQLite/PostgreSQL sem alterar os routers.
"""

from threading import Lock

from models import FaturaArmazenada, FaturaEntrada


class RepositorioFaturas:
    def __init__(self) -> None:
        self._dados: dict[int, FaturaArmazenada] = {}
        self._proximo_id = 1
        self._lock = Lock()

    def adicionar(self, fatura: FaturaEntrada) -> FaturaArmazenada:
        with self._lock:
            registro = FaturaArmazenada(id=self._proximo_id, **fatura.model_dump())
            self._dados[self._proximo_id] = registro
            self._proximo_id += 1
            return registro

    def listar(self) -> list[FaturaArmazenada]:
        return sorted(self._dados.values(), key=lambda f: f.mes_ano)

    def obter_por_mes(self, mes_ano: str) -> FaturaArmazenada | None:
        for f in self._dados.values():
            if f.mes_ano == mes_ano:
                return f
        return None

    def limpar(self) -> None:
        with self._lock:
            self._dados.clear()
            self._proximo_id = 1


# Instância única compartilhada pela aplicação.
repositorio = RepositorioFaturas()
