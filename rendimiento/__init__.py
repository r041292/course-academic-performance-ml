"""Predicción del rendimiento académico por asignatura (corrida final de la tesis)."""

from .config import Configuracion
from .corrida import Corrida
from .utilidades import configurar_registro

__all__ = ["Configuracion", "Corrida", "configurar_registro"]
