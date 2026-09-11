import json
import os
from langchain_core.tools import tool

@tool
def consultar_estado_rut(rut: str) -> str:
    """Consulta el estado del usuario, morosidad y grupo familiar en la BD mediante el RUT."""
    rut_clean = rut.strip().replace(".", "").replace("-", "")
    json_path = os.path.join("data", "datos_usuarios.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            usuarios = json.load(f)
            for user in usuarios:
                user_rut_clean = user["rut"].replace(".", "").replace("-", "")
                if user_rut_clean == rut_clean:
                    return json.dumps(user, ensure_ascii=False)
    return "RUT_NO_ENCONTRADO"