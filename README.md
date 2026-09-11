# 🤖 Asistente Virtual RAG con Agente IA - Fundación Bienestar Corporativo (FBC)

Solución de asistencia virtual inteligente basada en arquitectura RAG (Retrieval-Augmented Generation) y Agente Inteligente con llamado a herramientas (Tool Calling) para gestionar consultas de beneficiarios de la Fundación Bienestar Corporativo.

---

## 🏗️ Arquitectura del Sistema

El sistema se compone de tres módulos principales:
1. **Agente IA (Qwen 2.5 3B via Ollama):** Razona sobre la consulta del usuario y decide si invocar herramientas externas.
2. **Herramientas de Consulta (`tools.py`):** Consulta la base de datos mock (`datos_usuarios.json`) para verificar estado de morosidad y grupo familiar mediante el RUT.
3. **Pipeline RAG (ChromaDB + HuggingFace Embeddings):** Recupera contexto normativo relevante desde `terminos_condiciones.txt` para fundamentar las respuestas.

---

## 📂 Estructura del Repositorio

```text
fbc-rag-agent/
│
├── data/
│   ├── datos_usuarios.json         # Base de datos de usuarios (RUT, estado, morosidad)
│   └── terminos_condiciones.txt    # Documento normativo para recuperación RAG
│
├── app.py                          # Interfaz de usuario en Streamlit y lógica del agente
├── tools.py                        # Herramienta 'consultar_estado_rut' aislada
├── test_agent.py                   # Pruebas unitarias automatizadas con Pytest
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Exclusión de archivos de caché y entorno
└── README.md                       # Documentación técnica

⚙️ Requisitos Previos
Python: 3.10 o superior (Compatibilidad probada en Python 3.13).

Ollama: Instalado y en ejecución.

Modelo LLM local: qwen2.5:3b.

Para descargar e iniciar el modelo en Ollama:

Bash
ollama pull qwen2.5:3b
🚀 Instalación y Ejecución
Clonar el repositorio:

Bash
git clone <URL_DE_TU_REPOSITORIO>
cd fbc-rag-agent
Crear y activar el entorno virtual:

PowerShell
python -m venv venv
.\venv\Scripts\activate
Instalar dependencias:

Bash
pip install -r requirements.txt
Ejecutar pruebas unitarias:

Bash
pytest test_agent.py
Iniciar la aplicación:

Bash
streamlit run app.py
🧪 Pruebas de Funcionamiento
El archivo test_agent.py verifica mediante pytest los tres escenarios de negocio requeridos:

Usuario con estado ACTIVO.

Usuario en condición de MOROSO / DADO DE BAJA.

RUT no registrado en el sistema (RUT_NO_ENCONTRADO).

📝 Declaración de Uso de Inteligencia Artificial
De acuerdo con las directrices académicas, se utilizó IA generativa como herramienta de apoyo para el refactorizado de código de compatibilidad en Python 3.13 y estructuración inicial de pruebas unitarias. Todas las reglas de negocio, prompts del sistema y diseño de arquitectura fueron validados e implementados por el equipo.