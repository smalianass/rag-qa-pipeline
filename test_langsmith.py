
from dotenv import load_dotenv
from langsmith import traceable, Client

load_dotenv()

@traceable(
    name="test-connexion-langsmith",
    project_name="rag-docker-local",
)
def test_connection(question: str) -> str:
    return f"LangSmith fonctionne. Question test : {question}"

result = test_connection("Qu'est-ce que Docker ?")
print(result)

# Attend l'envoi des traces avant la fin du script.
Client().flush()
print("Test terminé. Vérifie le projet dans LangSmith.")
