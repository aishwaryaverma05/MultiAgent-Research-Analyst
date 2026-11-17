
from unittest.mock import patch

from evaluation.ragas_models import create_ragas_models


def test_create_ragas_models():
    with (
        patch("evaluation.ragas_models.create_chat_model") as chat_model,
        patch(
            "evaluation.ragas_models.create_embeddings_model"
        ) as embeddings_model,
        patch(
            "evaluation.ragas_models.LangchainLLMWrapper"
        ) as llm_wrapper,
        patch(
            "evaluation.ragas_models.LangchainEmbeddingsWrapper"
        ) as embeddings_wrapper,
    ):
        llm, embeddings = create_ragas_models()

    chat_model.assert_called_once()
    embeddings_model.assert_called_once()

    llm_wrapper.assert_called_once_with(chat_model.return_value)
    embeddings_wrapper.assert_called_once_with(
        embeddings_model.return_value
    )

    assert llm == llm_wrapper.return_value
    assert embeddings == embeddings_wrapper.return_value
