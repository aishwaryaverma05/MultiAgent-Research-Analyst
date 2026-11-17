
"""HTTP client for communicating with the FastAPI backend."""

import requests

API_BASE_URL = "http://127.0.0.1:8000"


class ResearchAPIClient:
    """Client for research API operations."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def submit_research(self, query: str) -> dict:
        """Submit a new research request."""

        response = self.session.post(
            f"{self.base_url}/research/",
            json={"query": query},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_research_status(self, task_id: str) -> dict:
        """Fetch research task status and result."""

        response = self.session.get(
            f"{self.base_url}/research/{task_id}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def submit_approval(
        self,
        task_id: str,
        approved: bool,
        feedback: str = "",
    ) -> dict:
        """Submit approval or rejection for a research report."""

        response = self.session.post(
            f"{self.base_url}/research/{task_id}/approval",
            json={
                "approved": approved,
                "feedback": feedback,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
