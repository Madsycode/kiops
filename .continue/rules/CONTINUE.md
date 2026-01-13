# CONTINUE.md

## Project Overview

**Project Name:** MLOps Platform

**Purpose:** This project is a Streamlit-based platform for defining, building, and deploying machine learning applications. It allows users to specify an intent, automatically generate an application profile, containerize the application, and deploy it to a target environment (simulated in this example).

**Key Technologies:**

*   Python
*   Streamlit
*   Docker
*   Scikit-learn (for dummy models)
*   Graphviz (for visualization)
*   Ollama/Google Gemini (for AI-powered code generation)

**High-Level Architecture:**

The application follows a tabbed interface:

*   **Onboarding:**  Defines the app's intent and generates a RichMLAppProfile using an AI provider.
*   **Synthesis:**  Containerizes the application by generating a Dockerfile and training script, then builds and runs the container.
*   **Chats:** Provides a basic chat interface powered by the same AI provider.

---

## Getting Started

**Prerequisites:**

*   Python 3.11
*   Streamlit
*   Docker Desktop (or a Docker environment)
*   Ollama (optional, for local AI model) or a Google Gemini API Key
*   `pip install -r requirements.txt` (a requirements.txt file would need to be created, based on the imports)

**Installation:**

1.  Clone the repository.
2.  Create a `.env` file and add your API keys:
    ```
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    ```
3.  Install dependencies: `pip install -r requirements.txt`
4.  If using Ollama, ensure it's running locally: `ollama serve`

**Basic Usage:**

1.  Run the Streamlit app: `streamlit run service.py`
2.  Enter an intent in the "Onboarding" tab.
3.  Generate the app profile.
4.  Generate the training code and Dockerfile in the "Synthesis" tab.
5.  Build and run the training container.
6.  Promote the model to a deployment node (simulated).

**Running Tests:**

This project does not have any automated tests. Manual testing is performed through the Streamlit interface.

---

## Project Structure

*   `service.py`: The main Streamlit application file.
*   `models.py`: Defines the `RichMLAppProfile` data model.
*   `styles.py`: Contains custom CSS for styling the Streamlit app.
*   `containerize.py`:  Handles Docker image building and container execution.
*   `generative.py`: Contains functions for interacting with the AI provider (Google Gemini or Ollama).

---

## Development Workflow

**Coding Standards:**

*   Follow PEP 8 guidelines.
*   Use type hints where appropriate.
*   Write clear and concise code.

**Testing Approach:**

Currently, there are no automated tests. Manual testing is performed through the Streamlit interface.

**Build and Deployment Process:**

The application uses Docker for containerization. The `containerize.py` module handles building and running containers.  Deployment is currently simulated within the Streamlit app.

**Contribution Guidelines:**

*   Fork the repository.
*   Create a new branch for your changes.
*   Submit a pull request with a clear description of your changes.

---

## Key Concepts

*   **RichMLAppProfile:** A data model representing the characteristics of a machine learning application.
*   **DockerExecutionEngine:** A class responsible for building and running Docker containers.
*   **AI Provider:** The AI service used for code generation (e.g., Google Gemini, Ollama).

---

## Common Tasks

*   **Generating a new app profile:** Enter an intent in the "Onboarding" tab and click "Generate Profile."
*   **Deploying an app:** Generate the profile, then click "Promote Model to Deployment Node" in the "Synthesis" tab.

---

## Troubleshooting

*   **Docker not found:** Ensure Docker Desktop is running.
*   **API key errors:** Verify your API key is correct and properly configured in the `.env` file.
*   **Ollama issues:** Ensure Ollama is running locally and accessible.

---

## References

*   [Streamlit Documentation](https://docs.streamlit.io/)
*   [Docker Documentation](https://docs.docker.com/)
*   [Ollama Documentation](https://ollama.com/)
*   [Google Gemini API Documentation](https://ai.google.dev/gemini-api/)