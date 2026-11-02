# Multi-Agent Research Analyst

An AI app where a small team of agents researches any topic for you. One agent searches the web, one reads the best page, one writes a report, and one reviews that report. You see everything in a clean Streamlit web app.

---

## 60-Second Summary

- **What it does:** You type a topic. You get a written research report, a review of that report, and a list of sources.
- **How it works:** Four steps run one after another: **Search, Read, Write, Critique**.
- **What it is built with:** Python, Streamlit, LangChain and LangGraph.
- **How to run it:** Install the requirements, add your API keys in a `.env` file, then run `streamlit run app.py`.
- **Main idea:** Instead of asking one AI to do everything, each agent does one job. This gives better and more reliable results.

---

## How It Works

```
Topic  ->  Search Agent  ->  Reader Agent  ->  Writer  ->  Critic  ->  Final Output
```

| Step | Agent | What it does |
|---|---|---|
| 1 | **Search Agent** | Looks on the web for recent and reliable information about your topic. |
| 2 | **Reader Agent** | Picks the most useful link from the search results and reads the full page. |
| 3 | **Writer Chain** | Combines the search results and the page content into a structured research report. |
| 4 | **Critic Chain** | Reads the report and gives feedback on quality, gaps and improvements. |

The output of each step becomes the input of the next step. All of this is controlled by `pipeline.py`.

---

## Features

- Four specialised agents, each with one clear job
- Live progress view: you can see which agent is working right now
- Final report, critic feedback, sources and raw research in separate tabs
- Download the report and the feedback as Markdown files
- Light mode and dark mode
- History of recent runs in the sidebar

---

## Tech Stack

| Part | Tool |
|---|---|
| Language | Python |
| Agents | LangChain and LangGraph |
| Web app | Streamlit |
| Secrets | `python-dotenv` (`.env` file) |

---

## Project Structure

```
MultiAgent-Research-Analyst/
├── app.py               # Streamlit web app (the user interface)
├── pipeline.py          # Runs the 4 steps in order and returns the results
├── agents.py            # Defines the search agent, reader agent, writer and critic
├── tools.py             # Tools the agents use (for example web search and page scraping)
├── requirements.txt     # Python packages needed
├── .streamlit/
│   └── config.toml      # Streamlit theme settings
└── .gitignore           # Keeps secrets and junk files out of GitHub
```

**Which file does what, in one line each:**
- `app.py` is what the user sees and clicks.
- `pipeline.py` is the brain that decides the order of the steps.
- `agents.py` and `tools.py` are the workers and the tools they use.

---

## Run It on Your Computer

**1. Clone the project**

```bash
git clone https://github.com/aishwaryaverma05/MultiAgent-Research-Analyst.git
cd MultiAgent-Research-Analyst
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

**3. Install the packages**

```bash
pip install -r requirements.txt
```

**4. Add your API keys**

Create a file named `.env` in the main folder and add your keys. Use the same variable names that your code reads in `agents.py` and `tools.py`. For example:

```
LLM_API_KEY=your_key_here
SEARCH_API_KEY=your_key_here
```

Never upload this file to GitHub. It is already listed in `.gitignore`.

**5. Start the app**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

You can also run only the pipeline, without the web app:

```bash
python pipeline.py
```

---

## How to Use the App

1. Type a topic in the box. Example: *Impact of generative AI on healthcare diagnostics*
2. Click **Run research**.
3. Watch the four steps change from Pending to In progress to Completed.
4. Read the result in the tabs: **Report**, **Critic feedback**, **Sources** and **Raw research**.
5. Download the report if you want to keep it.

---

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and choose this repository, the `main` branch, and `app.py` as the main file.
4. Open **Advanced settings**, choose Python 3.12 or 3.13, and paste your keys in **Secrets** like this:

```toml
LLM_API_KEY = "your_key_here"
SEARCH_API_KEY = "your_key_here"
```

5. Click **Deploy**.

---

## Good to Know

- The report is only as good as the sources the search agent finds. Always check important facts yourself.
- Each run uses your API keys, so each run can cost a small amount of money or use up free credits.
- A full run can take a few minutes because four steps run one after another.
- The reader agent reads one page per run. Reading more pages is a possible improvement.

---

## Future Ideas

- Read more than one web page per run
- Let the writer fix the report using the critic's feedback (a revision loop)
- Export the report as PDF
- Add a choice of report style (short summary, detailed report, bullet points)

---

## Author

Built by **Aishwarya Verma**.
GitHub: [@aishwaryaverma05](https://github.com/aishwaryaverma05)
