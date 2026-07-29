# FAQ Chatbot Prototype — AeroBook Pro

## Prototype Overview
A Python Flask web app that matches user questions to pre-built FAQs using:
- **NLP Preprocessing**: NLTK (tokenization, stopword removal) + SpaCy (lemmatization)
- **Matching**: Scikit-learn TF-IDF + Cosine Similarity
- **UI**: Clean HTML/JS chat interface

---

## 🧠 Antigravity Multi-Prompt Strategy (How This Was Built)
> Do **NOT** build the entire product in one prompt. Break it down:

| Step | Prompt Focus | What to Ask Antigravity |
|---|---|---|
| 1 | Requirements & Design | "Create a requirements.md and design a Flask FAQ chatbot architecture with NLP preprocessing and cosine similarity matching." |
| 2 | Data Layer | "Generate faq_data.json with 10 realistic laptop FAQs for the product 'AeroBook Pro'." |
| 3 | NLP Backend | "Write app.py with SpaCy/NLTK preprocessing and scikit-learn TF-IDF cosine similarity. Add `/api/chat` endpoint." |
| 4 | UI/Frontend | "Create templates/index.html with a modern dark chat UI, message bubbles, and quick-suggestion buttons." |
| 5 | Deployment | "Add render.yaml for Render deploy, requirements.txt, and .github/workflows/deploy.yml." |
| 6 | Integration & Test | "Test locally with `python app.py`, then initialize git, commit, and prepare GitHub push instructions." |

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```
Visit: `http://localhost:5000`

---

## 🌐 Live Deployment (Render)
This repo includes `render.yaml`. To deploy:
1. Push this repo to GitHub
2. Import the repo into [Render](https://render.com)
3. Render will read `render.yaml` and deploy automatically

---

## 📁 File Structure
```
faq_chatbot/
├── app.py
├── faq_data.json
├── requirements.txt
├── render.yaml
├── .github/workflows/deploy.yml
├── templates/index.html
└── .gitignore
```

---

## 🔄 Next Prototype Iterations
- [ ] Add user session history
- [ ] Integrate intent classification (e.g., `sklearn` SVM or `transformers`)
- [ ] Add authentication / admin panel to edit FAQs
- [ ] Deploy live via Render and verify end-to-end
