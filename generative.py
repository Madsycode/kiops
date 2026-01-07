import json
from openai import OpenAI
import google.generativeai as genai
from models import RichMLAppProfile

def query_ai_json(provider, api_key, base_url, model_name, prompt):
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)            
            model = genai.GenerativeModel(model_name,
                generation_config={
                    "response_mime_type": "application/json", 
                    "response_schema": RichMLAppProfile
                }
            )

            full_prompt = (
                "You are the KI Ops Executive Onboarding Agent for a 6G Network. "
                "Translate the following user intent into a formal Rich-ML-App Descriptor.\n\n"
                f"User Intent: {prompt}"
            )

            response = model.generate_content(full_prompt)
            return json.loads(response.text)
        else:
            client = OpenAI(base_url=base_url, api_key="ollama")
            schema = json.dumps(RichMLAppProfile.model_json_schema(), indent=2)
            system = f"Return ONLY valid JSON (NO additional comments, characters, descriptors, etc.) strictly adhering to: {schema}"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        return f"{str(e)}"

def query_ai_text(provider, api_key, base_url, model_name, prompt):
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(model_name).generate_content(prompt).text
        else:
            client = OpenAI(base_url=base_url, api_key="ollama")
            return client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content.strip()
    except Exception as e:
        return f"{str(e)}"