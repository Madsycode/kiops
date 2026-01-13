import json
import re
from openai import OpenAI
import google.generativeai as genai
from models import RichMLAppProfile

def _clean_response(text):
    """Helper to strip markdown code blocks from LLM response."""
    # Remove ```json and ``` or just ```
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    return text.strip()

def query_ai_json(provider, api_key, base_url, model_name, prompt):
    """Generates the App Profile JSON."""
    try:
        system_prompt = (
            "You are an MLOps Architect. Return ONLY valid JSON adhering to the 'RichMLAppProfile' schema. "
            "Do not include markdown formatting, backticks, or explanations. Just the JSON object."
        )

        if provider == "Google Gemini":
            genai.configure(api_key=api_key)            
            model = genai.GenerativeModel(model_name,
                generation_config = {
                    "response_mime_type": "application/json", 
                    "response_schema": RichMLAppProfile
                }
            )
            response = model.generate_content(system_prompt + "\nUser Intent: " + prompt)
            return json.loads(_clean_response(response.text))
        
        else:
            client = OpenAI(base_url=base_url, api_key="ollama")
            schema = json.dumps(RichMLAppProfile.model_json_schema(), indent=2)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\nSchema: {schema}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = _clean_response(response.choices[0].message.content)
            return json.loads(content)

    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response. Try again."}
    except Exception as e:
        return {"error": str(e)}

def query_ai_text(provider, api_key, base_url, model_name, prompt):
    """Generates raw text (Code, Dockerfiles, Chat)."""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            res = genai.GenerativeModel(model_name).generate_content(prompt)
            return _clean_response(res.text) # Clean marks just in case
        else:
            client = OpenAI(base_url=base_url, api_key="ollama")
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return _clean_response(res.choices[0].message.content)
    except Exception as e:
        return f"Error: {str(e)}"