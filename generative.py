import re
import json
from openai import OpenAI
import google.generativeai as genai
from ontology import RichMLAppProfile

def _clean_response(text):
    """Helper to strip markdown code blocks from LLM response."""
    text = re.sub(r'^```python\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    return text.strip()

def _unwrap_json(data):
    """
    Fixes common LLM issue where the JSON is wrapped in a root key 
    like {'RichMLAppProfile': {...}} or {'app': {...}}.
    """
    if isinstance(data, dict):
        # If it's a single key containing a dict, returns the inner dict
        # provided the inner dict looks like a profile (has 'name' and 'observables')
        if len(data) == 1:
            key = next(iter(data))
            val = data[key]
            if isinstance(val, dict) and "name" in val:
                return val
    return data

def query_ai_json(provider, api_key, base_url, model_name, prompt):
    """Generates the App Profile JSON with unpacking logic."""
    try:
        # Explicit instructions to prevent nesting
        system_prompt = (
            "RETURN ONLY! (NO extra chracters, comments, etc.) a valid JSON adhering STRICTLY to the 'RichMLAppProfile' schema. "
            "IMPORTANT: The output must be a FLAT JSON object. Do NOT wrap it in a root key like 'profile' or 'result'. "
            "ENSURE ALL fields are present and DO NOT wrap your output in markdown."
            "USE 'kiops-train-node' as container id for training."
        )

        content = ""
        
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)            
            model = genai.GenerativeModel(model_name,
                generation_config = {
                    "response_mime_type": "application/json", 
                    "response_schema": RichMLAppProfile
                }
            )
            response = model.generate_content(system_prompt + "\nUser Intent: " + prompt)
            content = response.text
        
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
            content = response.choices[0].message.content

        # Parse and Unwrap
        data = json.loads(_clean_response(content))
        data = _unwrap_json(data)
        
        return data

    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response. The model returned invalid JSON."}
    except Exception as e:
        return {"error": str(e)}

def query_ai_text(provider, api_key, base_url, model_name, prompt):
    """Generates raw text (Code, Dockerfiles, Chat)."""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            res = genai.GenerativeModel(model_name).generate_content(prompt)
            return _clean_response(res.text)
        else:
            client = OpenAI(base_url=base_url, api_key="ollama")
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return _clean_response(res.choices[0].message.content)
    except Exception as e:
        return f"Error: {str(e)}"