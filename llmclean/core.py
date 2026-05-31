"""
Core implementation of LLMClean with Universal AI Provider Support (Ollama, Gemini, OpenAI, Groq).
"""

import json
import os
import re
import requests
from typing import Any, List, Optional
import pandas as pd
from google import genai
from openai import OpenAI

OLLAMA_URL: str = "http://localhost:11434/api/generate"

def _ask_universal_llm(
    prompt: str, 
    provider: str = "ollama", 
    model: Optional[str] = None, 
    api_key: Optional[str] = None
) -> str:
    """
    Core routing engine that handles multi-provider LLM requests seamlessly.
    """
    provider = provider.lower().strip()
    
    # 1. GOOGLE GEMINI PROVIDER
    if provider == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            print("[ERROR] Gemini API Key missing. Pass 'api_key' or set GEMINI_API_KEY env variable.")
            return ""
        try:
            target_model = model or "gemini-2.5-flash"
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model=target_model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini API Execution Failure: {e}")
            return ""

    # 2. OPENAI (CHATGPT) PROVIDER
    elif provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("[ERROR] OpenAI API Key missing. Pass 'api_key' or set OPENAI_API_KEY env variable.")
            return ""
        try:
            target_model = model or "gpt-4o-mini"
            client = OpenAI(api_key=key)
            completion = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI API Execution Failure: {e}")
            return ""

    # 3. GROQ CLOUD PROVIDER (Free & Fast Llama/Mixtral)
    elif provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            print("[ERROR] Groq API Key missing. Pass 'api_key' or set GROQ_API_KEY env variable.")
            return ""
        try:
            target_model = model or "llama3-8b-8192"
            client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            completion = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] Groq API Execution Failure: {e}")
            return ""

    # 4. DEFAULT LOCAL OLLAMA PROVIDER
    else:
        try:
            target_model = model or "llama3.2"
            payload = {
                "model": target_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            print(f"[ERROR] Ollama Engine returned HTTP Status Code: {response.status_code}")
            return ""
        except requests.exceptions.ConnectionError:
            print("[ERROR] Connection refused. Verify Ollama is active via 'ollama serve' or configure a Cloud Provider.")
            return ""

def clean_text(
    text: Any,
    instruction: str = "Fix spelling and grammar errors. Return only the corrected text, nothing else.",
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Any:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text

    text_str = str(text).strip()
    if not text_str:
        return text

    prompt = f"{instruction}\n\nInput Text: {text_str}\nReturn ONLY the final output without any markdown formatting or extra talk."
    return _ask_universal_llm(prompt, provider=provider, model=model, api_key=api_key)

def clean_column(
    df: pd.DataFrame,
    column: str,
    instruction: Optional[str] = None,
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.Series:
    if column not in df.columns:
        print(f"[ERROR] Specified feature column '{column}' missing from DataFrame.")
        return df[column]

    if instruction is None:
        instruction = "Fix any spelling mistakes and standardize the format. Return only the cleaned value, nothing else."

    total_rows = len(df)
    print(f"\n[INFO] Initializing cleansing pipeline [{provider.upper()}] for '{column}' | Total Rows: {total_rows}")

    results: List[Any] = []
    for i, val in enumerate(df[column], 1):
        print(f"  Processing Record Stack: {i}/{total_rows}", end="\r")
        if pd.isna(val) or str(val).strip() == "":
            results.append(val)
        else:
            results.append(clean_text(str(val), instruction, provider=provider, model=model, api_key=api_key))

    print(f"\n[SUCCESS] Feature processing batch for '{column}' executed successfully.")
    return pd.Series(results, index=df.index)

def standardize_column(
    df: pd.DataFrame, 
    column: str, 
    categories: List[str], 
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.Series:
    cat_list = ", ".join(categories)
    instruction = (
        f"Map the input value to the closest matching option from this list ONLY: [{cat_list}]. "
        f"Return only the exact category name from the list, nothing else. "
        f"If nothing matches, return the original value unchanged."
    )
    return clean_column(df, column, instruction=instruction, provider=provider, model=model, api_key=api_key)

def fill_missing(
    df: pd.DataFrame,
    column: str,
    context_columns: Optional[List[str]] = None,
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.Series:
    result = df[column].copy()
    missing_count = df[column].isna().sum()

    if missing_count == 0:
        return result

    print(f"\n[INFO] Missing fields detection active on '{column}'. Imputing via {provider.upper()}...")

    for idx, row in df.iterrows():
        if pd.isna(row[column]) or str(row[column]).strip() == "":
            context = ""
            if context_columns:
                for ctx_col in context_columns:
                    if ctx_col in df.columns and pd.notna(row[ctx_col]):
                        context += f"- {ctx_col}: {row[ctx_col]}\n"

            if not context:
                context = "No structural meta-context available."

            prompt = (
                f"Based on the following context, what is the most likely value for '{column}'?\n"
                f"Context:\n{context}\n"
                f"Return only the exact single value, nothing else. No explanation."
            )

            filled_value = _ask_universal_llm(prompt, provider=provider, model=model, api_key=api_key)
            if filled_value:
                result[idx] = filled_value
                print(f"  [IMPUTED] Index {idx}: '{column}' -> filled with '{filled_value}'")

    return result

def detect_anomalies(
    df: pd.DataFrame, 
    column: str, 
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[str]:
    values = df[column].dropna().astype(str).tolist()
    sample = values[:30]

    prompt = (
        f"Here are values from a dataset column named '{column}':\n"
        f"{json.dumps(sample, ensure_ascii=False)}\n\n"
        f"Identify which values look like errors, typos, invalid entries, or anomalies.\n"
        f"Return ONLY a JSON array of suspicious values, like: [\"bad_value1\", \"bad_value2\"]\n"
        f"If everything looks valid, return an empty array: []\n"
        f"Return ONLY the JSON array. No explanations or extra text."
    )

    response = _ask_universal_llm(prompt, provider=provider, model=model, api_key=api_key)

    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []

def clean_dataframe(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    cleaned_df = df.copy()
    for col in columns:
        cleaned_df[col] = clean_column(cleaned_df, col, provider=provider, model=model, api_key=api_key)

    return cleaned_df
