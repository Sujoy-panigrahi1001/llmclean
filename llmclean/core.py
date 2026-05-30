"""
Core implementation of LLMClean text processing and DataFrame manipulation functions.
"""

import json
import re
from typing import Any, List, Optional
import pandas as pd
import requests

# Ollama local API configuration
OLLAMA_URL: str = "http://localhost:11434/api/generate"
DEFAULT_MODEL: str = "llama3.2"


def _ask_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Internal helper function to send a prompt to the local Ollama LLM.

    Args:
        prompt (str): The structured instruction payload.
        model (str): Target Ollama model name.

    Returns:
        str: Cleansed string response from the model.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print(
            "[ERROR] Connection refused. Verify Ollama is active via 'ollama serve'."
        )
        return ""
    except Exception as e:
        print(f"[ERROR] Unexpected execution failure during LLM inference: {str(e)}")
        return ""


def clean_text(
    text: Any,
    instruction: str = "Fix spelling and grammar errors. Return only the corrected text, nothing else.",
    model: str = DEFAULT_MODEL,
) -> Any:
    """
    Clean and normalize a singular textual value using a specified LLM.

    Args:
        text (Any): Raw scalar entry to cleanse.
        instruction (str): Formatting instructions for the model.
        model (str): Target Ollama model name.

    Returns:
        Any: Processed textual string or original value if null-like.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text

    text_str = str(text).strip()
    if not text_str:
        return text

    prompt = f"{instruction}\n\nInput: {text_str}\nOutput:"
    return _ask_llm(prompt, model)


def clean_column(
    df: pd.DataFrame,
    column: str,
    instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> pd.Series:
    """
    Apply iterative LLM cleansing routines across a targeted DataFrame column.

    Args:
        df (pd.DataFrame): Target source DataFrame.
        column (str): Label of the column to parse.
        instruction (Optional[str]): Overriding system instruction set.
        model (str): Target Ollama model name.

    Returns:
        pd.Series: Mutated Series containing cleansed elements.
    """
    if column not in df.columns:
        print(f"[ERROR] Specified feature column '{column}' missing from DataFrame.")
        return df[column]

    if instruction is None:
        instruction = (
            "Fix any spelling mistakes and standardize the format. "
            "Return only the cleaned value, nothing else."
        )

    total_rows = len(df)
    print(f"\n[INFO] Initializing cleansing pipeline for '{column}' | Total Rows: {total_rows}")

    results: List[Any] = []
    for i, val in enumerate(df[column], 1):
        print(f"  Processing Record Stack: {i}/{total_rows}", end="\r")
        if pd.isna(val) or str(val).strip() == "":
            results.append(val)
        else:
            results.append(clean_text(str(val), instruction, model))

    print(f"\n[SUCCESS] Feature processing batch for '{column}' executed successfully.")
    return pd.Series(results, index=df.index)


def standardize_column(
    df: pd.DataFrame, column: str, categories: List[str], model: str = DEFAULT_MODEL
) -> pd.Series:
    """
    Map polymorphic textual attributes into strict domain-specific categorical buckets.

    Args:
        df (pd.DataFrame): Target source DataFrame.
        column (str): Label of the column to categorize.
        categories (List[str]): Allowed canonical categorical array.
        model (str): Target Ollama model name.

    Returns:
        pd.Series: Standardized categorical pandas Series structure.
    """
    cat_list = ", ".join(categories)
    instruction = (
        f"Map the input value to the closest matching option from this list ONLY: [{cat_list}]. "
        f"Return only the exact category name from the list, nothing else. "
        f"If nothing matches, return the original value unchanged."
    )

    print(f"\n[INFO] Executing label normalization sequence on column: '{column}'")
    print(f"[INFO] Target domain boundaries: {categories}")
    return clean_column(df, column, instruction=instruction, model=model)


def fill_missing(
    df: pd.DataFrame,
    column: str,
    context_columns: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> pd.Series:
    """
    Perform predictive imputation on NaN metrics by analyzing cross-feature dependencies.

    Args:
        df (pd.DataFrame): Target source DataFrame.
        column (str): Objective attribute containing missing fields.
        context_columns (Optional[List[str]]): Auxiliary feature matrices used for context generation.
        model (str): Target Ollama model name.

    Returns:
        pd.Series: Imputed Series object with resolved elements.
    """
    result = df[column].copy()
    missing_count = df[column].isna().sum()

    if missing_count == 0:
        print(f"[INFO] Column '{column}' exhibits zero null fields. Bypassing imputation routines.")
        return result

    print(f"\n[INFO] Detected {missing_count} null instances in '{column}'. Initializing LLM heuristic fill...")

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
                f"Return only the value, nothing else. No explanation."
            )

            filled_value = _ask_llm(prompt, model)
            result[idx] = filled_value
            print(f"  [IMPUTED] Index {idx}: '{column}' -> filled with '{filled_value}'")

    print(f"[SUCCESS] Missing field normalization complete for attribute sequence: '{column}'.")
    return result


def detect_anomalies(df: pd.DataFrame, column: str, model: str = DEFAULT_MODEL) -> List[str]:
    """
    Evaluate structural data feeds to extract syntactical or contextual dataset anomalies.

    Args:
        df (pd.DataFrame): Target source DataFrame.
        column (str): Target processing matrix label.
        model (str): Target Ollama model name.

    Returns:
        List[str]: Parsed collection of flagged anomalous data instances.
    """
    values = df[column].dropna().astype(str).tolist()
    sample = values[:30]  # Optimize window array for standard tokens ceiling constraints

    prompt = (
        f"Here are values from a dataset column named '{column}':\n"
        f"{json.dumps(sample, ensure_ascii=False)}\n\n"
        f"Identify which values look like errors, typos, invalid entries, or anomalies.\n"
        f"Return ONLY a JSON array of suspicious values, like: [\"bad_value1\", \"bad_value2\"]\n"
        f"If everything looks valid, return an empty array: []\n"
        f"Return ONLY the JSON array. No explanations or extra text."
    )

    print(f"\n[INFO] Parsing processing stack for outlier metrics within target attribute: '{column}'...")
    response = _ask_llm(prompt, model)

    try:
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            anomalies: List[str] = json.loads(match.group())
            print(f"[SUCCESS] Extraction routine flagged {len(anomalies)} structural outliers inside '{column}'.")
            return anomalies
    except json.JSONDecodeError:
        pass

    print(f"[WARN] Deserialization protocol failed on anomaly matrix array output: {response[:100]}")
    return []


def clean_dataframe(
    df: pd.DataFrame, columns: Optional[List[str]] = None, model: str = DEFAULT_MODEL
) -> pd.DataFrame:
    """
    Batch execute processing pipelines across broad multi-variate textual features.

    Args:
        df (pd.DataFrame): Raw continuous feature input matrix.
        columns (Optional[List[str]]): Target subset elements. Defaults to object dtypes.
        model (str): Target Ollama model name.

    Returns:
        pd.DataFrame: Completely processed transformation DataFrame matrix wrapper.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
        print(f"[INFO] Automatically identified string objects: {columns}")

    cleaned_df = df.copy()
    print(f"\n[STARTING] Batch execution active for {len(columns)} target features.")

    for col in columns:
        cleaned_df[col] = clean_column(cleaned_df, col, model=model)

    print(f"\n[COMPLETE] Successfully optimized data matrices for user profile layers.")
    return cleaned_df
