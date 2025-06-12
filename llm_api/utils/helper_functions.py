# /llm_api/utils/helper_functions.py
import json
import sys
import aiofiles

async def read_from_pipe_or_file(prompt_arg, file_arg):
    """パイプまたはファイルからプロンプトを非同期に読み込む"""
    if not sys.stdin.isatty():
        # データがパイプされている場合
        return await aiofiles.open(sys.stdin.fileno(), mode='r', encoding='utf-8').read()
    if file_arg:
        # ファイルが指定されている場合
        async with aiofiles.open(file_arg, mode='r', encoding='utf-8') as f:
            return await f.read()
    if prompt_arg:
        # 引数としてプロンプトが渡された場合
        return prompt_arg
    return None

def format_json_output(data: dict) -> str:
    """辞書データを整形されたJSON文字列に変換する"""
    return json.dumps(data, indent=2, ensure_ascii=False)

def get_model_family(model_name: str) -> str:
    """
    モデル名からモデルファミリーを判定する関数を追加。
    例: 'llama3:8b-instruct-q5_K_M' -> 'llama'
    """
    if not model_name:
        return 'unknown'
    
    model_name_lower = model_name.lower()
    
    # 一般的なモデルファミリーのキーワードで判定
    if 'llama' in model_name_lower:
        return 'llama'
    if 'qwen' in model_name_lower:
        return 'qwen'
    if 'gemma' in model_name_lower:
        return 'gemma'
    if 'mistral' in model_name_lower or 'mixtral' in model_name_lower:
        return 'mistral'
    if 'phi' in model_name_lower:
        return 'phi'
    
    return 'unknown'