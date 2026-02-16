def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    if not text:
        return 0
    return len(str(text)) // 4

def truncate_text(text: str, max_tokens, model: str = "gpt-3.5-turbo") -> str:
    """
    Safely truncate text even if arguments are passed in the wrong order.
    """
    if not text:
        return ""
    
    # BTP FIX: Force max_tokens to be an int. 
    # If the TUI passed a string (like the model name) here, default to 1000.
    try:
        token_limit = int(max_tokens)
    except (ValueError, TypeError):
        token_limit = 1000
        
    max_chars = token_limit * 4
    
    if len(str(text)) <= max_chars:
        return str(text)
        
    return str(text)[:max_chars] + "... (truncated)"
