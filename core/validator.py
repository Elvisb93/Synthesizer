import hashlib
import logging
import re
from typing import List, Set, Optional, Any
import numpy as np
from .models import GeneratorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not found. Semantic similarity checking will be disabled.")

class UniquenessValidator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.long_text_history: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        
        self.has_transformers = HAS_SENTENCE_TRANSFORMERS
        
        self.model = None
        if self.has_transformers:
            try:
                # Use a small, fast model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}")
                self.has_transformers = False

    def is_unique(self, text: str, field_type: str = "Long Text", threshold_override: Optional[float] = None) -> bool:
        """
        Check if text is unique.
        For Short Text: Exact match only.
        For Long Text: Exact match AND Semantic Similarity.
        """
        if not text:
            return True # Empty text is considered "unique" in sense of not crashing
        
        # 1. Exact Match Check
        text_str = str(text)
        text_hash = hashlib.sha256(text_str.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            logger.info("Duplicate rejected: Exact match found.")
            return False
            
        # 2. Semantic Similarity Check (Only for Long Text)
        if field_type == "Long Text" and self.has_transformers and self.model:
            # Only check if we have history
            if self.embeddings is not None and len(self.embeddings) > 0:
                # Encode ONLY current text
                current_embedding = self.model.encode(text, convert_to_tensor=True)
                
                # Compute cosine similarity against cached embeddings
                # util.cos_sim returns a tenser of shape (1, len(history))
                cosine_scores = util.cos_sim(current_embedding, self.embeddings)[0]
                
                max_sim = float(max(cosine_scores))
                # Use override if provided, else global config
                threshold = threshold_override if threshold_override is not None else self.config.similarity_threshold
                
                if max_sim > threshold:
                    logger.info(f"Duplicate rejected: Semantic similarity {max_sim:.4f} > {threshold}")
                    return False

        # If we reach here, it's valid.
        return True

    def validate_regex(self, text: str, pattern: str) -> bool:
        """Phase 4: Validate against regex pattern."""
        if not pattern:
            return True
        
        # Shortcuts mapping
        shortcuts = {
            'email': r'^[\w\.-]+@[\w\.-]+\.\w+$',
            'phone': r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
            'zip': r'^\d{5}(-\d{4})?$',
            'postcode': r'^\d{5}(-\d{4})?$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'ipv4': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        }
        
        # Use shortcut if it matches, otherwise treat as raw regex
        real_pattern = shortcuts.get(pattern.lower(), pattern)
        
        try:
            return bool(re.match(real_pattern, text))
        except re.error:
            logger.error(f"Invalid regex pattern: {pattern}")
            return False

    def validate_logic(self, value: Any, expression: str, row_data: dict) -> bool:
        """Phase 4: Validate cross-column logic using eval().
        Expression example: "this > @[age]" -> "value > row_data['age']"
        """
        if not expression:
            return True
            
        # Interpolate @[column] with row_data values
        # We need to be careful about types.
        # Simple string replacement for names to values might be tricky with quotes.
        # Better approach: Place row_data into a local scope and use names directly?
        # But user syntax is @[name].
        
        try:
            # 1. Replace @[col_name] with row_data['col_name'] safely
            # We can pass row_data as a dict to eval
            # And replace @[col] with row_data['col'] in string? 
            
            # Regex to find @[name]
            # We replace @[name] with `row_data.get('name')`
            
            def replace_match(match):
                col_name = match.group(1)
                # We need to make sure we access the dict
                # If col_name has spaces, dict access works fine: row_data.get('Start Date')
                return f"row_data.get('{col_name}')"
            
            # The 'this' keyword represents the current value
            # define scope
            scope = {'this': value, 'row_data': row_data}
            
            # 0. Pre-process Natural Language
            # Map natural phrases to python operators
            # Note: Order matters (longest match first)
            replacements = [
                (r'\bis greater than\b', '>'),
                (r'\bgreater than\b', '>'),
                (r'\bafter\b', '>'),
                (r'\bis less than\b', '<'),
                (r'\bless than\b', '<'),
                (r'\bbefore\b', '<'),
                (r'\bis equal to\b', '=='),
                (r'\bequals\b', '=='),
                (r'\bis not\b', '!='),
                (r'\bnot equal\b', '!='),
                # "longer than 5" -> "len(this) > 5"
                # This is trickier regex. Let's do simple substitution first.
            ]
            
            normalized_expr = expression
            for phrase, op in replacements:
                normalized_expr = re.sub(phrase, op, normalized_expr, flags=re.IGNORECASE)
            
            # Special case: "longer than X" -> "len(this) > X"
            # Regex to find "longer than (\d+)" -> "len(this) > \1"
            normalized_expr = re.sub(r'longer than\s+(\d+)', r'len(this) > \1', normalized_expr, flags=re.IGNORECASE)
            # Also "shorter than X"
            normalized_expr = re.sub(r'shorter than\s+(\d+)', r'len(this) < \1', normalized_expr, flags=re.IGNORECASE)

            # Check for implied "this" (e.g. "> 10" becomes "this > 10")
            stripped = normalized_expr.strip()
            if stripped.startswith(('<', '>', '=', '!', '==', '!=', '>=', '<=')): 
                normalized_expr = f"this {normalized_expr}"

            # Transform expression: "this > @[age]" -> "this > row_data.get('age')"
            transformed_expr = re.sub(r'@\[(.*?)\]', replace_match, normalized_expr)
            
            # Allow basic builtins safely
            safe_globals = {"__builtins__": None, "len": len, "str": str, "int": int, "float": float}
            result = eval(transformed_expr, safe_globals, scope)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Logic validation error for '{expression}': {e}")
            return False

    def commit(self, text: str, field_type: str = "Long Text"):
        """Call this ONLY when the row is fully accepted to update the history."""
        # 1. Exact Match Check
        text_str = str(text)
        text_hash = hashlib.sha256(text_str.encode('utf-8')).hexdigest()
        self.seen_hashes.add(text_hash)
        
        if field_type == "Long Text" and self.has_transformers and self.model:
            self.long_text_history.append(text)
            
            # Encode just the new text
            new_embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Append to cache
            if self.embeddings is None:
                # Shape (1, 384) for MiniLM usually
                self.embeddings = new_embedding.unsqueeze(0)
            else:
                import torch
                # Concatenate along dimension 0
                self.embeddings = torch.cat((self.embeddings, new_embedding.unsqueeze(0)), 0)
            
    def clear(self):
        self.seen_hashes.clear()
        self.long_text_history.clear()
        self.embeddings = None
