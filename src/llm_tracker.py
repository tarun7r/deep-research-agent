"""LLM call tracking and token usage monitoring."""

from typing import Dict, Optional, Any
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


class LLMCallTracker:
    """Track LLM calls and token usage."""
    
    def __init__(self):
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def track_call(
        self,
        agent_name: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        model: str = "",
        success: bool = True,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track an LLM call."""
        call_info = {
            'agent': agent_name,
            'operation': operation,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'duration': round(duration, 2),
            'success': success,
            'error': error,
            'timestamp': time.time()
        }
        
        self.calls.append(call_info)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        logger.info(
            f"LLM Call [{agent_name}/{operation}]: "
            f"{input_tokens} in + {output_tokens} out = {input_tokens + output_tokens} tokens "
            f"({duration:.2f}s)"
        )
        
        return call_info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all LLM calls."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        total_duration = sum(call['duration'] for call in self.calls)
        
        # Group by agent
        by_agent = {}
        for call in self.calls:
            agent = call['agent']
            if agent not in by_agent:
                by_agent[agent] = {
                    'calls': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'duration': 0.0
                }
            by_agent[agent]['calls'] += 1
            by_agent[agent]['input_tokens'] += call['input_tokens']
            by_agent[agent]['output_tokens'] += call['output_tokens']
            by_agent[agent]['total_tokens'] += call['total_tokens']
            by_agent[agent]['duration'] += call['duration']
        
        return {
            'total_calls': len(self.calls),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'total_duration': round(total_duration, 2),
            'by_agent': by_agent,
            'successful_calls': sum(1 for c in self.calls if c['success']),
            'failed_calls': sum(1 for c in self.calls if not c['success'])
        }
    
    def get_calls(self) -> list:
        """Get all tracked calls."""
        return self.calls


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: 1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


def track_llm_call(agent_name: str, operation: str, model: str = ""):
    """Decorator to track LLM calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Try to extract token info from result if available
                input_tokens = kwargs.get('_input_tokens', 0)
                output_tokens = kwargs.get('_output_tokens', 0)
                
                # If not provided, estimate based on result
                if output_tokens == 0 and isinstance(result, str):
                    output_tokens = estimate_tokens(result)
                
                return result, {
                    'agent': agent_name,
                    'operation': operation,
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': duration,
                    'success': True
                }
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"LLM call failed: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"LLM call failed: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

