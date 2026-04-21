"""Groq API client - business logic only"""
import requests
from typing import List, Dict, Optional
import re

class GroqClient:
    """Groq API client for LLM inference"""
    
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are an expert AI assistant specializing in:
- Machine Learning and Deep Learning
- Neural Network Optimization (quantization, pruning, knowledge distillation)
- Model Deployment and Inference
- PyTorch, TensorFlow, and ONNX
- Model Compression Techniques
- Hardware Acceleration (GPU, TPU, edge devices)

IMPORTANT GUIDELINES:
- Stay focused on ML/DL, optimization, and deployment topics
- Provide practical, actionable advice
- Be concise but thorough
- Provide code examples when helpful

OPTIMIZATION EXECUTION RULES:
You can execute optimizations using special commands, but ONLY when:
1. You have ALREADY made a recommendation in a PREVIOUS message, AND
2. The user has EXPLICITLY agreed with words like: "yes", "do it", "apply it", "go ahead", "please", "sure", "okay"

Commands (use ONLY after user agrees):
- [APPLY_QUANTIZATION] - apply quantization
- [APPLY_PRUNING:X] - apply X% pruning (e.g., [APPLY_PRUNING:50])
- [APPLY_BOTH:X] - apply quantization + X% pruning

CRITICAL: 
- When FIRST recommending optimizations, DO NOT include any [APPLY_...] tags
- ONLY include tags in your RESPONSE to user's agreement
- If user asks questions or wants more info, DO NOT include tags

Example Flow:
User: "My model is too large"
You: "I recommend quantization + 50% pruning for 87% reduction. Should I apply this?" [NO TAGS HERE]

User: "yes, do it"
You: "Applying optimizations now! [APPLY_BOTH:50]" [TAGS ONLY HERE]

If asked about unrelated topics, politely redirect to ML/optimization."""
    
    def _is_relevant_query(self, query: str) -> bool:
        """Check if query is ML/optimization related"""
        # Conversational affirmatives
        affirmatives = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'do it', 
                       'go ahead', 'apply', 'please', 'try it']
        
        query_lower = query.lower().strip()
        
        # Check for conversational responses
        if any(query_lower.startswith(word) for word in affirmatives):
            return True
        
        # Check for ML keywords
        ml_keywords = [
            'model', 'neural', 'network', 'optimization', 'quantization', 'pruning',
            'distillation', 'training', 'inference', 'deep learning', 'machine learning',
            'ml', 'dl', 'ai', 'pytorch', 'tensorflow', 'onnx', 'gpu', 'cpu',
            'deployment', 'edge', 'mobile', 'compression', 'latency', 'accuracy',
            'optimize', 'reduce', 'compress', 'faster', 'smaller'
        ]
        
        return any(keyword in query_lower for keyword in ml_keywords)
    
    def _is_user_agreeing(self, message: str, conversation_history: List[Dict]) -> bool:
        """
        Check if user is agreeing to a previous recommendation
        Returns True only if:
        1. Message is a short affirmative, AND
        2. Previous assistant message contained a recommendation/question
        """
        message_lower = message.lower().strip()
        
        # Short affirmatives
        affirmatives = ['yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay', 
                       'do it', 'go ahead', 'apply it', 'please', 'apply', 'try it']
        
        # Must be short (5 words or less) and affirmative
        if len(message.split()) > 5:
            return False
        
        if not any(aff in message_lower for aff in affirmatives):
            return False
        
        # Check if last assistant message was a question/recommendation
        if conversation_history and len(conversation_history) > 0:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '')
                    break
            
            if last_assistant_msg:
                # Check if it was asking a question or making a recommendation
                recommendation_indicators = [
                    'recommend', 'suggest', 'should i', 'would you like',
                    'do you want', 'shall i', 'ready to apply', 'want me to',
                    'apply', '?'
                ]
                return any(indicator in last_assistant_msg.lower() 
                          for indicator in recommendation_indicators)
        
        return False
    
    def _extract_optimization_commands(self, response_text: str) -> Dict:
        """Extract optimization commands from AI response"""
        commands = {
            'quantization': '[APPLY_QUANTIZATION]' in response_text,
            'pruning': None,
            'both': None
        }
        
        # Check for pruning
        if '[APPLY_PRUNING:' in response_text:
            try:
                start = response_text.index('[APPLY_PRUNING:') + 15
                end = response_text.index(']', start)
                amount = int(response_text[start:end])
                commands['pruning'] = amount / 100
            except:
                pass
        
        # Check for both
        if '[APPLY_BOTH:' in response_text:
            try:
                start = response_text.index('[APPLY_BOTH:') + 12
                end = response_text.index(']', start)
                amount = int(response_text[start:end])
                commands['both'] = amount / 100
                commands['quantization'] = True
                commands['pruning'] = amount / 100
            except:
                pass
        
        return commands
    
    def chat(self, message: str, conversation_history: List[Dict] = None, 
             user_context: Dict = None) -> Dict:
        """
        Send message to Groq and get response
        
        Returns dict with:
        - 'response': AI response text
        - 'commands': Dict of optimization commands to execute
        """
        
        if not self.api_key:
            return {
                'response': "API key error. Please contact support.",
                'commands': None
            }
        
        if not self._is_relevant_query(message):
            return {
                'response': ("I specialize in ML model optimization. Ask me about "
                           "quantization, pruning, model deployment, or performance tuning!"),
                'commands': None
            }
        
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add user context with agreement detection
            if user_context:
                context_info = self._build_context(user_context)
                
                # Add specific instruction based on whether user is agreeing
                if self._is_user_agreeing(message, conversation_history or []):
                    context_info += "\n\n⚡ USER IS AGREEING TO PREVIOUS RECOMMENDATION"
                    context_info += "\n→ Include appropriate [APPLY_...] command in your response NOW"
                else:
                    context_info += "\n\n⚠️ This is a new query or question"
                    context_info += "\n→ DO NOT include [APPLY_...] commands yet - only make recommendations"
                
                if context_info:
                    messages.append({"role": "system", "content": context_info})
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history[-10:])
            
            messages.append({"role": "user", "content": message})
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content']
                
                # Extract commands
                commands = self._extract_optimization_commands(ai_response)
                
                # Clean response (remove command tags)
                clean_response = ai_response
                for tag in ['[APPLY_QUANTIZATION]', '[APPLY_PRUNING:', '[APPLY_BOTH:']:
                    if tag in clean_response:
                        # Remove the command tags
                        clean_response = re.sub(r'\[APPLY_[^\]]+\]', '', clean_response)
                
                return {
                    'response': clean_response.strip(),
                    'commands': commands if any(commands.values()) else None
                }
            
            elif response.status_code == 401:
                return {'response': "Invalid API key", 'commands': None}
            elif response.status_code == 429:
                return {'response': "Rate limit reached. Please wait a moment.", 'commands': None}
            else:
                return {'response': f"API Error ({response.status_code})", 'commands': None}
        
        except Exception as e:
            return {'response': f"Error: {str(e)}", 'commands': None}
    
    def _build_context(self, user_context: Dict) -> str:
        """Build context string from user's model info"""
        context_parts = []
        
        if user_context.get('model_uploaded'):
            context_parts.append(f"User's model: {user_context.get('model_name', 'Unknown')}")
        
        if user_context.get('original_metrics'):
            metrics = user_context['original_metrics']
            context_parts.append(f"Size: {metrics.get('size', 0):.1f} MB")
            context_parts.append(f"Parameters: {metrics.get('total_params', 0):,}")
            context_parts.append(f"Latency: {metrics.get('latency', 0):.1f} ms")
        
        if user_context.get('optimized'):
            reduction = user_context.get('size_reduction', 0)
            context_parts.append(f"Already optimized: {reduction:.1f}% size reduction")
            context_parts.append("⚠️ Model already optimized - DO NOT suggest [APPLY_...] commands")
        else:
            context_parts.append("Model not yet optimized - ready for optimization")
            context_parts.append("\n📋 Available optimization commands:")
            context_parts.append("- [APPLY_QUANTIZATION] for quantization only")
            context_parts.append("- [APPLY_PRUNING:X] for X% pruning only")
            context_parts.append("- [APPLY_BOTH:X] for quantization + X% pruning")
            context_parts.append("\n⚠️ IMPORTANT: Only include these tags AFTER user explicitly agrees!")
        
        return "\n".join(context_parts) if context_parts else ""
