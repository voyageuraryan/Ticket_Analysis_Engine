"""
Main RAG pipeline for SAP module classification.
"""

import google.generativeai as genai
from groq import Groq, RateLimitError
from typing import Dict, List, Optional
from loguru import logger
import re
import time
from .prompt_templates import PromptTemplates


class SAPModuleClassifier:
    """Main classifier using RAG pipeline."""
    
    def __init__(
        self,
        hybrid_retriever,
        sap_modules: List[str],
        llm_provider: str = "gemini",
        api_key: str = None,
        groq_api_keys: Optional[List[str]] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        use_raft: bool = True
    ):
        """
        Initialize classifier.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            sap_modules: List of SAP modules
            llm_provider: "gemini" or "groq"
            api_key: Primary API key (Gemini or Groq)
            groq_api_keys: List of backup Groq API keys for rotation
            model_name: LLM model name
            temperature: Generation temperature
            use_raft: Whether to use RAFT-style prompting
        """
        self.llm_provider = llm_provider
        self.hybrid_retriever = hybrid_retriever
        self.sap_modules = sap_modules
        self.temperature = temperature
        self.use_raft = use_raft
        self.model_name = model_name
        
        # Initialize LLM based on provider
        if llm_provider == "groq":
            # Setup Groq with rotation
            self.groq_api_keys = [api_key]
            if groq_api_keys:
                self.groq_api_keys.extend(groq_api_keys)
            
            self.current_key_index = 0
            self.groq_client = Groq(api_key=self.groq_api_keys[0])
            
            logger.info(f"Initialized SAPModuleClassifier with Groq: {model_name}")
            if len(self.groq_api_keys) > 1:
                logger.info(f"🔄 API key rotation enabled: {len(self.groq_api_keys)} keys available")
        else:
            # Setup Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized SAPModuleClassifier with Gemini: {model_name}")
        
        # Get system prompt
        self.system_prompt = PromptTemplates.get_system_prompt(sap_modules)
    
    def _rotate_api_key(self) -> bool:
        """
        Rotate to next Groq API key.
        
        Returns:
            True if rotated successfully, False if all keys exhausted
        """
        if self.llm_provider != "groq":
            return False
        
        if self.current_key_index < len(self.groq_api_keys) - 1:
            self.current_key_index += 1
            self.groq_client = Groq(api_key=self.groq_api_keys[self.current_key_index])
            logger.info(f"🔄 Rotated to API key #{self.current_key_index + 1}")
            return True
        else:
            logger.warning(f"⚠️  All {len(self.groq_api_keys)} API keys exhausted")
            return False
    
    def _call_llm(self, prompt: str, max_retries: int = 5) -> Optional[str]:
        """
        Call LLM with retry and rotation logic.
        
        Args:
            prompt: Prompt to send
            max_retries: Maximum retry attempts
            
        Returns:
            LLM response or None
        """
        if self.llm_provider == "groq":
            return self._call_groq_with_retry(prompt, max_retries)
        else:
            return self._call_gemini(prompt)
    
    def _call_groq_with_retry(self, prompt: str, max_retries: int = 5) -> Optional[str]:
        """Call Groq API with retry and key rotation."""
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                return response.choices[0].message.content.strip()
            
            except RateLimitError as e:
                logger.warning(f"⚠️  Rate limit hit (Key #{self.current_key_index + 1}): {str(e)}")
                
                if self._rotate_api_key():
                    logger.info(f"✅ Continuing with fresh rate limits (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"⏳ All keys exhausted, waiting 60 seconds...")
                        time.sleep(60)
                        # Reset to first key after waiting
                        self.current_key_index = 0
                        self.groq_client = Groq(api_key=self.groq_api_keys[0])
                        logger.info("🔄 Reset to first API key")
                    else:
                        logger.error("❌ All retries exhausted")
                        return None
            
            except Exception as e:
                logger.error(f"❌ Groq API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
        
        return None
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': 2048
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            return None
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse LLM response to extract module and reasoning.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed response dict
        """
        # Extract module
        module_match = re.search(r'Module:\s*([^\n]+)', response_text, re.IGNORECASE)
        predicted_module = module_match.group(1).strip() if module_match else "Unknown"
        
        # Extract confidence
        confidence_match = re.search(r'Confidence:\s*([^\n]+)', response_text, re.IGNORECASE)
        confidence_text = confidence_match.group(1).strip() if confidence_match else "Medium"
        
        # Map confidence to score
        confidence_map = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        confidence_score = confidence_map.get(confidence_text.lower(), 0.7)
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:(.*?)(?:Key Indicators:|Similar Tickets|$)', response_text, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract key indicators
        indicators_match = re.search(r'Key Indicators:(.*?)(?:Similar Tickets|$)', response_text, re.DOTALL | re.IGNORECASE)
        key_indicators = indicators_match.group(1).strip() if indicators_match else ""
        
        return {
            'module': predicted_module,
            'confidence': confidence_score,
            'confidence_level': confidence_text,
            'reasoning': reasoning,
            'key_indicators': key_indicators,
            'raw_response': response_text
        }
    
    def predict(
        self,
        summary: str,
        description: str,
        incident_number: str = None,
        top_k: int = 10
    ) -> Dict:
        """
        Predict SAP module for a ticket.
        
        Args:
            summary: Ticket summary
            description: Ticket description
            incident_number: Optional incident number
            top_k: Number of similar tickets to retrieve
            
        Returns:
            Prediction result with explanation
        """
        # Combine text
        query_text = f"{summary} [SEP] {description}"
        
        logger.info(f"Classifying ticket: {incident_number or 'NEW'}")
        
        # Retrieve similar tickets
        similar_tickets = self.hybrid_retriever.retrieve(
            query=query_text,
            k=top_k
        )
        
        # Separate by module for RAFT
        if self.use_raft and len(similar_tickets) >= 3:
            # Try to get golden and distractors
            modules_in_results = {}
            for ticket in similar_tickets:
                module = ticket['metadata']['Module']
                if module not in modules_in_results:
                    modules_in_results[module] = []
                modules_in_results[module].append(ticket)
            
            # Get golden docs (from most common module in results)
            most_common_module = max(modules_in_results.keys(), key=lambda k: len(modules_in_results[k]))
            golden_docs = modules_in_results[most_common_module][:2]
            
            # Get distractors (from other modules)
            distractor_docs = []
            for module, tickets in modules_in_results.items():
                if module != most_common_module:
                    distractor_docs.extend(tickets[:1])
            distractor_docs = distractor_docs[:2]
            
            # Format RAFT prompt
            user_prompt = PromptTemplates.format_raft_prompt(
                query_text=query_text,
                golden_docs=golden_docs,
                distractor_docs=distractor_docs,
                sap_modules=self.sap_modules
            )
        else:
            # Standard RAG prompt
            user_prompt = PromptTemplates.format_classification_prompt(
                query_text=query_text,
                similar_tickets=similar_tickets,
                sap_modules=self.sap_modules
            )
        
        # Generate prediction
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        response_text = self._call_llm(full_prompt)
        
        if not response_text:
            logger.error("LLM generation failed")
            return {
                'module': 'Unknown',
                'confidence': 0.0,
                'error': 'LLM call failed',
                'similar_tickets': []
            }
        
        # Parse response
        parsed = self._parse_response(response_text)
        
        # Add similar tickets info
        parsed['similar_tickets'] = [
            {
                'id': ticket.get('id'),
                'module': ticket['metadata']['Module'],
                'text': ticket['text'][:200] + '...',
                'similarity': ticket.get('similarity', 0),
                'rank': ticket.get('hybrid_rank', ticket.get('rank', 0))
            }
            for ticket in similar_tickets[:5]
        ]
        
        # Add input info
        parsed['input'] = {
            'summary': summary,
            'description': description,
            'incident_number': incident_number
        }
        
        # Add uncertainty flag
        parsed['needs_review'] = parsed['confidence'] < 0.6
        
        logger.info(f"Predicted module: {parsed['module']} (confidence: {parsed['confidence']:.2f})")
        
        return parsed
    
    def predict_batch(self, tickets: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Predict modules for multiple tickets.
        
        Args:
            tickets: List of ticket dicts with 'summary' and 'description'
            top_k: Number of similar tickets to retrieve
            
        Returns:
            List of prediction results
        """
        logger.info(f"Batch prediction for {len(tickets)} tickets")
        
        results = []
        for ticket in tickets:
            result = self.predict(
                summary=ticket.get('summary', ''),
                description=ticket.get('description', ''),
                incident_number=ticket.get('incident_number'),
                top_k=top_k
            )
            results.append(result)
        
        return results
