"""
Data augmentation module for handling class imbalance.
"""

import pandas as pd
import google.generativeai as genai
import ollama
from groq import Groq, RateLimitError
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm
import time


class TicketAugmenter:
    """Augment ticket data to balance classes."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", target_counts: Dict[str, int] = None, 
                 llm_provider: str = "groq", groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant",
                 use_local_llm: bool = False, local_model: str = "gemma:2b",
                 groq_api_keys: Optional[List[str]] = None):
        """
        Initialize augmenter.
        
        Args:
            api_key: Gemini API key (optional if using Groq/Ollama)
            model_name: Model to use for generation (Gemini)
            target_counts: Target counts per module
            llm_provider: LLM provider to use ("groq", "ollama", or "gemini")
            groq_api_key: Primary Groq API key (for Groq provider)
            groq_model: Groq model name
            use_local_llm: Whether to use local Ollama LLM (legacy parameter)
            local_model: Local Ollama model name
            groq_api_keys: List of backup Groq API keys for automatic rotation
        """
        self.llm_provider = llm_provider
        self.groq_model = groq_model
        self.local_model = local_model
        
        # Setup API key rotation for Groq
        self.groq_api_keys = []
        self.current_key_index = 0
        
        if llm_provider == "groq":
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required when using Groq provider")
            
            # Build list of API keys (primary + backups)
            self.groq_api_keys = [groq_api_key]
            if groq_api_keys:
                self.groq_api_keys.extend(groq_api_keys)
            
            # Initialize with first key
            self.groq_client = Groq(api_key=self.groq_api_keys[0])
            self.model = None
            
            logger.info(f"Using Groq API with model: {groq_model}")
            if len(self.groq_api_keys) > 1:
                logger.info(f"✅ API key rotation enabled: {len(self.groq_api_keys)} keys available")
            else:
                logger.warning("⚠️  Only 1 API key configured. Consider adding backup keys for automatic rotation.")
        
        # Initialize based on provider
        elif llm_provider == "ollama" or use_local_llm:
            self.groq_client = None
            self.model = None
            logger.info(f"Using local Ollama model: {local_model}")
        else:  # gemini
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required when using Gemini provider")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.groq_client = None
            logger.info(f"Using Gemini API with model: {model_name}")
        
        self.target_counts = target_counts or {
            'Basis': 4302,        # Keep original
            'HR & Payroll': 2000,
            'Procurement': 1500,
            'Connections': 1500,
            'FICO': 1500,
            'ABAP': 1500
        }
    
    def _rotate_api_key(self) -> bool:
        """
        Rotate to the next available Groq API key.
        
        Returns:
            True if rotation successful, False if no more keys available
        """
        if self.llm_provider != "groq" or len(self.groq_api_keys) <= 1:
            return False
        
        # Try next key
        self.current_key_index += 1
        
        if self.current_key_index >= len(self.groq_api_keys):
            logger.error("❌ All API keys exhausted! No more backup keys available.")
            return False
        
        # Switch to next key
        new_key = self.groq_api_keys[self.current_key_index]
        self.groq_client = Groq(api_key=new_key)
        
        logger.warning(f"🔄 Switched to backup API key #{self.current_key_index + 1}")
        logger.info("✅ Continuing with fresh rate limits...")
        
        return True
    
    def _call_groq_with_retry(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024, 
                              max_retries: int = 3) -> Optional[str]:
        """
        Call Groq API with automatic retry and key rotation on rate limit.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries (including key rotations)
            
        Returns:
            Generated text or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            except RateLimitError as e:
                logger.warning(f"⚠️  Rate limit hit: {str(e)}")
                
                # Try to rotate to next key
                if self._rotate_api_key():
                    logger.info(f"🔄 Retrying with new API key (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    # No more keys available
                    if attempt < max_retries - 1:
                        wait_time = 60
                        logger.warning(f"⏳ Waiting {wait_time} seconds for rate limit to reset...")
                        time.sleep(wait_time)
                        logger.info(f"🔄 Retrying (attempt {attempt + 1}/{max_retries})...")
                    else:
                        logger.error("❌ Max retries reached. Giving up on this request.")
                        return None
            
            except Exception as e:
                logger.error(f"❌ API call failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    return None
        
        return None
    
    def paraphrase_ticket(self, summary: str, description: str, module: str) -> Dict[str, str]:
        """
        Generate paraphrased version of ticket.
        
        Args:
            summary: Original summary
            description: Original description
            module: SAP module
            
        Returns:
            Dict with paraphrased summary and description
        """
        prompt = f"""You are an SAP support specialist. Paraphrase the following ticket while maintaining technical accuracy and the same meaning.

Original Ticket:
Summary: {summary}
Description: {description}
Module: {module}

Generate a paraphrased version that:
1. Uses different wording but conveys the same technical issue
2. Maintains SAP terminology accuracy
3. Keeps the same level of detail
4. Sounds natural and realistic

Paraphrased Ticket:
Summary:"""
        
        try:
            # Generate response based on provider
            if self.llm_provider == "groq":
                text = self._call_groq_with_retry(prompt, temperature=0.2, max_tokens=1024)
                if text is None:
                    logger.error("Failed to generate paraphrase after retries")
                    return {'summary': summary, 'description': description}
            elif self.llm_provider == "ollama":
                response = ollama.generate(model=self.local_model, prompt=prompt)
                text = response['response']
            else:  # gemini
                response = self.model.generate_content(prompt)
                text = response.text
            
            # Parse response
            lines = text.strip().split('\n')
            paraphrased_summary = ""
            paraphrased_description = ""
            
            current_field = None
            for line in lines:
                if line.startswith("Summary:"):
                    current_field = "summary"
                    paraphrased_summary = line.replace("Summary:", "").strip()
                elif line.startswith("Description:"):
                    current_field = "description"
                    paraphrased_description = line.replace("Description:", "").strip()
                elif current_field == "summary" and line.strip() and not line.startswith("Description"):
                    paraphrased_summary += " " + line.strip()
                elif current_field == "description" and line.strip():
                    paraphrased_description += " " + line.strip()
            
            return {
                'summary': paraphrased_summary or summary,
                'description': paraphrased_description or description
            }
        
        except Exception as e:
            logger.error(f"Paraphrasing failed: {e}")
            return {'summary': summary, 'description': description}
    
    def generate_synthetic_ticket(self, examples: List[Dict], module: str) -> Dict[str, str]:
        """
        Generate completely synthetic ticket based on examples.
        
        Args:
            examples: List of example tickets from same module
            module: Target module
            
        Returns:
            Dict with synthetic summary and description
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\nSummary: {ex['Summary']}\nDescription: {ex['Description']}"
            for i, ex in enumerate(examples[:3])
        ])
        
        prompt = f"""You are an SAP support specialist. Generate a NEW, realistic SAP {module} ticket based on these examples:

{examples_text}

Requirements:
1. Create a completely NEW issue (not a copy of examples)
2. Use realistic SAP terminology for {module} module
3. Include technical details and error messages if appropriate
4. Make it sound like a real user-reported issue
5. Keep it concise but informative

New Ticket:
Summary:"""
        
        try:
            # Generate response based on provider
            if self.llm_provider == "groq":
                text = self._call_groq_with_retry(prompt, temperature=0.3, max_tokens=1024)
                if text is None:
                    logger.error("Failed to generate synthetic ticket after retries")
                    return None
            elif self.llm_provider == "ollama":
                response = ollama.generate(model=self.local_model, prompt=prompt)
                text = response['response']
            else:  # gemini
                response = self.model.generate_content(prompt)
                text = response.text
            
            # Parse response (same as paraphrase)
            lines = text.strip().split('\n')
            synthetic_summary = ""
            synthetic_description = ""
            
            current_field = None
            for line in lines:
                if line.startswith("Summary:"):
                    current_field = "summary"
                    synthetic_summary = line.replace("Summary:", "").strip()
                elif line.startswith("Description:"):
                    current_field = "description"
                    synthetic_description = line.replace("Description:", "").strip()
                elif current_field == "summary" and line.strip() and not line.startswith("Description"):
                    synthetic_summary += " " + line.strip()
                elif current_field == "description" and line.strip():
                    synthetic_description += " " + line.strip()
            
            return {
                'summary': synthetic_summary,
                'description': synthetic_description
            }
        
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
            return None
    
    def augment_single_module(self, df: pd.DataFrame, module_name: str, delay_seconds: float = 0.0) -> pd.DataFrame:
        """
        Augment a single module.
        
        Args:
            df: Original cleaned dataset
            module_name: Name of the module to augment
            delay_seconds: Delay between API calls
            
        Returns:
            Augmented records for this module only
        """
        logger.info(f"Starting augmentation for module: {module_name}")
        
        if module_name not in self.target_counts:
            logger.error(f"Module {module_name} not found in target_counts")
            return pd.DataFrame()
        
        target_count = self.target_counts[module_name]
        module_df = df[df['Module'] == module_name]
        current_count = len(module_df)
        needed = target_count - current_count
        
        if needed <= 0:
            logger.info(f"{module_name}: {current_count} records (no augmentation needed)")
            return pd.DataFrame()
        
        logger.info(f"{module_name}: {current_count} → {target_count} (need {needed} more)")
        
        augmented_records = []
        
        # Strategy: 70% paraphrasing, 30% synthetic
        paraphrase_count = int(needed * 0.7)
        synthetic_count = needed - paraphrase_count
        
        # Paraphrasing
        logger.info(f"  Generating {paraphrase_count} paraphrased tickets...")
        for i in tqdm(range(paraphrase_count), desc=f"Paraphrasing {module_name}"):
            try:
                # Sample random ticket
                sample = module_df.sample(1).iloc[0]
                
                paraphrased = self.paraphrase_ticket(
                    sample['Summary'],
                    sample['Description'],
                    module_name
                )
                
                augmented_records.append({
                    'Incident': f"AUG_{module_name}_{len(augmented_records)}",
                    'Service': sample['Service'],
                    'Module': module_name,
                    'Status': sample['Status'],
                    'Summary': paraphrased['summary'],
                    'Description': paraphrased['description'],
                    'Priority': sample['Priority'],
                    'Team': sample['Team'],
                    'Augmented': 'Paraphrase'
                })
                
                time.sleep(delay_seconds)  # Rate limiting
            except Exception as e:
                logger.error(f"Error paraphrasing ticket {i}: {e}")
                continue
        
        # Synthetic generation
        logger.info(f"  Generating {synthetic_count} synthetic tickets...")
        examples = module_df.sample(min(5, len(module_df))).to_dict('records')
        
        for i in tqdm(range(synthetic_count), desc=f"Synthesizing {module_name}"):
            try:
                synthetic = self.generate_synthetic_ticket(examples, module_name)
                
                if synthetic:
                    augmented_records.append({
                        'Incident': f"AUG_{module_name}_{len(augmented_records)}",
                        'Service': 'SAP',
                        'Module': module_name,
                        'Status': 'Closed',
                        'Summary': synthetic['summary'],
                        'Description': synthetic['description'],
                        'Priority': 2,
                        'Team': 'SAP - ENZEN',
                        'Augmented': 'Synthetic'
                    })
                
                time.sleep(delay_seconds)  # Rate limiting
            except Exception as e:
                logger.error(f"Error generating synthetic ticket {i}: {e}")
                continue
        
        logger.info(f"Generated {len(augmented_records)} augmented records for {module_name}")
        
        return pd.DataFrame(augmented_records)
    
    def augment_dataset(self, df: pd.DataFrame, delay_seconds: float = 0.0) -> pd.DataFrame:
        """
        Augment dataset to balance classes.
        
        Args:
            df: Original cleaned dataset
            delay_seconds: Delay between API calls
            
        Returns:
            Augmented dataset
        """
        logger.info("Starting data augmentation")
        
        augmented_records = []
        
        for module, target_count in self.target_counts.items():
            module_df = df[df['Module'] == module]
            current_count = len(module_df)
            needed = target_count - current_count
            
            if needed <= 0:
                logger.info(f"{module}: {current_count} records (no augmentation needed)")
                continue
            
            logger.info(f"{module}: {current_count} → {target_count} (need {needed} more)")
            
            # Strategy: 70% paraphrasing, 30% synthetic
            paraphrase_count = int(needed * 0.7)
            synthetic_count = needed - paraphrase_count
            
            # Paraphrasing
            logger.info(f"  Generating {paraphrase_count} paraphrased tickets...")
            for _ in tqdm(range(paraphrase_count), desc=f"Paraphrasing {module}"):
                # Sample random ticket
                sample = module_df.sample(1).iloc[0]
                
                paraphrased = self.paraphrase_ticket(
                    sample['Summary'],
                    sample['Description'],
                    module
                )
                
                augmented_records.append({
                    'Incident': f"AUG_{len(augmented_records)}",
                    'Service': sample['Service'],
                    'Module': module,
                    'Status': sample['Status'],
                    'Summary': paraphrased['summary'],
                    'Description': paraphrased['description'],
                    'Priority': sample['Priority'],
                    'Team': sample['Team'],
                    'Augmented': 'Paraphrase'
                })
                
                time.sleep(delay_seconds)  # Rate limiting
            
            # Synthetic generation
            logger.info(f"  Generating {synthetic_count} synthetic tickets...")
            examples = module_df.sample(min(5, len(module_df))).to_dict('records')
            
            for _ in tqdm(range(synthetic_count), desc=f"Synthesizing {module}"):
                synthetic = self.generate_synthetic_ticket(examples, module)
                
                if synthetic:
                    augmented_records.append({
                        'Incident': f"AUG_{len(augmented_records)}",
                        'Service': 'SAP',
                        'Module': module,
                        'Status': 'Closed',
                        'Summary': synthetic['summary'],
                        'Description': synthetic['description'],
                        'Priority': 2,
                        'Team': 'SAP - ENZEN',
                        'Augmented': 'Synthetic'
                    })
                
                time.sleep(delay_seconds)  # Rate limiting
        
        # Combine original and augmented
        df_augmented = pd.DataFrame(augmented_records)
        df['Augmented'] = 'Original'
        df_final = pd.concat([df, df_augmented], ignore_index=True)
        
        logger.info(f"Augmentation complete: {len(df)} → {len(df_final)} records")
        
        return df_final
