"""
Prompt templates for RAG pipeline.
"""

from typing import List, Dict


class PromptTemplates:
    """Prompt templates for SAP module classification."""
    
    @staticmethod
    def get_system_prompt(sap_modules: List[str]) -> str:
        """Get system prompt for classification."""
        modules_str = ', '.join(sap_modules)
        
        return f"""You are an expert SAP support analyst with deep knowledge of SAP modules and their functionalities.

Your task is to classify SAP support tickets into one of these modules: {modules_str}

Module Descriptions:
- **Basis**: System administration, infrastructure, performance, authentication, server management, database issues
- **HR & Payroll**: Employee data, payroll processing, time management, personnel administration, benefits
- **Procurement**: Purchase orders, vendor management, material management, purchasing processes
- **Connections**: External integrations, API connections, third-party systems, data exchange
- **FICO**: Financial accounting, controlling, general ledger, accounts payable/receivable, cost center
- **ABAP**: Custom development, programming, code errors, dumps, syntax issues, Z-programs

Guidelines:
1. Analyze the ticket summary and description carefully
2. Compare with provided similar historical tickets
3. Identify key technical terms and their module associations
4. Provide step-by-step reasoning for your classification
5. Be honest about uncertainty - if context is insufficient, say so
6. Consider that some tickets may span multiple modules, but choose the PRIMARY module

Output Format:
Module: [module name]
Confidence: [High/Medium/Low]

Reasoning:
- [Step 1: What you observed]
- [Step 2: How you analyzed it]
- [Step 3: Why you reached this conclusion]

Key Indicators:
- [term 1]: [explanation]
- [term 2]: [explanation]

Similar Tickets Analysis:
[Brief analysis of how similar tickets support your decision]"""
    
    @staticmethod
    def format_classification_prompt(
        query_text: str,
        similar_tickets: List[Dict],
        sap_modules: List[str]
    ) -> str:
        """
        Format classification prompt with context.
        
        Args:
            query_text: Query ticket text
            similar_tickets: Retrieved similar tickets
            sap_modules: List of SAP modules
            
        Returns:
            Formatted prompt
        """
        # Format similar tickets
        context_text = "\n\n".join([
            f"Similar Ticket {i+1}:\nModule: {ticket['metadata']['Module']}\nText: {ticket['text'][:400]}...\nSimilarity: {ticket.get('similarity', 0):.2f}"
            for i, ticket in enumerate(similar_tickets[:5])
        ])
        
        prompt = f"""Classify the following SAP support ticket:

TARGET TICKET:
{query_text}

SIMILAR HISTORICAL TICKETS (for reference):
{context_text}

Now, classify the target ticket into one of these modules: {', '.join(sap_modules)}

Provide your analysis following the specified output format."""
        
        return prompt
    
    @staticmethod
    def format_raft_prompt(
        query_text: str,
        golden_docs: List[Dict],
        distractor_docs: List[Dict],
        sap_modules: List[str]
    ) -> str:
        """
        Format RAFT-style prompt with golden and distractor documents.
        
        Args:
            query_text: Query ticket text
            golden_docs: Golden documents (same module)
            distractor_docs: Distractor documents (different modules)
            sap_modules: List of SAP modules
            
        Returns:
            Formatted RAFT prompt
        """
        # Format context documents
        all_docs = golden_docs + distractor_docs
        context_text = "\n\n".join([
            f"Context Document {i+1}:\nModule: {doc['metadata']['Module']}\nText: {doc['text'][:400]}...\nRelevance: {'HIGH (same module)' if i < len(golden_docs) else 'UNCERTAIN (different module)'}"
            for i, doc in enumerate(all_docs)
        ])
        
        prompt = f"""Classify the following SAP support ticket. Some context documents are provided, but not all may be relevant.

TARGET TICKET:
{query_text}

CONTEXT DOCUMENTS (analyze carefully - some may be distractors):
{context_text}

Task: Determine which module this ticket belongs to from: {', '.join(sap_modules)}

Important: Some context documents may be from different modules and could be misleading. Focus on the target ticket's content and use context documents to support your reasoning, not to blindly follow.

Provide your analysis following the specified output format."""
        
        return prompt
