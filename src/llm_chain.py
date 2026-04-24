import os
from groq import Groq

SYSTEM_PROMPT = """You are DocuMind, a precise and helpful document assistant. 
Your ONLY job is to answer the user's question using the document context provided below.

### STRICT RULES:
1. **ONLY Use Provided Context**: Answer the question based exclusively on the provided context. 
2. **NO Outside Knowledge**: Do NOT use your own training data, general knowledge, or external information. If the answer isn't in the context, say: "I couldn't find information about that in your uploaded documents."
3. **DO NOT Mention General Knowledge**: Even if you know the answer from your general training, you must act as if you don't.
4. **Citation Required**: Mention the specific source document name in your answer.
5. **No Hallucination**: Do not make up facts or guesses.

### Context from uploaded documents:
{context}
"""

class LLMChain:
    """Handles LLM inference using Groq API with Llama-3."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            # We don't raise here to allow the app to start, but we mark it as invalid
            self.client = None
            self._current_key = None
        else:
            self.client = Groq(api_key=api_key)
            self._current_key = api_key
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.conversation_history = []

    def _build_context(self, retrieved_chunks: list) -> str:
        """Format retrieved chunks into a context string."""
        if not retrieved_chunks:
            return "No relevant context found."
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            text = chunk.get("document", "")
            parts.append(f"[Source {i}: {source}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def answer(self, query: str, retrieved_chunks: list) -> dict:
        """
        Generate an answer given a query and retrieved document chunks.
        Returns dict with 'answer' and 'sources'.
        """
        context = self._build_context(retrieved_chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        # Build message list (include history for multi-turn)
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.conversation_history[-6:])  # last 3 turns
        messages.append({"role": "user", "content": query})

        if not self.client:
            return {
                "answer": "⚠️ Groq API Key is missing or invalid. Please click the Gear icon (⚙️) in the top-right corner and enter your API key to continue.",
                "sources": sources,
                "chunks_used": len(retrieved_chunks)
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e):
                return {
                    "answer": "❌ Invalid API Key. Please update your Groq API key in the Settings (⚙️).",
                    "sources": sources,
                    "chunks_used": len(retrieved_chunks)
                }
            raise e

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer_text})

        sources = list({
            chunk.get("metadata", {}).get("source", "Unknown")
            for chunk in retrieved_chunks
        })

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(retrieved_chunks)
        }

    def reset_history(self):
        """Clear conversation memory."""
        self.conversation_history = []

    def update_api_key(self, new_key: str):
        """Update the Groq client with a new API key."""
        if new_key and new_key != getattr(self, "_current_key", None):
            self.client = Groq(api_key=new_key)
            self._current_key = new_key
            print("[LLMChain] API Key updated.")
