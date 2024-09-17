import os
from typing import List, Tuple
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from litellm import completion

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

class DualLLMReasoning:
    def __init__(self, model1: str, model2: str, max_iterations: int = 5):
        self.model1 = model1
        self.model2 = model2
        self.max_iterations = max_iterations
        
        self.answerer_template = PromptTemplate(
            input_variables=["question", "previous_answer"],
            template="Question: {question}\nPrevious answer: {previous_answer}\nProvide a more comprehensive answer:"
        )
        
        self.critic_template = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nAnswer: {answer}\nAct as a devil's advocate. What aspects of this answer need improvement or further exploration?"
        )
        
        self.answerer_chain = LLMChain(llm=OpenAI(model_name=self.model1), prompt=self.answerer_template)
        self.critic_chain = LLMChain(llm=OpenAI(model_name=self.model2), prompt=self.critic_template)
    
    def get_completion(self, model: str, messages: List[dict]) -> str:
        response = completion(model=model, messages=messages)
        return response.choices[0].message.content
    
    def reason(self, question: str) -> Tuple[str, List[str]]:
        conversation = []
        current_answer = ""
        
        for _ in range(self.max_iterations):
            # Answerer provides or improves the answer
            answerer_response = self.answerer_chain.run(question=question, previous_answer=current_answer)
            conversation.append(f"Answerer: {answerer_response}")
            current_answer = answerer_response
            
            # Critic evaluates the answer
            critic_response = self.critic_chain.run(question=question, answer=current_answer)
            conversation.append(f"Critic: {critic_response}")
            
            # Check if the critic is satisfied
            if "satisfactory" in critic_response.lower() or "comprehensive" in critic_response.lower():
                break
        
        # Generate final concise answer
        final_prompt = f"Summarize the following answer concisely:\n\n{current_answer}"
        final_answer = self.get_completion(self.model1, [{"role": "user", "content": final_prompt}])
        
        return final_answer, conversation

# Example usage
if __name__ == "__main__":
    reasoning_network = DualLLMReasoning("gpt-3.5-turbo", "gpt-4")
    question = "What are the potential implications of artificial general intelligence (AGI) on society?"
    
    final_answer, conversation = reasoning_network.reason(question)
    
    print("Conversation:")
    for turn in conversation:
        print(turn)
        print()
    
    print("Final Answer:")
    print(final_answer)
