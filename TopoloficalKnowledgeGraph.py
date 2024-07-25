import networkx as nx
from typing import List, Dict, Any
import requests
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TopologicalKnowledgeGraph:
    def __init__(self, small_llm_name: str, large_llm_name: str, search_api_key: str):
        self.graph = nx.DiGraph()
        self.small_llm = self.load_small_llm(small_llm_name)
        self.large_llm = self.load_large_llm(large_llm_name)
        self.search_api_key = search_api_key
        
    def load_small_llm(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model}
    
    def load_large_llm(self, model_name: str):
        # Implement loading of the large LLM (e.g., using an API)
        pass
    
    def search_engine_query(self, query: str) -> List[Dict[str, Any]]:
        # Implement search engine API call
        pass
    
    def generate_small_llm_response(self, prompt: str) -> str:
        inputs = self.small_llm["tokenizer"](prompt, return_tensors="pt")
        outputs = self.small_llm["model"].generate(**inputs, max_length=100)
        return self.small_llm["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    
    def generate_large_llm_response(self, prompt: str) -> str:
        # Implement large LLM API call
        pass
    
    def create_knowledge_node(self, topic: str):
        if topic not in self.graph.nodes:
            search_results = self.search_engine_query(topic)
            rag_prompt = f"Topic: {topic}\nSearch results: {search_results}\nSummarize key information:"
            small_llm_summary = self.generate_small_llm_response(rag_prompt)
            self.graph.add_node(topic, summary=small_llm_summary)
    
    def create_edge(self, source: str, target: str, relation: str):
        self.graph.add_edge(source, target, relation=relation)
    
    def expand_graph(self, root_topic: str, depth: int):
        self.create_knowledge_node(root_topic)
        if depth > 0:
            related_topics_prompt = f"List 3 related topics to '{root_topic}':"
            related_topics = self.generate_small_llm_response(related_topics_prompt).split(", ")
            for topic in related_topics:
                self.create_knowledge_node(topic)
                self.create_edge(root_topic, topic, "related_to")
                self.expand_graph(topic, depth - 1)
    
    def grade_large_llm_knowledge(self, topic: str) -> float:
        node_data = self.graph.nodes[topic]
        question_prompt = f"Generate a question about {topic} based on this summary: {node_data['summary']}"
        question = self.generate_small_llm_response(question_prompt)
        large_llm_answer = self.generate_large_llm_response(question)
        grading_prompt = f"Question: {question}\nExpected answer summary: {node_data['summary']}\nLarge LLM answer: {large_llm_answer}\nGrade the answer on a scale of 0 to 1:"
        grade = float(self.generate_small_llm_response(grading_prompt))
        return grade
    
    def update_node_confidence(self, topic: str):
        grade = self.grade_large_llm_knowledge(topic)
        self.graph.nodes[topic]['confidence'] = grade
    
    def get_node_confidence(self, topic: str) -> float:
        return self.graph.nodes[topic].get('confidence', 0.0)
    
    def predict_hallucination_risk(self, topic: str) -> str:
        confidence = self.get_node_confidence(topic)
        if confidence < 0.3:
            return "High risk of hallucination"
        elif confidence < 0.7:
            return "Moderate risk of hallucination"
        else:
            return "Low risk of hallucination"
    
    def query_with_confidence(self, query: str) -> Dict[str, Any]:
        large_llm_response = self.generate_large_llm_response(query)
        topic_extraction_prompt = f"Extract the main topic from this query: {query}"
        main_topic = self.generate_small_llm_response(topic_extraction_prompt)
        
        if main_topic not in self.graph.nodes:
            self.expand_graph(main_topic, depth=2)
        
        self.update_node_confidence(main_topic)
        hallucination_risk = self.predict_hallucination_risk(main_topic)
        
        return {
            "response": large_llm_response,
            "confidence": self.get_node_confidence(main_topic),
            "hallucination_risk": hallucination_risk
        }

    def visualize_knowledge_graph(self, figsize=(20, 20), node_size=3000, font_size=8):
        """
        Create a visual representation of the knowledge graph with accuracy information.
        
        :param figsize: Tuple specifying the figure size
        :param node_size: Size of the nodes in the graph
        :param font_size: Font size for node labels
        """
        # Create a spring layout for the graph
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Prepare node colors based on confidence scores
        node_colors = []
        for node in self.graph.nodes():
            confidence = self.get_node_confidence(node)
            # Map confidence score to a color (red for low, yellow for medium, green for high)
            color = plt.cm.RdYlGn(confidence)
            node_colors.append(color)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_size, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        
        # Add labels to the nodes
        labels = {}
        for node in self.graph.nodes():
            confidence = self.get_node_confidence(node)
            labels[node] = f"{node}\n({confidence:.2f})"
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=font_size, font_weight="bold", ax=ax)
        
        # Add a color bar to show the confidence scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Confidence Score', rotation=270, labelpad=25)
        
        # Remove axis
        ax.set_axis_off()
        
        # Set the title
        plt.title("Topological Knowledge Graph with Accuracy Information", fontsize=16)
        
        # Show the plot
        plt.tight_layout()
        plt.show()

    def build_knowledge_library(self, topics: List[str], depth: int = 2):
        """
        Build a library of questions and answers for given topics.
        
        :param topics: List of initial topics to explore
        :param depth: Depth of graph expansion for each topic
        """
        for topic in topics:
            self.expand_graph(topic, depth)
            
        for node in self.graph.nodes():
            question_prompt = f"Generate a question about {node}:"
            question = self.generate_small_llm_response(question_prompt)
            answer = self.generate_large_llm_response(question)
            self.graph.nodes[node]['question'] = question
            self.graph.nodes[node]['answer'] = answer
            self.update_node_confidence(node)

    def update_node_confidence(self, topic: str):
        """
        Update the confidence score for a given topic based on the large LLM's answer.
        """
        node_data = self.graph.nodes[topic]
        question = node_data.get('question', '')
        large_llm_answer = node_data.get('answer', '')
        expected_summary = node_data.get('summary', '')
        
        grading_prompt = f"Question: {question}\nExpected answer summary: {expected_summary}\nLarge LLM answer: {large_llm_answer}\nGrade the answer on a scale of 0 to 1:"
        grade = float(self.generate_small_llm_response(grading_prompt))
        self.graph.nodes[topic]['confidence'] = grade

    def generate_subject_questions(self, subject: str, num_questions: int = 10) -> List[Dict[str, str]]:
        """
        Generate a list of test questions for a given subject.
        
        :param subject: The subject area (e.g., 'math', 'programming', 'language')
        :param num_questions: Number of questions to generate
        :return: List of dictionaries containing questions and expected answers
        """
        questions = []
        for _ in range(num_questions):
            prompt = f"Generate a {subject} question with a clear, concise answer:"
            response = self.generate_small_llm_response(prompt)
            question, answer = response.split("\nAnswer: ")
            questions.append({"question": question.strip(), "expected_answer": answer.strip()})
        return questions

    def evaluate_response(self, expected: str, actual: str) -> float:
        """
        Evaluate the similarity between the expected and actual answers.
        
        :param expected: The expected answer
        :param actual: The actual answer provided by the large LLM
        :return: A score between 0 and 1 indicating the similarity
        """
        prompt = f"Compare these two answers and rate their similarity on a scale from 0 to 1, where 1 is identical in meaning and 0 is completely different:\nExpected: {expected}\nActual: {actual}\nSimilarity score:"
        score = float(self.generate_small_llm_response(prompt))
        return score

    def calculate_subject_score(self, subject: str, num_questions: int = 10) -> Dict[str, Any]:
        """
        Calculate a quantifiable metric score for a given subject.
        
        :param subject: The subject area to evaluate
        :param num_questions: Number of test questions to use
        :return: A dictionary containing the overall score and detailed results
        """
        questions = self.generate_subject_questions(subject, num_questions)
        total_score = 0
        detailed_results = []

        for q in questions:
            large_llm_answer = self.generate_large_llm_response(q['question'])
            question_score = self.evaluate_response(q['expected_answer'], large_llm_answer)
            total_score += question_score
            detailed_results.append({
                "question": q['question'],
                "expected_answer": q['expected_answer'],
                "llm_answer": large_llm_answer,
                "score": question_score
            })

        average_score = total_score / num_questions
        return {
            "subject": subject,
            "overall_score": average_score,
            "num_questions": num_questions,
            "detailed_results": detailed_results
        }

    def evaluate_multiple_subjects(self, subjects: List[str], num_questions: int = 10) -> Dict[str, float]:
        """
        Evaluate the LLM's performance across multiple subjects.
        
        :param subjects: List of subjects to evaluate
        :param num_questions: Number of questions per subject
        :return: Dictionary of subjects and their scores
        """
        subject_scores = {}
        for subject in subjects:
            result = self.calculate_subject_score(subject, num_questions)
            subject_scores[subject] = result['overall_score']
        return subject_scores


     def visualize_subject_performance(self, subject_scores: Dict[str, float]):
        """
        Create a bar chart to visualize the LLM's performance across different subjects.
        
        :param subject_scores: Dictionary of subjects and their scores
        """
        subjects = list(subject_scores.keys())
        scores = list(subject_scores.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(subjects, scores, color='skyblue')
        plt.title("LLM Performance Across Subjects", fontsize=16)
        plt.xlabel("Subjects", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def generate_performance_report(self, subjects: List[str], num_questions: int = 10) -> str:
        """
        Generate a detailed performance report for multiple subjects.
        
        :param subjects: List of subjects to evaluate
        :param num_questions: Number of questions per subject
        :return: A formatted string containing the performance report
        """
        report = "LLM Performance Report\n"
        report += "=====================\n\n"

        subject_scores = self.evaluate_multiple_subjects(subjects, num_questions)
        
        for subject, score in subject_scores.items():
            report += f"{subject.capitalize()} Performance:\n"
            report += f"Overall Score: {score:.2f}\n"
            report += f"Number of test questions: {num_questions}\n"
            report += "\nStrengths and Weaknesses:\n"
            
            if score > 0.8:
                report += f"- Excellent performance in {subject}\n"
            elif score > 0.6:
                report += f"- Good performance in {subject}, but room for improvement\n"
            else:
                report += f"- Needs significant improvement in {subject}\n"
            
            report += "\n" + "="*30 + "\n\n"

        average_score = sum(subject_scores.values()) / len(subject_scores)
        report += f"Overall Average Score Across All Subjects: {average_score:.2f}\n\n"

        report += "Recommendations:\n"
        if average_score > 0.8:
            report += "- The LLM shows strong performance across most subjects.\n"
            report += "- Focus on maintaining high quality and possibly expanding to more specialized topics.\n"
        elif average_score > 0.6:
            report += "- The LLM performs well but has room for improvement in some areas.\n"
            report += "- Consider additional training or fine-tuning in lower-scoring subjects.\n"
        else:
            report += "- The LLM needs significant improvement across multiple subjects.\n"
            report += "- Recommend comprehensive review and retraining, especially in the lowest-scoring areas.\n"

        return report
        
# Usage example
# tkg = TopologicalKnowledgeGraph("microsoft/phi-3", "gpt-3.5-turbo", "your_search_api_key")
# result = tkg.query_with_confidence("What is the capital of France?")
# print(result)


# Build the knowledge library
# initial_topics = ["Artificial Intelligence", "Climate Change", "Quantum Computing"]
# tkg.build_knowledge_library(initial_topics, depth=2)

# # Visualize the graph
# tkg.visualize_knowledge_graph()

# Initialize the graph
tkg = TopologicalKnowledgeGraph("microsoft/phi-2", "gpt-3.5-turbo", "your_search_api_key")

# Define subjects to evaluate
subjects = ["math", "programming", "language comprehension", "history", "science"]

# Generate and print the performance report
report = tkg.generate_performance_report(subjects, num_questions=5)
print(report)

# Visualize the performance across subjects
subject_scores = tkg.evaluate_multiple_subjects(subjects, num_questions=5)
tkg.visualize_subject_performance(subject_scores)
