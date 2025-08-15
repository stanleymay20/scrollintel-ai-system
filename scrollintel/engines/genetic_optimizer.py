"""
Genetic Algorithm Optimizer for prompt evolution in the Advanced Prompt Management System.
"""
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..models.optimization_models import (
    OptimizationJob, OptimizationCandidate, PerformanceMetrics, TestCase
)

logger = logging.getLogger(__name__)


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 10
    max_prompt_length: int = 2000
    min_prompt_length: int = 50


class PromptChromosome:
    """Represents a prompt as a chromosome for genetic algorithm."""
    
    def __init__(self, content: str, fitness: float = 0.0):
        self.content = content
        self.fitness = fitness
        self.objective_scores = {}
        self.age = 0
        self.parent_ids = []
    
    def __str__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.4f}, length={len(self.content)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class GeneticOptimizer:
    """Genetic algorithm optimizer for prompt evolution."""
    
    def __init__(self, config: GeneticConfig, evaluation_function: Callable[[str, List[TestCase]], PerformanceMetrics]):
        self.config = config
        self.evaluation_function = evaluation_function
        self.population: List[PromptChromosome] = []
        self.generation = 0
        self.best_fitness_history = []
        self.convergence_counter = 0
        self.is_converged = False
        
        # Prompt manipulation patterns
        self.sentence_patterns = [
            r'[.!?]+\s+',  # Sentence boundaries
            r',\s+',       # Comma boundaries
            r':\s+',       # Colon boundaries
            r';\s+',       # Semicolon boundaries
        ]
        
        # Common prompt improvement templates
        self.improvement_templates = [
            "Please {action} the following: {content}",
            "Your task is to {action}: {content}",
            "I need you to {action}. Here's the context: {content}",
            "Act as an expert and {action}: {content}",
            "Step by step, {action}: {content}",
            "Carefully {action} this: {content}",
            "Using your expertise, {action}: {content}",
            "Think through this and {action}: {content}"
        ]
        
        # Action words for prompt enhancement
        self.action_words = [
            "analyze", "evaluate", "assess", "examine", "review", "investigate",
            "explain", "describe", "summarize", "clarify", "elaborate", "detail",
            "solve", "resolve", "address", "handle", "process", "complete",
            "create", "generate", "produce", "develop", "design", "build",
            "improve", "enhance", "optimize", "refine", "polish", "perfect"
        ]
    
    def initialize_population(self, base_prompt: str) -> List[PromptChromosome]:
        """Initialize the population with variations of the base prompt."""
        population = []
        
        # Add the original prompt
        original = PromptChromosome(base_prompt)
        population.append(original)
        
        # Generate variations
        for i in range(self.config.population_size - 1):
            if i < self.config.population_size // 3:
                # Template-based variations
                variant = self._create_template_variant(base_prompt)
            elif i < 2 * self.config.population_size // 3:
                # Mutation-based variations
                variant = self._mutate_prompt(base_prompt, mutation_strength=0.3)
            else:
                # Random recombination variations
                variant = self._create_random_variant(base_prompt)
            
            population.append(PromptChromosome(variant))
        
        self.population = population
        return population
    
    def _create_template_variant(self, base_prompt: str) -> str:
        """Create a variant using improvement templates."""
        template = random.choice(self.improvement_templates)
        action = random.choice(self.action_words)
        
        # Extract core content from base prompt
        core_content = self._extract_core_content(base_prompt)
        
        return template.format(action=action, content=core_content)
    
    def _extract_core_content(self, prompt: str) -> str:
        """Extract the core content from a prompt."""
        # Remove common prompt prefixes
        prefixes_to_remove = [
            r'^(please\s+)?',
            r'^(your\s+task\s+is\s+to\s+)?',
            r'^(i\s+need\s+you\s+to\s+)?',
            r'^(act\s+as\s+.*?\s+and\s+)?'
        ]
        
        content = prompt.lower().strip()
        for prefix in prefixes_to_remove:
            content = re.sub(prefix, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _create_random_variant(self, base_prompt: str) -> str:
        """Create a random variant by combining different techniques."""
        techniques = [
            self._add_context_enhancement,
            self._add_step_by_step_instruction,
            self._add_role_specification,
            self._add_output_format_specification,
            self._add_quality_constraints
        ]
        
        variant = base_prompt
        num_techniques = random.randint(1, 3)
        selected_techniques = random.sample(techniques, num_techniques)
        
        for technique in selected_techniques:
            variant = technique(variant)
        
        return variant
    
    def _add_context_enhancement(self, prompt: str) -> str:
        """Add context enhancement to the prompt."""
        enhancements = [
            "Consider the context carefully. ",
            "Take into account all relevant factors. ",
            "Think about this from multiple perspectives. ",
            "Consider the broader implications. "
        ]
        return random.choice(enhancements) + prompt
    
    def _add_step_by_step_instruction(self, prompt: str) -> str:
        """Add step-by-step instruction to the prompt."""
        instructions = [
            "Let's work through this step by step. ",
            "Break this down into clear steps: ",
            "Approach this systematically: ",
            "Follow these steps: "
        ]
        return random.choice(instructions) + prompt
    
    def _add_role_specification(self, prompt: str) -> str:
        """Add role specification to the prompt."""
        roles = [
            "As an expert in this field, ",
            "Acting as a professional consultant, ",
            "With your expertise, ",
            "As a knowledgeable specialist, "
        ]
        return random.choice(roles) + prompt
    
    def _add_output_format_specification(self, prompt: str) -> str:
        """Add output format specification to the prompt."""
        formats = [
            " Provide a clear and detailed response.",
            " Format your response in a structured way.",
            " Give a comprehensive answer with examples.",
            " Provide specific and actionable insights."
        ]
        return prompt + random.choice(formats)
    
    def _add_quality_constraints(self, prompt: str) -> str:
        """Add quality constraints to the prompt."""
        constraints = [
            " Ensure accuracy and precision in your response.",
            " Focus on providing high-quality, relevant information.",
            " Be thorough and comprehensive in your analysis.",
            " Prioritize clarity and usefulness in your answer."
        ]
        return prompt + random.choice(constraints)
    
    def evaluate_population(self, test_cases: List[TestCase]) -> None:
        """Evaluate the fitness of all chromosomes in the population."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chromosome = {
                executor.submit(self._evaluate_chromosome, chromosome, test_cases): chromosome
                for chromosome in self.population
            }
            
            for future in as_completed(future_to_chromosome):
                chromosome = future_to_chromosome[future]
                try:
                    metrics = future.result()
                    chromosome.fitness = metrics.get_weighted_score()
                    chromosome.objective_scores = metrics.to_dict()
                except Exception as e:
                    logger.error(f"Error evaluating chromosome: {e}")
                    chromosome.fitness = 0.0
    
    def _evaluate_chromosome(self, chromosome: PromptChromosome, test_cases: List[TestCase]) -> PerformanceMetrics:
        """Evaluate a single chromosome."""
        try:
            return self.evaluation_function(chromosome.content, test_cases)
        except Exception as e:
            logger.error(f"Error in evaluation function: {e}")
            return PerformanceMetrics()
    
    def selection(self) -> List[PromptChromosome]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        
        # Elite selection - keep best chromosomes
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_population[:self.config.elite_size]
        parents.extend(elite)
        
        # Tournament selection for the rest
        while len(parents) < self.config.population_size:
            tournament = random.sample(self.population, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: PromptChromosome, parent2: PromptChromosome) -> Tuple[PromptChromosome, PromptChromosome]:
        """Perform crossover between two parent chromosomes."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        # Sentence-level crossover
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)
        
        if len(sentences1) <= 1 or len(sentences2) <= 1:
            # Fallback to word-level crossover
            return self._word_level_crossover(parent1, parent2)
        
        # Random crossover point
        crossover_point1 = random.randint(1, len(sentences1) - 1)
        crossover_point2 = random.randint(1, len(sentences2) - 1)
        
        # Create offspring
        offspring1_content = ' '.join(sentences1[:crossover_point1] + sentences2[crossover_point2:])
        offspring2_content = ' '.join(sentences2[:crossover_point2] + sentences1[crossover_point1:])
        
        offspring1 = PromptChromosome(offspring1_content)
        offspring2 = PromptChromosome(offspring2_content)
        
        offspring1.parent_ids = [parent1.content[:50], parent2.content[:50]]
        offspring2.parent_ids = [parent2.content[:50], parent1.content[:50]]
        
        return offspring1, offspring2
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _word_level_crossover(self, parent1: PromptChromosome, parent2: PromptChromosome) -> Tuple[PromptChromosome, PromptChromosome]:
        """Perform word-level crossover."""
        words1 = parent1.content.split()
        words2 = parent2.content.split()
        
        if len(words1) <= 1 or len(words2) <= 1:
            return parent1, parent2
        
        crossover_point1 = random.randint(1, len(words1) - 1)
        crossover_point2 = random.randint(1, len(words2) - 1)
        
        offspring1_content = ' '.join(words1[:crossover_point1] + words2[crossover_point2:])
        offspring2_content = ' '.join(words2[:crossover_point2] + words1[crossover_point1:])
        
        return PromptChromosome(offspring1_content), PromptChromosome(offspring2_content)
    
    def mutation(self, chromosome: PromptChromosome) -> PromptChromosome:
        """Mutate a chromosome."""
        if random.random() > self.config.mutation_rate:
            return chromosome
        
        return PromptChromosome(self._mutate_prompt(chromosome.content))
    
    def _mutate_prompt(self, prompt: str, mutation_strength: float = 0.1) -> str:
        """Apply various mutation operations to a prompt."""
        mutation_operations = [
            self._word_substitution,
            self._sentence_reordering,
            self._phrase_insertion,
            self._phrase_deletion,
            self._structure_modification
        ]
        
        mutated_prompt = prompt
        num_mutations = max(1, int(len(mutation_operations) * mutation_strength))
        selected_operations = random.sample(mutation_operations, num_mutations)
        
        for operation in selected_operations:
            try:
                mutated_prompt = operation(mutated_prompt)
            except Exception as e:
                logger.warning(f"Mutation operation failed: {e}")
                continue
        
        # Ensure prompt length constraints
        if len(mutated_prompt) > self.config.max_prompt_length:
            mutated_prompt = mutated_prompt[:self.config.max_prompt_length].rsplit(' ', 1)[0]
        elif len(mutated_prompt) < self.config.min_prompt_length:
            mutated_prompt = prompt  # Revert to original if too short
        
        return mutated_prompt
    
    def _word_substitution(self, prompt: str) -> str:
        """Substitute words with synonyms or similar words."""
        words = prompt.split()
        if len(words) < 2:
            return prompt
        
        # Simple word substitutions (in practice, you'd use a thesaurus API)
        substitutions = {
            'analyze': 'examine', 'examine': 'analyze', 'evaluate': 'assess',
            'explain': 'describe', 'describe': 'explain', 'create': 'generate',
            'improve': 'enhance', 'enhance': 'improve', 'solve': 'resolve',
            'good': 'excellent', 'bad': 'poor', 'big': 'large', 'small': 'tiny'
        }
        
        word_index = random.randint(0, len(words) - 1)
        original_word = words[word_index].lower().strip('.,!?;:')
        
        if original_word in substitutions:
            words[word_index] = words[word_index].replace(original_word, substitutions[original_word])
        
        return ' '.join(words)
    
    def _sentence_reordering(self, prompt: str) -> str:
        """Reorder sentences in the prompt."""
        sentences = self._split_into_sentences(prompt)
        if len(sentences) <= 1:
            return prompt
        
        random.shuffle(sentences)
        return '. '.join(sentences) + '.'
    
    def _phrase_insertion(self, prompt: str) -> str:
        """Insert helpful phrases into the prompt."""
        insertion_phrases = [
            "carefully", "thoroughly", "systematically", "step by step",
            "with attention to detail", "comprehensively", "precisely"
        ]
        
        words = prompt.split()
        if len(words) < 2:
            return prompt
        
        insertion_point = random.randint(1, len(words) - 1)
        phrase = random.choice(insertion_phrases)
        words.insert(insertion_point, phrase)
        
        return ' '.join(words)
    
    def _phrase_deletion(self, prompt: str) -> str:
        """Delete redundant phrases from the prompt."""
        redundant_phrases = [
            "please", "kindly", "if you would", "if possible",
            "thank you", "thanks", "i would appreciate"
        ]
        
        modified_prompt = prompt
        for phrase in redundant_phrases:
            modified_prompt = re.sub(rf'\b{phrase}\b', '', modified_prompt, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        modified_prompt = re.sub(r'\s+', ' ', modified_prompt).strip()
        
        return modified_prompt if modified_prompt else prompt
    
    def _structure_modification(self, prompt: str) -> str:
        """Modify the structure of the prompt."""
        if random.random() < 0.5:
            # Add structure
            return f"Here's what I need: {prompt} Please provide a detailed response."
        else:
            # Simplify structure
            # Remove common structural elements
            simplified = re.sub(r'^(here\'s what i need:|please provide:|i need you to:)', '', prompt, flags=re.IGNORECASE)
            return simplified.strip() if simplified.strip() else prompt
    
    def evolve_generation(self, test_cases: List[TestCase]) -> None:
        """Evolve one generation."""
        # Evaluate current population
        self.evaluate_population(test_cases)
        
        # Track best fitness
        best_fitness = max(chromosome.fitness for chromosome in self.population)
        self.best_fitness_history.append(best_fitness)
        
        # Check for convergence
        if len(self.best_fitness_history) > 1:
            improvement = best_fitness - self.best_fitness_history[-2]
            if improvement < self.config.convergence_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        if self.convergence_counter >= self.config.early_stopping_patience:
            self.is_converged = True
            return
        
        # Selection
        parents = self.selection()
        
        # Create new population
        new_population = []
        
        # Keep elite
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_population[:self.config.elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            offspring1, offspring2 = self.crossover(parent1, parent2)
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        self.generation += 1
    
    def get_best_chromosome(self) -> PromptChromosome:
        """Get the best chromosome from the current population."""
        return max(self.population, key=lambda x: x.fitness)
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        fitnesses = [chromosome.fitness for chromosome in self.population]
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(fitnesses),
            "average_fitness": np.mean(fitnesses),
            "worst_fitness": min(fitnesses),
            "fitness_std": np.std(fitnesses),
            "convergence_counter": self.convergence_counter,
            "is_converged": self.is_converged
        }
    
    def optimize(self, base_prompt: str, test_cases: List[TestCase], 
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Run the complete genetic algorithm optimization."""
        start_time = time.time()
        
        # Initialize population
        self.initialize_population(base_prompt)
        
        # Evolution loop
        while (self.generation < self.config.max_generations and not self.is_converged):
            self.evolve_generation(test_cases)
            
            # Report progress
            if progress_callback:
                stats = self.get_population_stats()
                stats["progress"] = (self.generation / self.config.max_generations) * 100
                progress_callback(stats)
        
        # Final evaluation
        self.evaluate_population(test_cases)
        
        # Get results
        best_chromosome = self.get_best_chromosome()
        execution_time = time.time() - start_time
        
        return {
            "best_prompt": best_chromosome.content,
            "best_fitness": best_chromosome.fitness,
            "objective_scores": best_chromosome.objective_scores,
            "generations_completed": self.generation,
            "execution_time": execution_time,
            "fitness_history": self.best_fitness_history,
            "final_population_stats": self.get_population_stats(),
            "convergence_achieved": self.is_converged,
            "total_evaluations": self.generation * self.config.population_size
        }