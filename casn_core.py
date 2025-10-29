# casn_core.py - The atomic unit of the new AI revolution
# Context-Aware Semantic Networks: Beyond Neural Networks to Cognitive Architecture

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class Knowledge:
    """A single piece of knowledge with provenance - the foundation of explainable AI"""
    content: Any
    confidence: float  # 0.0 to 1.0
    source: str
    timestamp: datetime
    usage_count: int = 0
    success_rate: float = 1.0
    
    def get_hash(self) -> str:
        """Unique identifier for deduplication"""
        return hashlib.md5(str(self.content).encode()).hexdigest()

@dataclass  
class Context:
    """Rich contextual state - the key differentiator from neural networks"""
    goal: str
    constraints: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CASN:
    """Context-Aware Semantic Node - The new computational primitive revolutionizing AI"""
    
    def __init__(self, specialty: str = "general"):
        self.id = hashlib.md5(specialty.encode()).hexdigest()[:8]
        self.specialty = specialty
        
        # POSSESSION - What this CASN knows (explicit knowledge store)
        self.knowledge_store: Dict[str, Knowledge] = {}
        self.skill_patterns: Dict[str, callable] = {}
        
        # ACQUISITION - Learning parameters that adapt
        self.learning_threshold = 0.1  # Minimum relevance to acquire
        self.max_knowledge_items = 1000  # Prevents infinite growth
        
        # APPLICATION - Usage tracking for continuous improvement
        self.query_history: List[tuple] = []
        self.collaboration_network: List[str] = []
        
        print(f"✨ CASN-{self.id} initialized with specialty: {specialty}")
        
    def acquire(self, content: Any, context: Context, source: str = "direct") -> bool:
        """
        CRITICAL: This is where learning happens - unlike backpropagation, this is conscious acquisition
        Returns True if knowledge was acquired, False if rejected
        """
        # Calculate relevance to specialty and context
        relevance = self._calculate_relevance(content, context)
        
        if relevance < self.learning_threshold:
            return False
            
        # Create knowledge object with full provenance
        knowledge = Knowledge(
            content=content,
            confidence=relevance,
            source=source,
            timestamp=datetime.now()
        )
        
        # Deduplicate - no redundant storage
        k_hash = knowledge.get_hash()
        if k_hash in self.knowledge_store:
            # Update confidence if new info is more confident
            existing = self.knowledge_store[k_hash]
            existing.confidence = max(existing.confidence, relevance)
            existing.usage_count += 1
            return True
            
        # Add to store
        self.knowledge_store[k_hash] = knowledge
        
        # Prune if over capacity (evolutionary pressure)
        if len(self.knowledge_store) > self.max_knowledge_items:
            self._prune_knowledge()
            
        return True
    
    def apply(self, query: str, context: Context) -> Optional[Dict[str, Any]]:
        """
        CRITICAL: This is where intelligence manifests - explicit reasoning with full traceability
        Returns best answer with complete reasoning trace
        """
        # Search for relevant knowledge
        relevant = self._search_knowledge(query, context)
        
        if not relevant:
            return None
            
        # Synthesize answer from multiple knowledge pieces
        answer = self._synthesize(relevant, query, context)
        
        # Update usage statistics (reinforcement learning)
        for k in relevant:
            k.usage_count += 1
            
        # Track query for continuous learning
        self.query_history.append((query, answer, datetime.now()))
        
        return {
            "answer": answer,
            "confidence": self._aggregate_confidence(relevant),
            "sources": [k.source for k in relevant],
            "reasoning": self._explain_reasoning(relevant, query),
            "knowledge_used": len(relevant),
            "casn_id": self.id
        }
    
    def collaborate(self, query: str, context: Context, 
                   network: List['CASN']) -> Dict[str, Any]:
        """
        REVOLUTIONARY: Multiple CASNs solve together - emergence through collaboration
        This is where collective intelligence happens
        """
        # Try own knowledge first
        own_answer = self.apply(query, context)
        
        # Query other CASNs in parallel (distributed intelligence)
        other_answers = []
        for casn in network:
            if casn.id == self.id:
                continue
            answer = casn.apply(query, context)
            if answer:
                other_answers.append((casn.specialty, answer))
                
        # Integrate perspectives through weighted consensus
        if not other_answers and not own_answer:
            return {"answer": None, "reason": "No knowledge found in network"}
            
        final_answer = self._integrate_answers(
            own_answer, other_answers, query, context
        )
        
        # Learn from collaboration (peer learning)
        for specialty, answer in other_answers:
            if answer["confidence"] > 0.7:
                self._maybe_acquire_from_peer(answer, context, specialty)
                
        return final_answer
    
    def _calculate_relevance(self, content: Any, context: Context) -> float:
        """Calculate how relevant content is to this CASN's specialty and current context"""
        content_str = str(content).lower()
        
        # Check specialty match
        specialty_match = 0.5 if self.specialty.lower() in content_str else 0.0
        
        # Check goal alignment
        goal_words = context.goal.lower().split()
        goal_match = sum(1 for word in goal_words if word in content_str) / max(len(goal_words), 1)
        
        # Contextual relevance
        context_bonus = 0.2 if any(constraint.lower() in content_str for constraint in context.constraints) else 0.0
        
        return min(1.0, specialty_match + goal_match + context_bonus)
    
    def _search_knowledge(self, query: str, context: Context) -> List[Knowledge]:
        """Find relevant knowledge for query using semantic similarity"""
        query_lower = query.lower()
        
        matches = []
        for k in self.knowledge_store.values():
            content_str = str(k.content).lower()
            
            # Calculate relevance score combining multiple factors
            word_matches = sum(1 for word in query_lower.split() if word in content_str)
            recency_bonus = max(0, (7 - (datetime.now() - k.timestamp).days)) / 7 * 0.2
            usage_bonus = min(k.usage_count * 0.1, 0.3)
            
            relevance = word_matches * k.confidence * k.success_rate + recency_bonus + usage_bonus
            
            if relevance > 0:
                matches.append((relevance, k))
                
        # Return top matches sorted by relevance
        matches.sort(reverse=True)
        return [k for _, k in matches[:5]]
    
    def _synthesize(self, knowledge_items: List[Knowledge], 
                   query: str, context: Context) -> str:
        """Combine multiple knowledge pieces into coherent answer"""
        if not knowledge_items:
            return f"I don't have knowledge about '{query}' in my {self.specialty} domain yet."
            
        # Enhanced synthesis considering context and confidence
        high_conf_items = [k for k in knowledge_items if k.confidence > 0.8]
        if high_conf_items:
            primary_content = str(high_conf_items[0].content)
        else:
            primary_content = str(knowledge_items[0].content)
            
        # Add supporting information
        supporting_info = []
        for k in knowledge_items[1:3]:  # Use top 2 supporting pieces
            supporting_info.append(f"Additionally: {str(k.content)}")
            
        result = primary_content
        if supporting_info:
            result += " " + " ".join(supporting_info)
            
        return result
    
    def _aggregate_confidence(self, items: List[Knowledge]) -> float:
        """Calculate overall confidence using weighted average"""
        if not items:
            return 0.0
        
        # Weight by usage count and recency
        total_weight = 0
        weighted_confidence = 0
        
        for k in items:
            weight = (k.usage_count + 1) * k.success_rate
            weighted_confidence += k.confidence * weight
            total_weight += weight
            
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _explain_reasoning(self, items: List[Knowledge], query: str) -> str:
        """Generate detailed explanation of reasoning process"""
        if not items:
            return f"No relevant knowledge found for '{query}' in {self.specialty} domain"
            
        explanations = []
        for i, k in enumerate(items):
            age_days = (datetime.now() - k.timestamp).days
            explanations.append(
                f"[{i+1}] Used knowledge from {k.source} "
                f"(confidence: {k.confidence:.2f}, "
                f"used {k.usage_count} times, "
                f"{age_days} days old)"
            )
        
        reasoning = f"Reasoning for {self.specialty} CASN-{self.id}: " + "; ".join(explanations)
        return reasoning
    
    def _integrate_answers(self, own: Optional[Dict], 
                          others: List[tuple], 
                          query: str, context: Context) -> Dict[str, Any]:
        """Integrate multiple CASN answers using sophisticated consensus"""
        
        candidates = []
        if own:
            candidates.append(("self", own))
        candidates.extend(others)
        
        if not candidates:
            return {"answer": None, "confidence": 0.0}
            
        # Weighted voting based on confidence and specialty relevance
        query_terms = query.lower().split()
        best_score = 0
        best_answer = None
        
        for specialty, answer in candidates:
            # Base score from confidence
            score = answer["confidence"]
            
            # Bonus for specialty alignment
            if any(term in specialty.lower() for term in query_terms):
                score += 0.3
                
            # Bonus for knowledge breadth
            if "knowledge_used" in answer and answer["knowledge_used"] > 1:
                score += 0.1
                
            if score > best_score:
                best_score = score
                best_answer = (specialty, answer)
        
        if best_answer:
            specialty, answer = best_answer
            return {
                "answer": answer["answer"],
                "confidence": answer["confidence"],
                "primary_source": specialty,
                "contributors": [c[0] for c in candidates],
                "reasoning": f"Network consensus: consulted {len(candidates)} CASNs, "
                           f"best answer from {specialty}-specialized CASN. " + 
                           answer.get("reasoning", ""),
                "collaboration_bonus": len(candidates) > 1
            }
        else:
            return {"answer": "Network failed to reach consensus", "confidence": 0.0}
    
    def _maybe_acquire_from_peer(self, peer_answer: Dict, context: Context, peer_specialty: str):
        """Learn from successful peer responses (peer learning mechanism)"""
        if peer_answer["confidence"] > 0.8:
            self.acquire(
                peer_answer["answer"],
                context,
                source=f"peer_learning_from_{peer_specialty}"
            )
    
    def _prune_knowledge(self):
        """Remove least useful knowledge using evolutionary pressure"""
        # Score by: confidence * success_rate * usage_count * recency
        scored = []
        for k_hash, k in self.knowledge_store.items():
            age_penalty = max(0.1, 1 - (datetime.now() - k.timestamp).days / 30)
            score = k.confidence * k.success_rate * (k.usage_count + 1) * age_penalty
            scored.append((score, k_hash, k))
            
        scored.sort(reverse=True)
        
        # Keep top 80% of knowledge
        keep_count = int(self.max_knowledge_items * 0.8)
        self.knowledge_store = {
            k_hash: k for _, k_hash, k in scored[:keep_count]
        }
        
        print(f"🧹 CASN-{self.id} pruned knowledge store to {keep_count} items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Comprehensive diagnostic information"""
        return {
            "id": self.id,
            "specialty": self.specialty,
            "knowledge_count": len(self.knowledge_store),
            "total_queries": len(self.query_history),
            "avg_confidence": sum(k.confidence for k in self.knowledge_store.values()) / 
                            max(len(self.knowledge_store), 1),
            "most_used_knowledge": max(self.knowledge_store.values(), 
                                     key=lambda k: k.usage_count, 
                                     default=None),
            "learning_threshold": self.learning_threshold,
            "collaboration_count": len(self.collaboration_network)
        }