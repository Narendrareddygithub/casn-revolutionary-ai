# casn_network.py - Advanced Network Management for Scaling CASN Intelligence
# Revolutionary distributed cognitive architecture

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict
from casn_core import CASN, Context, Knowledge

@dataclass
class NetworkMetrics:
    """Comprehensive network performance tracking"""
    total_nodes: int = 0
    active_collaborations: int = 0
    knowledge_distribution: Dict[str, int] = field(default_factory=dict)
    average_response_time: float = 0.0
    success_rate: float = 0.0
    emergent_solutions: int = 0  # Problems solved only through collaboration
    peer_learning_events: int = 0
    network_efficiency: float = 0.0

class CASNNetwork:
    """Advanced CASN Network Management System
    
    Manages large-scale CASN networks with:
    - Dynamic node discovery and registration
    - Load balancing across specialized CASNs
    - Network-wide knowledge synchronization
    - Performance monitoring and optimization
    - Emergent behavior tracking
    """
    
    def __init__(self, network_id: str = "default"):
        self.network_id = network_id
        self.nodes: Dict[str, CASN] = {}  # CASN_ID -> CASN instance
        self.specialization_map: Dict[str, List[str]] = defaultdict(list)  # specialty -> [CASN_IDs]
        self.collaboration_history: List[Dict[str, Any]] = []
        self.metrics = NetworkMetrics()
        self.knowledge_graph: Dict[str, List[str]] = defaultdict(list)  # topic -> [related_topics]
        self.load_balancer: Dict[str, int] = defaultdict(int)  # CASN_ID -> current_load
        
        print(f"🌐 CASN Network '{network_id}' initialized")
    
    def register_casn(self, casn: CASN, capabilities: List[str] = None) -> bool:
        """Register a CASN node with the network
        
        Args:
            casn: CASN instance to register
            capabilities: List of additional capabilities beyond specialty
            
        Returns:
            True if registration successful
        """
        if casn.id in self.nodes:
            print(f"⚠️ CASN-{casn.id} already registered")
            return False
        
        # Register the node
        self.nodes[casn.id] = casn
        self.specialization_map[casn.specialty].append(casn.id)
        
        # Register additional capabilities
        if capabilities:
            for capability in capabilities:
                self.specialization_map[capability].append(casn.id)
        
        # Initialize load tracking
        self.load_balancer[casn.id] = 0
        
        # Update metrics
        self.metrics.total_nodes = len(self.nodes)
        self.metrics.knowledge_distribution[casn.specialty] = len(casn.knowledge_store)
        
        print(f"✅ CASN-{casn.id} ({casn.specialty}) registered to network")
        return True
    
    def find_experts(self, query: str, context: Context, max_experts: int = 5) -> List[CASN]:
        """Find most relevant CASNs for a query using intelligent matching
        
        Args:
            query: The question or problem to solve
            context: Context information for relevance scoring
            max_experts: Maximum number of experts to return
            
        Returns:
            List of most relevant CASN experts, load-balanced
        """
        query_lower = query.lower()
        context_terms = context.goal.lower().split() + [c.lower() for c in context.constraints]
        
        # Score each CASN for relevance
        relevance_scores = []
        
        for casn_id, casn in self.nodes.items():
            # Base relevance from specialty match
            specialty_match = 1.0 if casn.specialty.lower() in query_lower else 0.0
            
            # Context alignment bonus
            context_match = sum(1 for term in context_terms 
                              if term in casn.specialty.lower()) / max(len(context_terms), 1)
            
            # Knowledge relevance (how much relevant knowledge this CASN has)
            knowledge_relevance = len([k for k in casn.knowledge_store.values() 
                                     if any(word in str(k.content).lower() 
                                           for word in query_lower.split())]) / max(len(casn.knowledge_store), 1)
            
            # Load balancing penalty (prefer less loaded CASNs)
            load_penalty = self.load_balancer[casn_id] * 0.1
            
            # Historical success rate with similar queries
            success_bonus = casn.get_stats()['avg_confidence'] * 0.3
            
            total_score = specialty_match + context_match + knowledge_relevance + success_bonus - load_penalty
            
            if total_score > 0.1:  # Minimum relevance threshold
                relevance_scores.append((total_score, casn))
        
        # Sort by relevance and return top experts
        relevance_scores.sort(reverse=True)
        experts = [casn for _, casn in relevance_scores[:max_experts]]
        
        # Update load balancing
        for casn in experts:
            self.load_balancer[casn.id] += 1
        
        return experts
    
    def collaborative_solve(self, query: str, context: Context, 
                          lead_specialty: str = None) -> Dict[str, Any]:
        """Solve complex problems using network-wide collaboration
        
        Args:
            query: Problem to solve
            context: Context for the problem
            lead_specialty: Optional specialty to lead the collaboration
            
        Returns:
            Comprehensive solution with network-wide insights
        """
        start_time = datetime.now()
        
        # Find expert CASNs for this problem
        if lead_specialty and lead_specialty in self.specialization_map:
            # Use specified lead specialty
            lead_casn_id = self.specialization_map[lead_specialty][0]
            lead_casn = self.nodes[lead_casn_id]
            experts = self.find_experts(query, context, max_experts=4)
            if lead_casn not in experts:
                experts.insert(0, lead_casn)
        else:
            # Auto-select best experts
            experts = self.find_experts(query, context, max_experts=5)
        
        if not experts:
            return {"answer": None, "reason": "No relevant experts found in network"}
        
        print(f"🤝 Collaborating: {[f'{e.specialty}-{e.id}' for e in experts]}")
        
        # Lead CASN attempts solution first
        lead_casn = experts[0]
        primary_solution = lead_casn.apply(query, context)
        
        # Get perspectives from other experts
        expert_solutions = []
        for expert in experts[1:]:
            solution = expert.apply(query, context)
            if solution:
                expert_solutions.append((expert.specialty, expert.id, solution))
        
        # Synthesize collaborative solution
        if not primary_solution and not expert_solutions:
            # No individual solutions found - try network emergence
            emergent_solution = self._attempt_emergent_solution(query, context, experts)
            if emergent_solution:
                self.metrics.emergent_solutions += 1
                return emergent_solution
            else:
                return {"answer": None, "reason": "Network could not solve problem"}
        
        # Integrate solutions using advanced consensus
        final_solution = self._integrate_network_solutions(
            primary_solution, expert_solutions, query, context, experts
        )
        
        # Track collaboration metrics
        response_time = (datetime.now() - start_time).total_seconds()
        self._update_collaboration_metrics(query, context, experts, final_solution, response_time)
        
        # Enable peer learning across network
        if final_solution['confidence'] > 0.8:
            self._propagate_successful_knowledge(final_solution, context, experts)
        
        return final_solution
    
    def _attempt_emergent_solution(self, query: str, context: Context, 
                                 experts: List[CASN]) -> Optional[Dict[str, Any]]:
        """Attempt to find emergent solution when individual CASNs fail
        
        This is where true network intelligence emerges - combining partial
        knowledge from multiple CASNs to solve problems none could solve alone
        """
        print("⚡ Attempting emergent network solution...")
        
        # Gather partial knowledge from all experts
        partial_knowledge = []
        for expert in experts:
            # Get knowledge items that have any relevance to query
            relevant_items = expert._search_knowledge(query, context)
            if relevant_items:
                partial_knowledge.extend([(expert.specialty, item) for item in relevant_items[:2]])
        
        if len(partial_knowledge) < 2:
            return None
        
        # Attempt to synthesize emergent solution
        combined_content = []
        combined_confidence = 0
        sources = []
        
        for specialty, knowledge in partial_knowledge:
            combined_content.append(f"From {specialty}: {knowledge.content}")
            combined_confidence += knowledge.confidence * knowledge.success_rate
            sources.append(f"{specialty}:{knowledge.source}")
        
        # Network emergent confidence (often higher than individual confidences)
        network_confidence = min(1.0, combined_confidence / len(partial_knowledge) * 1.2)
        
        if network_confidence > 0.4:  # Emergent solution threshold
            emergent_answer = " ".join(combined_content)
            
            return {
                "answer": f"Network emergent solution: {emergent_answer}",
                "confidence": network_confidence,
                "sources": sources,
                "reasoning": f"Emergent solution synthesized from {len(partial_knowledge)} partial knowledge items across {len(set(s for s, _ in partial_knowledge))} specialties",
                "solution_type": "emergent_network_intelligence",
                "contributors": list(set(s for s, _ in partial_knowledge))
            }
        
        return None
    
    def _integrate_network_solutions(self, primary: Optional[Dict], 
                                   others: List[Tuple[str, str, Dict]], 
                                   query: str, context: Context,
                                   experts: List[CASN]) -> Dict[str, Any]:
        """Advanced integration of multiple CASN solutions"""
        
        all_solutions = []
        if primary:
            all_solutions.append(("primary", primary))
        
        for specialty, casn_id, solution in others:
            all_solutions.append((f"{specialty}-{casn_id}", solution))
        
        if not all_solutions:
            return {"answer": None, "confidence": 0.0}
        
        # Advanced scoring considers multiple factors
        scored_solutions = []
        query_terms = query.lower().split()
        
        for source, solution in all_solutions:
            # Base confidence score
            base_score = solution['confidence']
            
            # Specialty alignment bonus
            specialty_bonus = 0.2 if any(term in source.lower() for term in query_terms) else 0.0
            
            # Knowledge breadth bonus
            breadth_bonus = 0.1 if solution.get('knowledge_used', 0) > 1 else 0.0
            
            # Historical performance bonus
            casn_id = source.split('-')[-1] if '-' in source else None
            if casn_id and casn_id in self.nodes:
                casn = self.nodes[casn_id]
                performance_bonus = casn.get_stats()['avg_confidence'] * 0.15
            else:
                performance_bonus = 0.0
            
            total_score = base_score + specialty_bonus + breadth_bonus + performance_bonus
            scored_solutions.append((total_score, source, solution))
        
        # Select best solution
        scored_solutions.sort(reverse=True)
        best_score, best_source, best_solution = scored_solutions[0]
        
        # Enhanced solution with network context
        network_solution = {
            "answer": best_solution['answer'],
            "confidence": best_score,
            "primary_source": best_source,
            "network_contributors": [source for _, source, _ in all_solutions],
            "reasoning": f"Network analysis: {len(all_solutions)} solutions evaluated, best from {best_source}. " + 
                        best_solution.get('reasoning', ''),
            "collaboration_benefit": len(all_solutions) > 1,
            "network_consensus": len(all_solutions) >= 3
        }
        
        return network_solution
    
    def _update_collaboration_metrics(self, query: str, context: Context, 
                                    experts: List[CASN], solution: Dict[str, Any],
                                    response_time: float):
        """Update comprehensive network metrics"""
        
        # Record collaboration event
        collaboration = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for storage
            "experts": [f"{e.specialty}-{e.id}" for e in experts],
            "solution_confidence": solution.get('confidence', 0.0),
            "response_time": response_time,
            "solution_type": solution.get('solution_type', 'standard')
        }
        
        self.collaboration_history.append(collaboration)
        
        # Update network metrics
        self.metrics.active_collaborations += 1
        
        # Update average response time (rolling average)
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (self.metrics.average_response_time * 0.9 + 
                                                response_time * 0.1)
        
        # Update success rate
        successful = solution.get('confidence', 0.0) > 0.5
        if successful:
            self.metrics.success_rate = (self.metrics.success_rate * 0.95 + 1.0 * 0.05)
        else:
            self.metrics.success_rate = (self.metrics.success_rate * 0.95 + 0.0 * 0.05)
        
        # Calculate network efficiency (successful collaborations / total collaborations)
        recent_collabs = self.collaboration_history[-50:]  # Last 50 collaborations
        recent_success = sum(1 for c in recent_collabs if c['solution_confidence'] > 0.5)
        self.metrics.network_efficiency = recent_success / max(len(recent_collabs), 1)
    
    def _propagate_successful_knowledge(self, solution: Dict[str, Any], 
                                      context: Context, experts: List[CASN]):
        """Propagate successful solutions as knowledge across the network"""
        
        if solution['confidence'] < 0.8:
            return
        
        # Create knowledge from successful solution
        successful_knowledge = solution['answer']
        source = f"network_collaboration_{datetime.now().strftime('%Y%m%d')}"
        
        # Propagate to relevant CASNs (not all, to avoid knowledge pollution)
        for expert in experts[:3]:  # Limit to top 3 contributors
            acquired = expert.acquire(successful_knowledge, context, source)
            if acquired:
                self.metrics.peer_learning_events += 1
        
        print(f"🌱 Knowledge propagated to {min(3, len(experts))} network CASNs")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Comprehensive network status and analytics"""
        
        # Calculate specialization distribution
        specialization_stats = {}
        for specialty, casn_ids in self.specialization_map.items():
            casn_count = len(casn_ids)
            total_knowledge = sum(len(self.nodes[cid].knowledge_store) for cid in casn_ids)
            avg_confidence = sum(self.nodes[cid].get_stats()['avg_confidence'] for cid in casn_ids) / casn_count
            
            specialization_stats[specialty] = {
                "casn_count": casn_count,
                "total_knowledge": total_knowledge,
                "avg_confidence": avg_confidence
            }
        
        # Recent performance analysis
        recent_collabs = self.collaboration_history[-20:] if self.collaboration_history else []
        avg_recent_confidence = sum(c['solution_confidence'] for c in recent_collabs) / max(len(recent_collabs), 1)
        
        return {
            "network_id": self.network_id,
            "metrics": {
                "total_nodes": self.metrics.total_nodes,
                "total_collaborations": len(self.collaboration_history),
                "network_efficiency": self.metrics.network_efficiency,
                "average_response_time": self.metrics.average_response_time,
                "success_rate": self.metrics.success_rate,
                "emergent_solutions": self.metrics.emergent_solutions,
                "peer_learning_events": self.metrics.peer_learning_events
            },
            "specializations": specialization_stats,
            "recent_performance": {
                "avg_confidence": avg_recent_confidence,
                "collaboration_count": len(recent_collabs)
            },
            "load_balance": dict(self.load_balancer)
        }
    
    def optimize_network(self) -> Dict[str, Any]:
        """Optimize network performance and balance"""
        optimizations = []
        
        # Reset load balancing
        self.load_balancer = defaultdict(int)
        optimizations.append("Load balancing reset")
        
        # Identify underperforming CASNs
        underperformers = []
        for casn_id, casn in self.nodes.items():
            stats = casn.get_stats()
            if stats['avg_confidence'] < 0.3 and stats['total_queries'] > 10:
                underperformers.append(casn_id)
        
        if underperformers:
            optimizations.append(f"Identified {len(underperformers)} underperforming CASNs")
        
        # Suggest knowledge redistribution if needed
        knowledge_imbalance = max(len(casn.knowledge_store) for casn in self.nodes.values()) - \
                            min(len(casn.knowledge_store) for casn in self.nodes.values())
        
        if knowledge_imbalance > 20:
            optimizations.append("Knowledge redistribution recommended")
        
        return {
            "optimizations_applied": optimizations,
            "network_health": "optimal" if self.metrics.network_efficiency > 0.8 else "needs_attention",
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for network improvement"""
        recommendations = []
        
        if self.metrics.network_efficiency < 0.7:
            recommendations.append("Consider adding more specialized CASNs")
        
        if self.metrics.average_response_time > 2.0:
            recommendations.append("Optimize knowledge search algorithms")
        
        if self.metrics.emergent_solutions == 0 and len(self.collaboration_history) > 10:
            recommendations.append("Network may benefit from more diverse specializations")
        
        if not recommendations:
            recommendations.append("Network operating at optimal performance")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # This would be imported and used with actual CASN instances
    print("🌐 CASN Network Management System loaded")
    print("Ready for large-scale cognitive architecture deployment!")