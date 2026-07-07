import mesa
import numpy as np
from scipy.spatial import cKDTree
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from slp import SingleLayerPerceptron
from data_loader import EbolaDataLoader


class EbolaAgent(mesa.Agent):
    """
    An agent in the Ebola SEIRD model.

    States:
      S = Susceptible
      E = Exposed (incubating, not yet infectious)
      I = Infectious
      R = Recovered (immune)
      D = Dead
    """

    def __init__(self, model, state='S'):
        super().__init__(model)
        self.state = state

        # Agent attributes (randomised, calibrated from Ebola data)
        self.age = self.random.randint(1, 80)
        self.age_norm = self.age / 80.0
        self.contacts = self.random.randint(1, 15)
        self.contacts_norm = self.contacts / 15.0
        self.healthcare_access = model.country_params.get('healthcare_access', 0.3)
        self.crowding = model.country_params.get('population_density', 0.5)

        # Disease progression counters
        self.days_exposed = 0
        self.days_infectious = 0
        self.incubation_period = max(2, int(self.random.gauss(
            model.country_params.get('incubation_mean', 9), 3
        )))  # Ebola: 2-21 days, mean ~9
        self.infectious_period = max(4, int(self.random.gauss(10, 3)))  # 6-16 days

        # SLP risk score (updated each step)
        self.risk_score = 0.5

    def step(self):
        """Agent step: move, check infection, progress disease."""
        if self.state == 'D':
            return  # Dead agents don't act

        self.move()

        if self.state == 'S':
            self.check_infection()
        elif self.state == 'E':
            self.progress_exposed()
        elif self.state == 'I':
            self.progress_infectious()

    def move(self):
        """Random movement on continuous space."""
        if self.state == 'D':
            return

        x, y = self.pos
        # Reduced movement for infectious agents (Ebola patients are often bedridden)
        speed = 0.5 if self.state == 'I' else 1.5
        dx = self.random.uniform(-speed, speed)
        dy = self.random.uniform(-speed, speed)

        new_x = max(0, min(self.model.space.x_max, x + dx))
        new_y = max(0, min(self.model.space.y_max, y + dy))

        self.model.space.move_agent(self, (new_x, new_y))

    def check_infection(self):
        """Check if susceptible agent gets infected by nearby infectious agents using KDTree."""
        if not hasattr(self.model, 'kdtree') or self.model.kdtree is None:
            return

        # Query KDTree for neighbors within radius 3.0
        indices = self.model.kdtree.query_ball_point(self.pos, r=3.0)
        
        # Get infectious neighbors (excluding self)
        infectious_neighbors = []
        agent_list = list(self.model.agents)
        for idx in indices:
            neighbor = agent_list[idx]
            if neighbor != self and neighbor.state == 'I':
                infectious_neighbors.append(neighbor)

        if not infectious_neighbors:
            return

        # SLP-gated infection: risk_score * infection_rate * contact_factor
        for neighbor in infectious_neighbors:
            contact_factor = self.contacts_norm
            base_rate = self.model.infection_rate
            slp_risk = self.risk_score

            infection_prob = slp_risk * base_rate * contact_factor

            # Healthcare access reduces infection chance
            infection_prob *= (1.0 - self.healthcare_access * 0.5)

            if self.random.random() < infection_prob:
                self.state = 'E'
                self.days_exposed = 0
                break

    def progress_exposed(self):
        """Progress through Ebola incubation period (E → I)."""
        self.days_exposed += 1
        if self.days_exposed >= self.incubation_period:
            self.state = 'I'
            self.days_infectious = 0

    def progress_infectious(self):
        """Progress through infectious period (I → R or D)."""
        self.days_infectious += 1
        if self.days_infectious >= self.infectious_period:
            # Ebola CFR: country-specific (39-74%)
            cfr = self.model.country_params.get('cfr', 0.5)

            # Age factor: older patients have higher mortality
            age_factor = 1.0 + (self.age_norm - 0.5) * 0.4

            # Healthcare reduces mortality
            healthcare_factor = 1.0 - self.healthcare_access * 0.3

            death_prob = cfr * age_factor * healthcare_factor
            death_prob = min(max(death_prob, 0.1), 0.95)  # Clamp

            if self.random.random() < death_prob:
                self.state = 'D'
            else:
                self.state = 'R'


class EbolaDiseaseModel(mesa.Model):
    """
    Ebola Disease Spread Model using Mesa 

    Features:
    - SEIRD compartmental model on ContinuousSpace
    - SLP-gated infection decisions
    - Real Ebola data calibration per country
    - DataCollector for per-step statistics
    """

    def __init__(self, n_agents=5000, infection_rate=0.15,
                 country='Guinea', initial_infected=3,
                 slp=None, data_loader=None):
        super().__init__()

        self.n_agents = n_agents
        self.infection_rate = infection_rate
        self.country = country
        self.initial_infected = initial_infected
        self.current_step = 0

        # Load country-specific parameters
        if data_loader is None:
            try:
                data_loader = EbolaDataLoader()
            except Exception:
                data_loader = None

        self.data_loader = data_loader

        if data_loader:
            self.country_params = data_loader.get_abm_params(country)
            self.infection_rate = self.country_params.get('infection_rate', infection_rate)
        else:
            self.country_params = {
                'infection_rate': infection_rate,
                'cfr': 0.5,
                'healthcare_access': 0.3,
                'population_density': 0.5,
                'incubation_mean': 9,
            }

        # SLP for risk prediction
        self.slp = slp
        if self.slp is None:
            self.slp = SingleLayerPerceptron(n_inputs=4, learning_rate=0.1)
            if data_loader:
                X, y = data_loader.get_slp_training_data()
                if len(X) > 0:
                    self.slp.train(X, y, epochs=100)

        # Continuous space (100x100)
        self.space = ContinuousSpace(100, 100, torus=True)

        # Create agents
        for i in range(n_agents):
            state = 'I' if i < initial_infected else 'S'
            agent = EbolaAgent(self, state=state)

            # Place randomly on space
            x = self.random.uniform(0, 100)
            y = self.random.uniform(0, 100)
            self.space.place_agent(agent, (x, y))

        # Build initial KDTree and update risk scores
        self._build_kdtree()
        self._update_risk_scores()

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                'Susceptible': lambda m: self._count_state(m, 'S'),
                'Exposed': lambda m: self._count_state(m, 'E'),
                'Infectious': lambda m: self._count_state(m, 'I'),
                'Recovered': lambda m: self._count_state(m, 'R'),
                'Dead': lambda m: self._count_state(m, 'D'),
                'Step': lambda m: m.current_step,
            }
        )

        # Collect initial state
        self.datacollector.collect(self)

        # History for charts
        self._history = []
        self._record_state()

    @staticmethod
    def _count_state(model, state):
        """Count agents in a given state."""
        return sum(1 for a in model.agents if a.state == state)

    def _build_kdtree(self):
        """Build a KDTree for fast spatial queries."""
        agent_list = list(self.agents)
        positions = [a.pos for a in agent_list]
        self.kdtree = cKDTree(positions)

    def _update_risk_scores(self):
        """Update all agents' SLP risk scores in batch."""
        if self.slp is None or not self.slp.is_trained:
            return

        agent_list = list(self.agents)
        if not agent_list:
            return

        # Are there any infected agents?
        total_infected = sum(1 for a in agent_list if a.state == 'I')

        # Build feature matrix for all agents dynamically
        features = []
        for a in agent_list:
            if a.state != 'S' or total_infected == 0:
                features.append([a.age_norm, 0.0, a.healthcare_access, a.crowding])
            else:
                # Calculate dynamic spatial features using KDTree
                # 1. Local Crowding (number of agents in radius)
                indices = self.kdtree.query_ball_point(a.pos, r=4.0)
                neighbors = [agent_list[idx] for idx in indices if agent_list[idx] != a]
                
                local_crowding = min(len(neighbors) / 20.0, 1.0)
                
                # 2. Infected Contacts (ratio of infected neighbors)
                infected_neighbors = sum(1 for n in neighbors if n.state == 'I')
                contacts_norm = min(infected_neighbors / 5.0, 1.0)
                
                # Update agent properties so UI sees them
                a.crowding = local_crowding
                a.contacts_norm = contacts_norm
                
                features.append([a.age_norm, a.contacts_norm, a.healthcare_access, a.crowding])

        features = np.array(features)

        # Batch predict base risk (intrinsic vulnerability)
        base_risks = self.slp.predict_batch(features)

        # Assign risk scores back to agents
        for agent, base_risk in zip(agent_list, base_risks):
            # If the agent is not susceptible or there are no infected agents in the model, risk is 0
            if agent.state != 'S' or total_infected == 0:
                agent.risk_score = 0.0
            else:
                agent.risk_score = float(base_risk)

    def _record_state(self):
        """Record current state counts for history."""
        counts = self.get_state_counts()
        counts['step'] = self.current_step
        self._history.append(counts)

    def step(self):
        """Run one simulation step."""
        self.current_step += 1

        # Shuffle and step all agents
        self.agents.shuffle_do("step")

        # Build KDTree for the new positions
        self._build_kdtree()

        # Update SLP risk scores every 5 steps (performance optimization)
        if self.current_step % 5 == 0:
            self._update_risk_scores()

        # Collect data
        self.datacollector.collect(self)
        self._record_state()

    def get_state_counts(self):
        """Return current state counts as dict."""
        counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0, 'D': 0}
        for agent in self.agents:
            counts[agent.state] = counts.get(agent.state, 0) + 1
        return counts

    def get_agent_positions(self):
        """Return all agent positions and states for canvas rendering."""
        agents_data = []
        for agent in self.agents:
            agents_data.append({
                'id': agent.unique_id,
                'x': round(agent.pos[0], 2),
                'y': round(agent.pos[1], 2),
                'state': agent.state,
                'risk': round(agent.risk_score, 4),
            })
        return agents_data

    def get_top_agents(self, n=10):
        """Return top N agents by risk score for risk cards."""
        living_agents = [a for a in self.agents if a.state != 'D']
        sorted_agents = sorted(living_agents, key=lambda a: a.risk_score, reverse=True)

        top = []
        for agent in sorted_agents[:n]:
            top.append({
                'id': agent.unique_id,
                'risk': round(agent.risk_score * 100, 1),
                'state': agent.state,
                'age': agent.age,
                'contacts': agent.contacts,
            })
        return top

    def get_history(self):
        """Return step-by-step state counts for charts."""
        return self._history

    def get_full_state(self):
        """Return complete model state for frontend update."""
        counts = self.get_state_counts()
        return {
            'step': self.current_step,
            'counts': counts,
            'total_alive': counts['S'] + counts['E'] + counts['I'] + counts['R'],
            'total_dead': counts['D'],
            'agents': self.get_agent_positions(),
            'top_agents': self.get_top_agents(10),
            'history': self._history,
        }


# Quick test
if __name__ == '__main__':
    print("Testing EbolaDiseaseModel...")
    print("Loading Ebola data and training SLP...")

    model = EbolaDiseaseModel(n_agents=100, country='Guinea', initial_infected=3)

    print(f"\nInitial state: {model.get_state_counts()}")
    print(f"Top 5 agents by risk:")
    for a in model.get_top_agents(5):
        print(f"  Agent {a['id']}: risk={a['risk']}%, state={a['state']}, age={a['age']}")

    # Run 20 steps
    for i in range(20):
        model.step()

    print(f"\nAfter 20 steps: {model.get_state_counts()}")
    print(f"History length: {len(model.get_history())}")

    # Show SLP info
    slp_info = model.slp.get_info()
    print(f"\nSLP trained: {slp_info['is_trained']}, Accuracy: {slp_info['accuracy']:.4f}")
    print(f"SLP weights: {model.slp.get_weights_str()}")
