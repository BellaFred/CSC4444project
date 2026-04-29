import torch
import sys
import os
import numpy as np
from collections import Counter
import copy
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)
sys.path.append(current_dir)

try:
    from tinyzero.models import LinearNetwork
    from tinyzero.agents import AlphaZeroAgent
    from tinyzero.mcts import search 
    from spades_env import SpadesEnv, action_to_card #
    from mcts import ISMCTSAgent 
    print("All modules imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def fixed_classic_value_fn(game):
    """Bypasses the boolean bug in ClassicMCTSAgent.value_fn."""
    game_copy = copy.deepcopy(game)
    result = game_copy.get_first_person_result()
    while result is None:
        actions = game_copy.get_legal_actions()
        if not actions or len(actions) == 0: 
            break
        game_copy.step(np.random.choice(actions))
        result = game_copy.get_first_person_result()
    return float(result) if result is not None else 0.0

def fixed_classic_policy_fn(game):
    """Returns a numpy array of size 66 (required by MCTS search)."""
  
    size = 66 
    return np.ones(size, dtype=np.float32) / size

class TinyZeroCompetitorAdapter:
    """Adapts different agents to the SpadesEnv interface."""
    def __init__(self, agent_type, agent_obj=None):
        self.agent_type = agent_type
        self.agent_obj = agent_obj

    def act(self, env):
        if self.agent_type == "ISMCTS":
         
            obs = env._state.observation(env.current_player)
            if env.phase == "bid":
                return int(self.agent_obj.bid(obs))
            else:
                legal_actions = env.get_legal_actions()
                legal_cards = [action_to_card(a) for a in legal_actions]
                card = self.agent_obj.play(obs, legal_cards)
          
                return int(card.index + 14)

        elif self.agent_type == "AlphaZero":
            root = search(
                env, 
                value_fn=self.agent_obj.value_fn, 
                policy_fn=self.agent_obj.policy_fn, 
                iterations=32
            )
            return int(root.children_actions[np.argmax(root.children_visits)])

        elif self.agent_type == "ClassicMCTS":
            root = search(
                env, 
                value_fn=fixed_classic_value_fn, 
                policy_fn=fixed_classic_policy_fn, 
                iterations=32
            )
            return int(root.children_actions[np.argmax(root.children_visits)])

def run_tournament():
    wandb.init(
        project="spades-alpha-tournament",
        name="ISMCTS_vs_AlphaZero_Comparison",
        config={
            "mcts_iterations": 32,
            "ismcts_n_sims": 30,
            "device": "gpu" if torch.cuda.is_available() else "cpu"
        }
    )
    print(f"Initializing Environment...")
    env = SpadesEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(root_dir, "spades/out/model.pth")
    model = LinearNetwork(env.observation_shape, env.action_space).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"AlphaZero Model loaded.")
    model.eval()

    az_agent = TinyZeroCompetitorAdapter("AlphaZero", AlphaZeroAgent(model))
    ismcts_agent = TinyZeroCompetitorAdapter("ISMCTS", ISMCTSAgent(n_det=10, n_sims=30))
    classic_agent = TinyZeroCompetitorAdapter("ClassicMCTS")

    scenario_1 = [ismcts_agent, az_agent, ismcts_agent, az_agent]
    
    scenario_2 = [ismcts_agent, classic_agent, ismcts_agent, classic_agent]

    scenarios = [
        ("ISMCTS vs AlphaZero", scenario_1),
        ("ISMCTS vs Classic MCTS", scenario_2)
    ]

    games_per_scenario = 10 

    for name, matchup in scenarios:
        print(f"\n" + "="*40)
        print(f"MATCHUP: {name}")
        print("="*40)
        
        for game_num in range(games_per_scenario):
            print(f"  Starting Game {game_num + 1}/{games_per_scenario}...")
            env.reset()
            done = False
            while not done:
                p_idx = env.current_player
                action = matchup[p_idx].act(env)
                env.step(action) 
                done = env.is_terminal
            
            
            tricks = env._state.tricks_won
            t0_tricks = tricks[0] + tricks[2]
            t1_tricks = tricks[1] + tricks[3]
            winner = 0 if t0_tricks > t1_tricks else 1

            
            wandb.log({
                f"{name}/Team0_Tricks": t0_tricks,
                f"{name}/Team1_Tricks": t1_tricks,
                f"{name}/Winner": winner,
                "overall_game_count": (scenarios.index((name, matchup)) * games_per_scenario) + game_num
            })

            
            if game_num == games_per_scenario - 1:
                columns = ["Trick", "P0_Card", "P1_Card", "P2_Card", "P3_Card", "Winner"]
                history_table = wandb.Table(columns=columns)
                for i, trick_cards in enumerate(env._state.tricks_played):
                    history_table.add_data(
                        i+1, str(trick_cards[0]), str(trick_cards[1]), 
                        str(trick_cards[2]), str(trick_cards[3]),
                        env._state.trick_winners[i]
                    )
                wandb.log({f"History/{name}_Final_Game": history_table})

            print(f"    Result: Team {winner} wins ({t0_tricks}-{t1_tricks})")
    
    wandb.finish()

if __name__ == "__main__":
    try:
        run_tournament()
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()