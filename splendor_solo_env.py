"""
Splendor Solo Play Gymnasium Environment
Based strictly on the provided solo play rulebook.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from itertools import combinations


class SplendorSoloEnv(gym.Env):
    """
    A Gymnasium environment for solo Splendor gameplay.
    
    Action Space: Discrete(39)
        - Actions 0-9: Take 3 unique gems (10 combinations from 5 colors)
        - Actions 10-14: Take 2 identical gems (5 actions, one per color)
        - Actions 15-23: Purchase face-up cards (9 cards)
        - Actions 24-26: Purchase reserved cards (3 reserved slots)
        - Actions 27-35: Reserve face-up cards (9 cards)
        - Actions 36-38: Reserve from deck top (3 decks: L1, L2, L3)
    
    Observation Space: Dict with:
        - bank_tokens: (6,) - [black, white, red, blue, green, gold]
        - player_tokens: (6,) - [black, white, red, blue, green, gold]
        - player_bonuses: (5,) - [black, white, red, blue, green]
        - player_points: scalar
        - player_reserved: (21,) - 3 cards × (color_id + points + 5 costs)
        - face_up_cards: (63,) - 9 cards × (color_id + points + 5 costs)
        - noble_tiles: (12,) - 2 nobles × (points + 5 costs)
        - game_clock: scalar - remaining L3 cards
    """
    
    metadata = {'render_modes': []}
    
    # Color mapping: black=0, white=1, red=2, blue=3, green=4, gold=5
    COLOR_MAP = {'black': 0, 'white': 1, 'red': 2, 'blue': 3, 'green': 4}
    COLOR_NAMES = ['black', 'white', 'red', 'blue', 'green']
    
    def __init__(self, cards_path: str = "cards.csv", nobles_path: str = "noble_tiles.csv"):
        super().__init__()
        
        # Load data
        self.all_cards = pd.read_csv(cards_path)
        self.all_nobles = pd.read_csv(nobles_path)
        
        # Convert NaN to 0 in cost columns
        cost_cols = ['cost_black', 'cost_white', 'cost_red', 'cost_blue', 'cost_green']
        self.all_cards[cost_cols] = self.all_cards[cost_cols].fillna(0).astype(int)
        self.all_nobles[cost_cols] = self.all_nobles[cost_cols].fillna(0).astype(int)
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'bank_tokens': gym.spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32),
            'player_tokens': gym.spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32),
            'player_bonuses': gym.spaces.Box(low=0, high=100, shape=(5,), dtype=np.int32),
            'player_points': gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'player_reserved': gym.spaces.Box(low=-1, high=100, shape=(21,), dtype=np.int32),
            'face_up_cards': gym.spaces.Box(low=-1, high=100, shape=(63,), dtype=np.int32),
            'noble_tiles': gym.spaces.Box(low=0, high=100, shape=(12,), dtype=np.int32),
            'game_clock': gym.spaces.Box(low=0, high=17, shape=(1,), dtype=np.int32),
        })
        
        # Define action space (39 discrete actions)
        self.action_space = gym.spaces.Discrete(39)
        
        # Generate action mappings
        self._generate_action_mappings()
        
        # Initialize state variables
        self._initialize_state()
    
    def _generate_action_mappings(self):
        """Generate mappings between action integers and game actions."""
        self.action_map = {}
        action_id = 0
        
        # Actions 0-9: Take 3 unique gems
        for combo in combinations(range(5), 3):
            self.action_map[action_id] = ('take_3_unique', combo)
            action_id += 1
        
        # Actions 10-14: Take 2 identical gems
        for color in range(5):
            self.action_map[action_id] = ('take_2_identical', color)
            action_id += 1
        
        # Actions 15-23: Purchase face-up cards (positions 0-8)
        for pos in range(9):
            self.action_map[action_id] = ('purchase_face_up', pos)
            action_id += 1
        
        # Actions 24-26: Purchase reserved cards (slots 0-2)
        for slot in range(3):
            self.action_map[action_id] = ('purchase_reserved', slot)
            action_id += 1
        
        # Actions 27-35: Reserve face-up cards (positions 0-8)
        for pos in range(9):
            self.action_map[action_id] = ('reserve_face_up', pos)
            action_id += 1
        
        # Actions 36-38: Reserve from deck (tiers 0=L1, 1=L2, 2=L3)
        for tier in range(3):
            self.action_map[action_id] = ('reserve_deck', tier)
            action_id += 1
    
    def _initialize_state(self):
        """Initialize all state variables."""
        self.bank_tokens = np.zeros(6, dtype=np.int32)
        self.player_tokens = np.zeros(6, dtype=np.int32)
        self.player_bonuses = np.zeros(5, dtype=np.int32)
        self.player_points = 0
        self.player_reserved = []  # List of card dicts
        self.purchased_cards = []  # List of card dicts
        self.face_up_cards = [None] * 9  # 9 face-up card slots
        self.noble_tiles = []  # List of 2 noble dicts
        self.acquired_nobles = []  # Nobles already acquired
        self.decks = {1: [], 2: [], 3: []}  # Decks by tier
        self.game_clock = 17
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation dict
            info: Info dict containing 'legal_actions' mask
        """
        super().reset(seed=seed)
        
        # Initialize state
        self._initialize_state()
        
        # Rule 2.1: Set bank tokens (4 of each gem, 5 gold)
        self.bank_tokens[:5] = 4
        self.bank_tokens[5] = 5
        
        # Rule 2.2: Shuffle and reveal 2 nobles
        nobles_list = self.all_nobles.to_dict('records')
        self.np_random.shuffle(nobles_list)
        self.noble_tiles = nobles_list[:2]
        
        # Rule 2.3: Shuffle decks and reveal 9 cards
        for tier in [1, 2, 3]:
            tier_cards = self.all_cards[self.all_cards['tier'] == tier].to_dict('records')
            self.np_random.shuffle(tier_cards)
            self.decks[tier] = tier_cards
        
        # Reveal 3 cards from each tier (9 total)
        for i in range(3):  # 3 tiers
            for j in range(3):  # 3 cards per tier
                slot_idx = i * 3 + j
                if self.decks[i + 1]:
                    self.face_up_cards[slot_idx] = self.decks[i + 1].pop(0)
        
        # Rule 2.4: Set game clock to 17
        self.game_clock = 17
        
        # Get observation and legal actions
        observation = self._get_observation()
        legal_actions = self._get_legal_actions()
        
        info = {'legal_actions': legal_actions}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action from 0 to 38
            
        Returns:
            observation: New observation dict
            reward: Prestige points earned this step
            terminated: Whether game has ended
            truncated: Always False (terminated handles game clock)
            info: Info dict with 'legal_actions' mask
        """
        # Check if action is legal
        legal_actions = self._get_legal_actions()
        if not legal_actions[action]:
            # Illegal action: large negative reward and terminate
            observation = self._get_observation()
            return observation, -100.0, True, False, {'legal_actions': legal_actions}
        
        reward = 0.0
        terminated = False
        
        # Decode action
        action_type, action_params = self.action_map[action]
        
        # Phase 1: Execute action
        if action_type == 'take_3_unique':
            self._execute_take_3_unique(action_params)
        elif action_type == 'take_2_identical':
            self._execute_take_2_identical(action_params)
        elif action_type == 'purchase_face_up':
            reward = self._execute_purchase_face_up(action_params)
        elif action_type == 'purchase_reserved':
            reward = self._execute_purchase_reserved(action_params)
        elif action_type == 'reserve_face_up':
            self._execute_reserve_face_up(action_params)
        elif action_type == 'reserve_deck':
            self._execute_reserve_deck(action_params)
        
        # Rule 5.2: Enforce token limit (max 10 tokens)
        self._enforce_token_limit()
        
        # Check for deck depletion end (Rule 6.2)
        # This happens if a L3 face-up slot is empty and L3 deck is empty
        for i in range(6, 9):  # L3 slots are indices 6, 7, 8
            if self.face_up_cards[i] is None and not self.decks[3]:
                terminated = True
        
        # Phase 2: Game clock tick (Rule 3.2)
        if not terminated and self.game_clock > 0:
            self.game_clock -= 1
            # If no Level 3 deck, decrement happens but we check if it's now 0
            if self.game_clock == 0:
                terminated = True  # Rule 6.1: Timer end
        
        # Get new observation and legal actions
        observation = self._get_observation()
        legal_actions = self._get_legal_actions()
        info = {'legal_actions': legal_actions}
        
        return observation, reward, terminated, False, info
    
    def _execute_take_3_unique(self, colors: Tuple[int, int, int]):
        """Execute action: Take 3 unique gem tokens."""
        for color in colors:
            self.bank_tokens[color] -= 1
            self.player_tokens[color] += 1
    
    def _execute_take_2_identical(self, color: int):
        """Execute action: Take 2 identical gem tokens."""
        self.bank_tokens[color] -= 2
        self.player_tokens[color] += 2
    
    def _execute_purchase_face_up(self, position: int) -> float:
        """Execute action: Purchase a face-up card."""
        card = self.face_up_cards[position]
        reward = self._purchase_card(card)
        
        # Remove card from face-up and refill (Rule 5.1)
        self.face_up_cards[position] = None
        self._refill_face_up_card(position)
        
        return reward
    
    def _execute_purchase_reserved(self, slot: int) -> float:
        """Execute action: Purchase a reserved card."""
        card = self.player_reserved[slot]
        reward = self._purchase_card(card)
        
        # Remove card from reserved hand
        self.player_reserved.pop(slot)
        
        return reward
    
    def _purchase_card(self, card: dict) -> float:
        """
        Common logic for purchasing a card.
        
        Returns:
            reward: Total prestige points earned (card + nobles)
        """
        reward = 0.0
        
        # Calculate cost and pay
        costs = [
            card['cost_black'],
            card['cost_white'],
            card['cost_red'],
            card['cost_blue'],
            card['cost_green']
        ]
        
        for color_idx, cost in enumerate(costs):
            # Use bonuses first, then tokens, then gold
            remaining_cost = cost - self.player_bonuses[color_idx]
            if remaining_cost > 0:
                tokens_to_spend = min(remaining_cost, self.player_tokens[color_idx])
                self.player_tokens[color_idx] -= tokens_to_spend
                self.bank_tokens[color_idx] += tokens_to_spend
                remaining_cost -= tokens_to_spend
                
                # Use gold for remaining
                if remaining_cost > 0:
                    self.player_tokens[5] -= remaining_cost
                    self.bank_tokens[5] += remaining_cost
        
        # Add card to purchased cards
        self.purchased_cards.append(card)
        
        # Add bonus
        color_idx = self.COLOR_MAP[card['color']]
        self.player_bonuses[color_idx] += 1
        
        # Add points
        points = card['points']
        self.player_points += points
        reward += points
        
        # Rule 5.3: Check for noble visits
        noble_reward = self._check_noble_visits()
        reward += noble_reward
        
        return reward
    
    def _execute_reserve_face_up(self, position: int):
        """Execute action: Reserve a face-up card."""
        card = self.face_up_cards[position]
        self.player_reserved.append(card)
        
        # Take gold if available
        if self.bank_tokens[5] > 0:
            self.bank_tokens[5] -= 1
            self.player_tokens[5] += 1
        
        # Remove card from face-up and refill (Rule 5.1)
        self.face_up_cards[position] = None
        self._refill_face_up_card(position)
    
    def _execute_reserve_deck(self, tier: int):
        """Execute action: Reserve from deck top."""
        deck_tier = tier + 1  # tier 0=L1, 1=L2, 2=L3
        card = self.decks[deck_tier].pop(0)
        self.player_reserved.append(card)
        
        # Take gold if available
        if self.bank_tokens[5] > 0:
            self.bank_tokens[5] -= 1
            self.player_tokens[5] += 1
    
    def _refill_face_up_card(self, position: int):
        """Refill a face-up card slot from its corresponding deck."""
        # Determine which deck this position belongs to
        tier = (position // 3) + 1  # positions 0-2: L1, 3-5: L2, 6-8: L3
        
        if self.decks[tier]:
            self.face_up_cards[position] = self.decks[tier].pop(0)
    
    def _check_noble_visits(self) -> float:
        """
        Check if player qualifies for any noble tiles.
        
        Returns:
            reward: Points from acquired nobles
        """
        reward = 0.0
        nobles_to_acquire = []
        
        for noble in self.noble_tiles:
            if noble in self.acquired_nobles:
                continue
            
            # Check if bonuses meet requirements
            costs = [
                noble['cost_black'],
                noble['cost_white'],
                noble['cost_red'],
                noble['cost_blue'],
                noble['cost_green']
            ]
            
            meets_requirements = True
            for color_idx, cost in enumerate(costs):
                if self.player_bonuses[color_idx] < cost:
                    meets_requirements = False
                    break
            
            if meets_requirements:
                nobles_to_acquire.append(noble)
                self.acquired_nobles.append(noble)
                points = noble['points']
                self.player_points += points
                reward += points
        
        return reward
    
    def _enforce_token_limit(self):
        """
        Enforce the 10-token limit by discarding excess tokens.
        Rule 5.2: At the end of an action, if the agent has more than 10 tokens,
        it must return tokens to the bank until it has exactly 10.
        
        Strategy: Discard non-gold gems first (in order: black, white, red, blue, green),
        then discard gold tokens if necessary.
        """
        total_tokens = np.sum(self.player_tokens)
        
        if total_tokens <= 10:
            return
        
        tokens_to_discard = total_tokens - 10
        
        # Discard priority: non-gold gems first (black, white, red, blue, green)
        for color_idx in range(5):  # 0=black, 1=white, 2=red, 3=blue, 4=green
            while tokens_to_discard > 0 and self.player_tokens[color_idx] > 0:
                self.player_tokens[color_idx] -= 1
                self.bank_tokens[color_idx] += 1
                tokens_to_discard -= 1
        
        # If still need to discard, discard gold tokens
        while tokens_to_discard > 0 and self.player_tokens[5] > 0:
            self.player_tokens[5] -= 1
            self.bank_tokens[5] += 1
            tokens_to_discard -= 1
    
    def _get_observation(self) -> Dict:
        """
        Get current observation.
        
        Returns:
            observation: Dict matching observation_space
        """
        obs = {
            'bank_tokens': self.bank_tokens.copy(),
            'player_tokens': self.player_tokens.copy(),
            'player_bonuses': self.player_bonuses.copy(),
            'player_points': np.array([self.player_points], dtype=np.int32),
            'player_reserved': self._encode_cards(self.player_reserved, max_cards=3),
            'face_up_cards': self._encode_cards(self.face_up_cards, max_cards=9),
            'noble_tiles': self._encode_nobles(self.noble_tiles),
            'game_clock': np.array([self.game_clock], dtype=np.int32),
        }
        return obs
    
    def _encode_cards(self, cards: List, max_cards: int) -> np.ndarray:
        """
        Encode a list of cards into a flat array.
        
        Each card: [color_id, points, cost_black, cost_white, cost_red, cost_blue, cost_green]
        Total: 7 features per card
        
        Args:
            cards: List of card dicts (or None for empty slots)
            max_cards: Maximum number of cards
            
        Returns:
            Flat array of size max_cards * 7
        """
        encoded = np.full(max_cards * 7, -1, dtype=np.int32)
        
        for i, card in enumerate(cards):
            if i >= max_cards:
                break
            if card is None:
                continue
            
            offset = i * 7
            encoded[offset] = self.COLOR_MAP[card['color']]
            encoded[offset + 1] = card['points']
            encoded[offset + 2] = card['cost_black']
            encoded[offset + 3] = card['cost_white']
            encoded[offset + 4] = card['cost_red']
            encoded[offset + 5] = card['cost_blue']
            encoded[offset + 6] = card['cost_green']
        
        return encoded
    
    def _encode_nobles(self, nobles: List[dict]) -> np.ndarray:
        """
        Encode noble tiles into a flat array.
        
        Each noble: [points, cost_black, cost_white, cost_red, cost_blue, cost_green]
        Total: 6 features per noble
        
        Returns:
            Flat array of size 2 * 6 = 12
        """
        encoded = np.zeros(12, dtype=np.int32)
        
        for i, noble in enumerate(nobles):
            if i >= 2:
                break
            
            offset = i * 6
            encoded[offset] = noble['points']
            encoded[offset + 1] = noble['cost_black']
            encoded[offset + 2] = noble['cost_white']
            encoded[offset + 3] = noble['cost_red']
            encoded[offset + 4] = noble['cost_blue']
            encoded[offset + 5] = noble['cost_green']
        
        return encoded
    
    def _get_legal_actions(self) -> np.ndarray:
        """
        Get a binary mask of legal actions.
        
        Returns:
            Array of shape (39,) with 1 for legal actions, 0 for illegal
        """
        legal = np.zeros(39, dtype=np.int8)
        
        for action_id in range(39):
            action_type, action_params = self.action_map[action_id]
            
            if action_type == 'take_3_unique':
                # Check if all 3 colors are available in bank
                if all(self.bank_tokens[c] >= 1 for c in action_params):
                    legal[action_id] = 1
            
            elif action_type == 'take_2_identical':
                # Rule 4.2: Need at least 4 tokens of that color in bank
                if self.bank_tokens[action_params] >= 4:
                    legal[action_id] = 1
            
            elif action_type == 'purchase_face_up':
                # Check if card exists and is affordable
                card = self.face_up_cards[action_params]
                if card is not None and self._can_afford(card):
                    legal[action_id] = 1
            
            elif action_type == 'purchase_reserved':
                # Check if reserved slot has a card and is affordable
                if action_params < len(self.player_reserved):
                    card = self.player_reserved[action_params]
                    if self._can_afford(card):
                        legal[action_id] = 1
            
            elif action_type == 'reserve_face_up':
                # Rule 4.3: Check if card exists and player has space
                card = self.face_up_cards[action_params]
                if card is not None and len(self.player_reserved) < 3:
                    legal[action_id] = 1
            
            elif action_type == 'reserve_deck':
                # Check if deck has cards and player has space
                deck_tier = action_params + 1
                if self.decks[deck_tier] and len(self.player_reserved) < 3:
                    legal[action_id] = 1
        
        return legal
    
    def _can_afford(self, card: dict) -> bool:
        """
        Check if player can afford a card.
        
        Args:
            card: Card dict with costs
            
        Returns:
            True if affordable, False otherwise
        """
        costs = [
            card['cost_black'],
            card['cost_white'],
            card['cost_red'],
            card['cost_blue'],
            card['cost_green']
        ]
        
        gold_needed = 0
        
        for color_idx, cost in enumerate(costs):
            # Calculate how much we need after using bonuses
            remaining_cost = cost - self.player_bonuses[color_idx]
            if remaining_cost > 0:
                # Check if we have enough tokens of this color
                tokens_available = self.player_tokens[color_idx]
                if tokens_available < remaining_cost:
                    # Need gold to cover the difference
                    gold_needed += remaining_cost - tokens_available
        
        # Check if we have enough gold
        return gold_needed <= self.player_tokens[5]
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
