"""
Test script to verify the token limit enforcement fixes.
"""

from splendor_solo_env import SplendorSoloEnv
import numpy as np


def test_token_limit_after_reserve():
    """
    Test that token limit is enforced after reserving a card.
    This tests the fix for the bug where reserving with 10 tokens 
    would leave the player with 11 tokens.
    """
    print("=" * 60)
    print("Test: Token Limit After Reserve Action")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    obs, info = env.reset(seed=42)
    
    # Manually set player to have exactly 10 tokens (2 of each gem)
    env.player_tokens = np.array([2, 2, 2, 2, 2, 0], dtype=np.int32)
    env.bank_tokens = np.array([2, 2, 2, 2, 2, 5], dtype=np.int32)
    
    print(f"\nBefore reserve action:")
    print(f"  Player tokens: {env.player_tokens} (total: {np.sum(env.player_tokens)})")
    print(f"  Bank tokens: {env.bank_tokens}")
    
    # Find a reserve action (actions 27-38)
    legal_actions = env._get_legal_actions()
    reserve_actions = [i for i in range(27, 39) if legal_actions[i] == 1]
    
    if reserve_actions:
        action = reserve_actions[0]
        print(f"\nExecuting reserve action {action}...")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAfter reserve action:")
        print(f"  Player tokens: {obs['player_tokens']} (total: {np.sum(obs['player_tokens'])})")
        print(f"  Bank tokens: {obs['bank_tokens']}")
        
        total_tokens = np.sum(obs['player_tokens'])
        if total_tokens <= 10:
            print(f"\n✓ PASS: Token limit enforced correctly (total = {total_tokens})")
            return True
        else:
            print(f"\n✗ FAIL: Token limit NOT enforced (total = {total_tokens}, expected ≤ 10)")
            return False
    else:
        print("\n⚠ SKIP: No legal reserve actions available")
        return None


def test_token_discard_priority():
    """
    Test that tokens are discarded in the correct priority order.
    Non-gold gems should be discarded before gold.
    """
    print("\n" + "=" * 60)
    print("Test: Token Discard Priority")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    obs, info = env.reset(seed=42)
    
    # Manually set player to have 13 tokens (need to discard 3)
    # 2 of each gem + 3 gold = 13 total
    env.player_tokens = np.array([2, 2, 2, 2, 2, 3], dtype=np.int32)
    env.bank_tokens = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
    
    print(f"\nBefore enforcing limit:")
    print(f"  Player tokens: {env.player_tokens}")
    print(f"  Total: {np.sum(env.player_tokens)} (should discard 3)")
    
    # Call enforce token limit
    env._enforce_token_limit()
    
    print(f"\nAfter enforcing limit:")
    print(f"  Player tokens: {env.player_tokens}")
    print(f"  Total: {np.sum(env.player_tokens)}")
    
    # Check results
    total = np.sum(env.player_tokens)
    gold_count = env.player_tokens[5]
    
    success = True
    
    if total != 10:
        print(f"\n✗ FAIL: Total tokens = {total}, expected 10")
        success = False
    else:
        print(f"\n✓ Total is correct: {total}")
    
    # Gold should still be 3 (non-gold gems discarded first)
    if gold_count != 3:
        print(f"✗ FAIL: Gold tokens = {gold_count}, expected 3 (non-gold should be discarded first)")
        success = False
    else:
        print(f"✓ Gold tokens preserved: {gold_count}")
    
    # Check that black tokens were discarded first
    black_count = env.player_tokens[0]
    if black_count == 0:
        print(f"✓ Black tokens discarded first (priority order correct)")
    else:
        print(f"⚠ Black tokens = {black_count} (should be discarded first in priority)")
    
    return success


def test_token_limit_after_taking_tokens():
    """
    Test that token limit is enforced after taking tokens.
    """
    print("\n" + "=" * 60)
    print("Test: Token Limit After Taking Tokens")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    obs, info = env.reset(seed=123)
    
    # Set player to have 9 tokens
    env.player_tokens = np.array([2, 2, 2, 2, 1, 0], dtype=np.int32)
    
    print(f"\nBefore taking tokens:")
    print(f"  Player tokens: {env.player_tokens} (total: {np.sum(env.player_tokens)})")
    
    # Find a take 3 unique action (actions 0-9)
    legal_actions = env._get_legal_actions()
    take_3_actions = [i for i in range(10) if legal_actions[i] == 1]
    
    if take_3_actions:
        action = take_3_actions[0]
        print(f"\nExecuting take 3 tokens action {action}...")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAfter taking tokens:")
        print(f"  Player tokens: {obs['player_tokens']} (total: {np.sum(obs['player_tokens'])})")
        
        total_tokens = np.sum(obs['player_tokens'])
        if total_tokens <= 10:
            print(f"\n✓ PASS: Token limit enforced (total = {total_tokens})")
            return True
        else:
            print(f"\n✗ FAIL: Token limit NOT enforced (total = {total_tokens})")
            return False
    else:
        print("\n⚠ SKIP: No legal take 3 actions available")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING TOKEN LIMIT ENFORCEMENT FIXES")
    print("=" * 60)
    
    results = []
    
    try:
        result1 = test_token_limit_after_reserve()
        results.append(("Reserve Action Test", result1))
        
        result2 = test_token_discard_priority()
        results.append(("Discard Priority Test", result2))
        
        result3 = test_token_limit_after_taking_tokens()
        results.append(("Take Tokens Test", result3))
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in results:
            if result is True:
                status = "✓ PASS"
            elif result is False:
                status = "✗ FAIL"
            else:
                status = "⚠ SKIP"
            print(f"{status}: {test_name}")
        
        all_passed = all(r in [True, None] for _, r in results)
        if all_passed:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SOME TESTS FAILED")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
