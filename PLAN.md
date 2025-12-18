# RHEA with Negamax Override - Implementation Plan

## Overview

Create a hybrid algorithm that uses RHEA for move selection but allows negamax to override the choice when negamax detects that RHEA's move leads to a losing position.

## Test Results (10 games each)

### RHEA+Override vs Plain RHEA
- **RHEA+Override wins: 3 (30%)**
- **Plain RHEA wins: 7 (70%)**
- Draws: 0

### RHEA+Override vs Negamax
- RHEA+Override wins: 0 (0%)
- **Negamax wins: 10 (100%)**
- Draws: 0

### Analysis
- Plain RHEA outperforms RHEA+Override, suggesting the override may be too aggressive
- Pure Negamax (depth 4) dominates both RHEA variants
- The simple win/lose heuristic may not be sophisticated enough
- Possible improvements: tune threshold, increase override depth, or use voronoi in simple eval

## Current State

- **RHEA** (`rhea.rs`): Main algorithm, 400 evolutions per move, uses complex fitness with health/length/voronoi/center scoring
- **Negamax** (`rhea.rs:437-709`): Exists but unused in production, uses voronoi-based evaluation
- **Main entry** (`main.rs`): Only uses RHEA via `ga.get_move()`

## Implementation Steps

### Step 1: Create Simple Win/Lose Heuristic for Negamax

**File**: `rhea.rs`

Create a new evaluation function for the override negamax that focuses purely on survival:

```rust
fn evaluate_simple(&self, game: &Game, player: &str, depth: usize) -> i32 {
    // Win: +10000 (prefer earlier wins)
    // Lose: -10000 (prefer later deaths)
    // Draw/ongoing: 0
}
```

This simple heuristic should:
- Return large positive score if player is the only survivor
- Return large negative score if player is eliminated
- Return 0 or very small positional score otherwise (maybe just health)

### Step 2: Create `NegamaxOverride` Struct

**File**: `rhea.rs`

New struct that wraps the simple negamax:

```rust
pub struct NegamaxOverride {
    pub max_depth: usize,       // Probably 3-4 for speed
    pub bad_threshold: i32,     // Score below which we consider a move "bad"
}
```

Methods:
- `is_move_bad(&self, game: &Game, player: &str, direction: Direction) -> bool`
  - Simulates the move
  - Runs negamax from resulting position
  - Returns true if score < bad_threshold

- `get_override_move(&self, game: &Game, player: &str, rhea_move: Direction) -> Option<Direction>`
  - If rhea_move is not bad, return None
  - Otherwise, find best move according to negamax and return Some(move)

### Step 3: Create Hybrid Algorithm Entry Point

**File**: `rhea.rs`

Create a new `RheaWithOverride` struct or function:

```rust
pub struct RheaWithOverride {
    pub rhea: RHEA,
    pub override_check: NegamaxOverride,
}

impl RheaWithOverride {
    pub fn get_move(&self) -> Direction {
        let rhea_move = self.rhea.get_move();

        // Check if negamax thinks this move is terrible
        if let Some(override_move) = self.override_check
            .get_override_move(&self.rhea.game, &self.rhea.player, rhea_move)
        {
            println!("Negamax override: {} -> {}", rhea_move, override_move);
            return override_move;
        }

        rhea_move
    }
}
```

### Step 4: Add Algorithm Configuration

**File**: `rhea.rs`

Extend the existing `Algorithm` enum:

```rust
pub enum Algorithm {
    Rhea(RheaConfig),
    Negamax,
    RheaWithNegamaxOverride(RheaConfig, NegamaxOverrideConfig),  // NEW
}
```

### Step 5: Create Test Infrastructure

**File**: `rhea.rs` (tests section)

Add comparison tests:

```rust
#[test]
#[ignore]  // Long running
fn compare_rhea_vs_rhea_with_override() {
    // Run tournament: RHEA vs RheaWithOverride
    // Track wins, losses, and override frequency
}
```

### Step 6: Optional - Wire Up Main Entry Point

**File**: `main.rs`

If tests show improvement, update main.rs to use the hybrid approach:

```rust
// Option A: Always use hybrid
let move_choice = rhea_with_override.get_move();

// Option B: Feature flag or config
```

## Testing Plan

1. **Unit tests**: Verify simple heuristic correctly identifies wins/losses
2. **Integration tests**: Run games between:
   - Pure RHEA vs Pure RHEA (baseline)
   - RheaWithOverride vs Pure RHEA (comparison)
   - RheaWithOverride vs Negamax (comparison)
3. **Metrics to track**:
   - Win rate
   - Override frequency (how often negamax overrides RHEA)
   - Game length (turns survived)

## Configuration Parameters

| Parameter | Suggested Value | Notes |
|-----------|----------------|-------|
| Override negamax depth | 3-4 | Balance speed vs accuracy |
| Bad threshold | -5000 | Below this, move is considered dangerous |
| RHEA evolutions | 400 | Keep same as current |
| RHEA population | 50 | Keep same as current |

## Files Modified

1. `rhea.rs` - Add NegamaxOverride, RheaWithOverride, simple evaluation
2. `main.rs` - (Optional) Update to use new hybrid algorithm
3. Tests in `rhea.rs` - Add comparison tests

## Risks and Considerations

- **Performance**: Running negamax on each move adds computation time
  - Mitigation: Use shallow depth (3-4) for override check
- **Over-caution**: Negamax might be too conservative and override good aggressive plays
  - Mitigation: Tune bad_threshold carefully
- **Conflicting evaluations**: RHEA and negamax may value positions differently
  - Mitigation: Simple win/lose heuristic reduces this gap
