# Plan: Bitboard Food Spawning Implementation

## Overview

Add optional food spawning logic to the bitboard implementation following the official Battlesnake rules from the [rules repository](https://github.com/BattlesnakeOfficial/rules).

## Battlesnake Food Spawning Rules

Based on the official rules:
- **Default minimum food**: 1
- **Default spawn chance**: 15% per turn
- **Spawning logic**:
  1. If current food count < `minimum_food`, spawn food until reaching minimum
  2. Otherwise, with `food_spawn_chance`% probability, spawn exactly 1 food
- **Valid spawn locations**: Any cell not occupied by snake bodies or existing food

## Implementation Steps

### Step 1: Add Food Spawning Configuration to Game struct

**File**: `src/bitboard.rs`

Add configuration fields to the `Game` struct:

```rust
#[derive(Debug, Clone)]
pub struct Game {
    pub occupied: Board,
    pub food: Board,
    pub snakes: Vec<Snake>,
    pub turn: usize,
    pub width: usize,
    // New fields for food spawning
    pub food_spawn_chance: u8,    // Percentage (0-100), default 15
    pub minimum_food: usize,      // Default 1
}
```

Add constants for defaults:
```rust
pub const DEFAULT_FOOD_SPAWN_CHANCE: u8 = 15;
pub const DEFAULT_MINIMUM_FOOD: usize = 1;
```

### Step 2: Update Game::create() method

Modify `Game::create()` to accept optional food spawning parameters:

```rust
impl Game {
    pub fn create(snakes: Vec<Snake>, food: Vec<u128>, turn: usize, width: usize) -> Self {
        Self::create_with_settings(
            snakes,
            food,
            turn,
            width,
            DEFAULT_FOOD_SPAWN_CHANCE,
            DEFAULT_MINIMUM_FOOD
        )
    }

    pub fn create_with_settings(
        snakes: Vec<Snake>,
        food: Vec<u128>,
        turn: usize,
        width: usize,
        food_spawn_chance: u8,
        minimum_food: usize,
    ) -> Self {
        let occupied = snakes.iter().fold(0, |a, s| a | s.body_board);
        let food = food.iter().fold(0, |a, s| set_index(a, *s, 1));

        Self {
            occupied,
            turn,
            food,
            width,
            snakes,
            food_spawn_chance,
            minimum_food,
        }
    }
}
```

### Step 3: Implement get_unoccupied_points() helper function

Add a method to find all valid spawn locations:

```rust
impl Game {
    /// Returns a vector of all unoccupied cell indices.
    /// A cell is unoccupied if it's not part of any snake body and doesn't have food.
    fn get_unoccupied_points(&self) -> Vec<u128> {
        let mut unoccupied = Vec::new();
        let board_size = (self.width * self.width) as u128;

        // Combined bitmap of all occupied cells (snakes + food)
        let occupied_or_food = self.occupied | self.food;

        for pos in 0..board_size {
            if (occupied_or_food >> pos) & 1 == 0 {
                unoccupied.push(pos);
            }
        }

        unoccupied
    }

    /// Counts the number of food items currently on the board.
    fn count_food(&self) -> usize {
        self.food.count_ones() as usize
    }
}
```

### Step 4: Implement spawn_food() method

Add the core food spawning logic:

```rust
impl Game {
    /// Spawns food according to Battlesnake rules.
    /// - If food count < minimum_food, spawn to reach minimum
    /// - Otherwise, food_spawn_chance% probability to spawn 1 food
    ///
    /// Uses the provided RNG for deterministic testing or game simulation.
    pub fn spawn_food<R: Rng>(&mut self, rng: &mut R) {
        let current_food = self.count_food();

        let food_to_spawn = if current_food < self.minimum_food {
            // Spawn enough to reach minimum
            self.minimum_food - current_food
        } else if self.food_spawn_chance > 0 {
            // Random chance to spawn 1 food
            let roll: u8 = rng.gen_range(0..100);
            if roll < self.food_spawn_chance { 1 } else { 0 }
        } else {
            0
        };

        if food_to_spawn == 0 {
            return;
        }

        let mut unoccupied = self.get_unoccupied_points();

        if unoccupied.is_empty() {
            return; // No valid spawn locations
        }

        // Shuffle for random placement
        unoccupied.shuffle(rng);

        // Spawn food at the first N unoccupied positions
        for pos in unoccupied.into_iter().take(food_to_spawn) {
            self.food = set_index(self.food, pos, 1);
        }
    }
}
```

### Step 5: Add advance_turn_with_spawning() method

Create a variant of `advance_turn` that includes food spawning:

```rust
impl Game {
    /// Advances the game turn with optional food spawning.
    /// This follows the official Battlesnake turn order:
    /// 1. Move snakes
    /// 2. Check collisions/eliminations
    /// 3. Feed snakes
    /// 4. Spawn food (new)
    pub fn advance_turn_with_spawning<R: Rng>(&mut self, moves: Vec<Move>, rng: &mut R) {
        // Execute standard turn logic
        self.advance_turn(moves);

        // Spawn food after all other turn processing
        self.spawn_food(rng);
    }
}
```

### Step 6: Update battlesnake.rs to parse food settings

**File**: `src/battlesnake.rs`

Parse food spawning settings from the API request:

```rust
pub fn request_to_game(req: &MoveRequest) -> bitboard::Game {
    let food = req.board.food
        .iter()
        .map(|c| coord_to_index(c, req.board.width))
        .collect();

    let snakes = req.board.snakes
        .iter()
        .map(|bs| battlesnake_to_snake(bs, req.board.width))
        .collect();

    // Extract food settings from ruleset if available
    let food_spawn_chance = req.game.ruleset.settings
        .as_ref()
        .and_then(|s| s.food_spawn_chance)
        .unwrap_or(bitboard::DEFAULT_FOOD_SPAWN_CHANCE);

    let minimum_food = req.game.ruleset.settings
        .as_ref()
        .and_then(|s| s.minimum_food)
        .unwrap_or(bitboard::DEFAULT_MINIMUM_FOOD);

    bitboard::Game::create_with_settings(
        snakes,
        food,
        req.turn.try_into().unwrap(),
        req.board.width.try_into().unwrap(),
        food_spawn_chance,
        minimum_food,
    )
}
```

Add necessary struct fields for parsing (check if they already exist in API types):

```rust
#[derive(Deserialize, Debug, Clone)]
pub struct RulesetSettings {
    #[serde(rename = "foodSpawnChance")]
    pub food_spawn_chance: Option<u8>,
    #[serde(rename = "minimumFood")]
    pub minimum_food: Option<usize>,
    // ... other fields
}
```

### Step 7: Add unit tests

**File**: `src/bitboard.rs`

```rust
#[cfg(test)]
mod food_spawning_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn count_food_empty() {
        let g = Game::create(vec![], vec![], 0, 11);
        assert_eq!(g.count_food(), 0);
    }

    #[test]
    fn count_food_multiple() {
        let g = Game::create(vec![], vec![0, 5, 10], 0, 11);
        assert_eq!(g.count_food(), 3);
    }

    #[test]
    fn get_unoccupied_excludes_snakes() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let unoccupied = g.get_unoccupied_points();

        assert!(!unoccupied.contains(&0));
        assert!(!unoccupied.contains(&1));
        assert!(!unoccupied.contains(&2));
        assert!(unoccupied.contains(&3));
    }

    #[test]
    fn get_unoccupied_excludes_food() {
        let g = Game::create(vec![], vec![5, 10], 0, 11);
        let unoccupied = g.get_unoccupied_points();

        assert!(!unoccupied.contains(&5));
        assert!(!unoccupied.contains(&10));
        assert!(unoccupied.contains(&0));
    }

    #[test]
    fn spawn_food_reaches_minimum() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_settings(vec![], vec![], 0, 11, 0, 3);

        assert_eq!(g.count_food(), 0);
        g.spawn_food(&mut rng);
        assert_eq!(g.count_food(), 3);
    }

    #[test]
    fn spawn_food_no_spawn_when_at_minimum() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_settings(vec![], vec![0], 0, 11, 0, 1);

        assert_eq!(g.count_food(), 1);
        g.spawn_food(&mut rng);
        assert_eq!(g.count_food(), 1); // No spawn, chance is 0
    }

    #[test]
    fn spawn_food_chance_deterministic() {
        // Test with 100% spawn chance
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_settings(vec![], vec![0], 0, 11, 100, 1);

        g.spawn_food(&mut rng);
        assert_eq!(g.count_food(), 2); // Should always spawn with 100% chance
    }

    #[test]
    fn spawn_food_no_spawn_on_full_board() {
        let mut rng = StdRng::seed_from_u64(42);
        // Create a game where all cells are occupied by snake
        let body: Vec<u128> = (0..121).collect();
        let s = Snake::create(String::from("test"), 100, body);
        let mut g = Game::create_with_settings(vec![s], vec![], 0, 11, 100, 5);

        g.spawn_food(&mut rng);
        assert_eq!(g.count_food(), 0); // No valid spawn locations
    }

    #[test]
    fn advance_turn_with_spawning() {
        let mut rng = StdRng::seed_from_u64(42);
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let mut g = Game::create_with_settings(vec![s], vec![], 0, 11, 100, 1);

        g.advance_turn_with_spawning(
            vec![(String::from("test"), Direction::Up)],
            &mut rng
        );

        // Turn advanced and food spawned
        assert_eq!(g.turn, 1);
        assert!(g.count_food() >= 1);
    }
}
```

### Step 8: Integration with RHEA (optional consideration)

**File**: `src/rhea.rs`

Consider how food spawning affects the evolutionary algorithm:

1. **Option A - Disable during simulation**: Keep food spawning disabled during RHEA simulations for consistency and speed
2. **Option B - Enable with seeded RNG**: Use deterministic RNG for reproducible simulations

Recommendation: Start with Option A (disabled) to avoid adding randomness to fitness evaluation, which could make evolution less stable.

```rust
// In RHEA simulation, use advance_turn() without spawning
// Food spawning only matters for the real game state
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/bitboard.rs` | Add config fields, spawn_food(), get_unoccupied_points(), count_food(), advance_turn_with_spawning(), new tests |
| `src/battlesnake.rs` | Parse food settings from API, update request_to_game() |

## Testing Strategy

1. Unit tests for individual functions (count_food, get_unoccupied_points, spawn_food)
2. Integration tests for advance_turn_with_spawning
3. Deterministic tests using seeded RNG (StdRng::seed_from_u64)
4. Edge cases: full board, no snakes, zero spawn chance

## Backwards Compatibility

- Existing `Game::create()` signature unchanged (uses defaults)
- Existing `advance_turn()` unchanged (no spawning)
- New functionality opt-in via `create_with_settings()` and `advance_turn_with_spawning()`

## Dependencies

May need to add `rand` shuffle capability:
```rust
use rand::seq::SliceRandom; // For shuffle
```

This is already available since the project uses `rand` crate.
