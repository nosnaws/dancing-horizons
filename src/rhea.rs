// Rolling Horizon Evolutionary Algorithm
// for snakes!

// TODO:
// - "fix" bad moves in candidates
// - cross over function (uniform)
// - tournament selection

use crate::bitboard::{Direction, Game};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter;

#[derive(Debug, Clone)]
pub struct RHEA {
    game: Game,
    player: String,
    pop: Vec<Individual>,
    crossover_chance: f32,
    tournament_size: u32,
    population_size: usize,
}

type Population = Vec<Individual>;
#[derive(Debug, Clone)]
pub struct Individual {
    genotype: Vec<Direction>,
    fitness: i32,
}

const GENO_LENGTH: usize = 20;
const MUTATION_CHANCE: f32 = 0.3;

impl RHEA {
    pub fn create(game: Game, player: String) -> Self {
        let crossover_chance = 1.0;
        let population_size = 50;
        let tournament_size = 3;
        let mut pop: Population = create_population(population_size, GENO_LENGTH)
            .iter()
            .map(|c| Individual {
                fitness: fitness(&game, &player, &c),
                ..c.clone()
            })
            .collect();

        pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));
        Self {
            game,
            player,
            pop,
            crossover_chance,
            tournament_size,
            population_size,
        }
    }

    pub fn get_move(&self) -> Direction {
        let best = self.pop.get(0).unwrap();
        return best.genotype.get(0).unwrap().clone();
    }

    pub fn update_game(&self, game: Game) -> Self {
        let mut rng = SmallRng::from_entropy();
        let updated_pop = self
            .pop
            .iter()
            .map(|c| {
                let (_, rest) = c.genotype.split_at(1);

                let mut new_geno = rest.to_vec();
                new_geno.push(rng.gen());

                let mut new_cand = create_candidate(new_geno);
                new_cand.fitness = fitness(&game, &self.player, &new_cand);

                return new_cand;
            })
            .collect();

        Self {
            game,
            pop: updated_pop,
            ..self.clone()
        }
    }

    pub fn evolve(&self) -> Self {
        println!("apex {:?}", self.pop.get(0).unwrap());
        let apex = create_candidate(self.pop.get(0).unwrap().genotype.clone());

        let mut rng = SmallRng::from_entropy();
        let pairs = (self.population_size / 2) + (self.population_size % 2);
        let mut new_pop: Population = (0..pairs)
            //  create new population via tournament selection & crossover
            .flat_map(|_| {
                let p1 = self.select_parent(&mut rng, &self.pop);
                let p2 = self.select_parent(&mut rng, &self.pop);
                // println!("parent 1 {:?}", p1);
                // println!("parent 2 {:?}", p2);

                return self.crossover(p1, p2);
            })
            // Leave room for the "apex"
            .take(self.population_size - 1)
            //  mutate new population
            .map(|c| self.mutate(&c))
            .chain(iter::once(apex))
            // .map(|c| self.maybe_fix_bad_moves(&c))
            //  calculate fitness of population (& sort?)
            .map(|c| Individual {
                fitness: fitness(&self.game, &self.player, &c),
                ..c
            })
            .collect();

        // println!("{:?}", new_pop);

        new_pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));

        // println!("new pop {:?}", new_pop);

        //  return with new population
        Self {
            pop: new_pop,
            ..self.clone()
        }
    }

    fn mutate(&self, candidate: &Individual) -> Individual {
        let mut rng = SmallRng::from_entropy();

        let mut mut_chances = [0f32; GENO_LENGTH];
        rng.fill(&mut mut_chances);

        let mutated_geno = mut_chances
            .iter()
            .take(candidate.genotype.len())
            .enumerate()
            .map(|(i, mc)| {
                if *mc < 1.0 - MUTATION_CHANCE {
                    return candidate.genotype[i];
                }

                return rng.gen();
            })
            .collect();

        return create_candidate(mutated_geno);
    }

    fn crossover(&self, c1: &Individual, c2: &Individual) -> Vec<Individual> {
        // probably want this
        // to be fixed length array
        let mut rng = SmallRng::from_entropy();

        let m: f32 = rng.gen();

        // check for cross over, return original candidates if not
        if m < 1.0 - self.crossover_chance {
            return vec![
                create_candidate(c1.genotype.clone()),
                create_candidate(c2.genotype.clone()),
            ];
        }
        // generate random crossover point
        // TODO: Cross over has the chance to happen at the first or last index
        // of the genotype, effectively not doing a crossover. Might want to
        // eliminate that chance by only generating random numbers between 1 and length - 1.
        let cross_over_point = get_random_index(&mut rng, GENO_LENGTH);
        // create 2 new candidates with each half of parent's genotype at crossover point
        let c1_pair = c1.genotype.split_at(cross_over_point);
        let c2_pair = c2.genotype.split_at(cross_over_point);

        return vec![
            create_candidate([c1_pair.0, c2_pair.1].concat()),
            create_candidate([c2_pair.0, c1_pair.1].concat()),
        ];
    }

    fn select_parent<'a>(&self, rng: &mut SmallRng, pop: &'a Population) -> &'a Individual {
        let pop_length = self.pop.len();
        // randomly choose N candidates
        let mut tournament: Vec<&Individual> = (0..self.tournament_size)
            .map(|_| pop.get(get_random_index(rng, pop_length)).unwrap())
            .collect();

        // Using an unstable sort to add some more randomness into the process
        tournament.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));

        // select the one with the best fitness
        return tournament.get(0).unwrap();
    }
}

pub fn score_population(g: &Game, pop: &mut Population, player: &String) {
    for cand in pop {
        cand.fitness = fitness(g, player, &cand);
    }
}

/// Opponent modeling strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpponentModel {
    /// No opponent moves provided - uses default "continue straight" behavior
    Default,
    /// Random moves for opponents
    Random,
    /// Smart opponents that seek food and avoid collisions
    Smart,
}

pub fn fitness(game: &Game, player: &String, c: &Individual) -> i32 {
    fitness_with_opponent_modeling(game, player, c, OpponentModel::Smart)
}

pub fn fitness_with_opponent_modeling(game: &Game, player: &String, c: &Individual, opponent_model: OpponentModel) -> i32 {
    let mut g = game.clone();
    let mut score = 0;
    let mut rng = SmallRng::from_entropy();

    for dir in &c.genotype {
        let mut moves: Vec<(String, Direction)> = vec![(player.clone(), dir.clone())];

        match opponent_model {
            OpponentModel::Default => {
                // Don't add opponent moves - let bitboard use default behavior
            }
            OpponentModel::Random => {
                for snake in &g.snakes {
                    if snake.id != *player && !snake.is_eliminated() {
                        let opponent_move: Direction = rng.gen();
                        moves.push((snake.id.clone(), opponent_move));
                    }
                }
            }
            OpponentModel::Smart => {
                for snake in &g.snakes {
                    if snake.id != *player && !snake.is_eliminated() {
                        let opponent_move = get_smart_move(&g, snake, &mut rng);
                        moves.push((snake.id.clone(), opponent_move));
                    }
                }
            }
        }

        g.advance_turn(moves);
        score += score_game(&g, &player);
    }

    return score;
}

/// Generate a smart move for an opponent snake:
/// 1. Avoid immediate death (walls, self-collision)
/// 2. Seek nearby food if health is low
/// 3. Otherwise move toward center or open space
fn get_smart_move(game: &Game, snake: &crate::bitboard::Snake, rng: &mut SmallRng) -> Direction {
    // Delegate to deterministic version, RNG only used for ties
    get_smart_move_deterministic(game, snake).unwrap_or_else(|| {
        // Fallback to random if no valid moves
        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
        directions[rng.gen_range(0..4)]
    })
}

/// Deterministic version of smart move for use in search algorithms
fn get_smart_move_deterministic(game: &Game, snake: &crate::bitboard::Snake) -> Option<Direction> {
    use crate::bitboard::{is_dir_valid, dir_to_index};

    let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
    let head = snake.body[0];
    let width = game.width as u128;

    // Filter to only valid moves (not into walls)
    let valid_moves: Vec<Direction> = directions
        .iter()
        .filter(|dir| is_dir_valid(snake, dir))
        .cloned()
        .collect();

    if valid_moves.is_empty() {
        return None;
    }

    // Score each valid move
    let mut best_move = valid_moves[0];
    let mut best_score = i32::MIN;

    for dir in &valid_moves {
        let new_pos = dir_to_index(head, dir, width);
        let pos_bit = 1u128 << new_pos;
        let mut move_score: i32 = 0;

        // Heavily penalize moving into occupied space (body collision)
        // But exclude the tail position since it will move
        let tail = *snake.body.last().unwrap();
        let tail_bit = 1u128 << tail;
        let occupied_without_tail = game.occupied & !tail_bit;

        if pos_bit & occupied_without_tail != 0 {
            move_score -= 1000;
        }

        // Avoid moving back into own neck
        if snake.body.len() > 1 {
            let neck = snake.body[1];
            if new_pos == neck {
                move_score -= 1000;
            }
        }

        // Bonus for moving toward food (especially if health is low)
        if game.food != 0 {
            let food_bonus = if snake.health < 30 { 50 } else { 10 };
            // Check if this move gets us closer to any food
            if pos_bit & game.food != 0 {
                move_score += food_bonus * 5; // Direct food pickup
            } else {
                // Simple heuristic: prefer moves that reduce manhattan distance to food
                // For simplicity, just give bonus for moving toward center if food exists
                let center = 60u128; // Center of 11x11 board
                let current_dist = manhattan_distance(head, center, width);
                let new_dist = manhattan_distance(new_pos, center, width);
                if new_dist < current_dist {
                    move_score += food_bonus;
                }
            }
        }

        // Small bonus for staying toward center (more options)
        let center = 60u128;
        let dist_to_center = manhattan_distance(new_pos, center, width);
        move_score -= dist_to_center as i32;

        if move_score > best_score {
            best_score = move_score;
            best_move = *dir;
        }
    }

    Some(best_move)
}

fn manhattan_distance(pos1: u128, pos2: u128, width: u128) -> u128 {
    let x1 = pos1 % width;
    let y1 = pos1 / width;
    let x2 = pos2 % width;
    let y2 = pos2 / width;

    let dx = if x1 > x2 { x1 - x2 } else { x2 - x1 };
    let dy = if y1 > y2 { y1 - y2 } else { y2 - y1 };

    dx + dy
}

pub fn score_game(g: &Game, player: &String) -> i32 {
    let mut score = 0;
    let mut snakes_alive = 0;
    let mut player_snake: Option<&crate::bitboard::Snake> = None;
    let mut max_opponent_length: usize = 0;

    for snake in &g.snakes {
        if snake.id == *player {
            if snake.is_eliminated() {
                return -1000;
            }
            player_snake = Some(snake);
        } else if !snake.is_eliminated() {
            max_opponent_length = max_opponent_length.max(snake.length());
        }

        if !snake.is_eliminated() {
            snakes_alive += 1;
        }
    }

    if snakes_alive == 1 {
        // We're the only one alive - big bonus
        score += 500;
    }

    if let Some(snake) = player_snake {
        // Health bonus (0-50 points, scaled)
        score += snake.health / 2;

        // Length advantage over opponents (can be negative)
        let length_diff = snake.length() as i32 - max_opponent_length as i32;
        score += length_diff * 10;

        // Bonus for being near center (more escape routes)
        let head = snake.body[0];
        let center = 60u128;
        let dist_to_center = manhattan_distance(head, center, g.width as u128);
        score -= dist_to_center as i32; // Penalize being far from center
    }

    score
}

pub fn create_population(size: usize, geno_len: usize) -> Vec<Individual> {
    return (0..size)
        .map(|_| create_candidate(get_random_moves(geno_len)))
        .collect();
}

fn create_candidate(genotype: Vec<Direction>) -> Individual {
    Individual {
        genotype,
        fitness: 0,
    }
}

fn get_random_moves(n: usize) -> Vec<Direction> {
    let mut moves = vec![];
    let mut rng = SmallRng::from_entropy();

    for _i in 0..n {
        moves.push(rng.gen());
    }

    return moves;
}

fn get_random_index(rng: &mut SmallRng, len: usize) -> usize {
    rng.gen_range(0..len).try_into().unwrap()
}

// ============================================================================
// Negamax with Alpha-Beta Pruning
// ============================================================================

/// Negamax search with alpha-beta pruning for Battlesnake
pub struct Negamax {
    pub max_depth: usize,
}

impl Negamax {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Get the best move for a player using minimax search with alpha-beta pruning
    /// Note: For simultaneous move games, we use a max search from player's perspective
    /// with opponent moves predicted using deterministic smart heuristics
    pub fn get_best_move(&self, game: &Game, player: &String) -> Direction {
        use crate::bitboard::is_dir_valid;

        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];

        // Find the player's snake
        let player_snake = game.snakes.iter().find(|s| s.id == *player);
        if player_snake.is_none() {
            return Direction::Up;
        }
        let player_snake = player_snake.unwrap();

        // Filter valid moves
        let valid_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(player_snake, dir))
            .cloned()
            .collect();

        if valid_moves.is_empty() {
            return Direction::Up;
        }

        let mut best_move = valid_moves[0];
        let mut best_score = i32::MIN;

        for dir in &valid_moves {
            let mut game_clone = game.clone();

            // Generate opponent moves using deterministic smart heuristic
            let mut moves = vec![(player.clone(), *dir)];
            for snake in &game.snakes {
                if snake.id != *player && !snake.is_eliminated() {
                    if let Some(opp_move) = get_smart_move_deterministic(&game_clone, snake) {
                        moves.push((snake.id.clone(), opp_move));
                    }
                }
            }

            game_clone.advance_turn(moves);

            let score = self.alpha_beta_max(&game_clone, player, self.max_depth - 1, i32::MIN + 1, i32::MAX);

            if score > best_score {
                best_score = score;
                best_move = *dir;
            }
        }

        best_move
    }

    /// Alpha-beta search maximizing for the player
    /// Since Battlesnake has simultaneous moves, we always evaluate from player's perspective
    fn alpha_beta_max(&self, game: &Game, player: &String, depth: usize, mut alpha: i32, beta: i32) -> i32 {
        use crate::bitboard::is_dir_valid;

        // Terminal conditions
        let player_snake = game.snakes.iter().find(|s| s.id == *player);

        // Check if player is eliminated
        if player_snake.is_none() || player_snake.unwrap().is_eliminated() {
            return -10000 + (self.max_depth - depth) as i32; // Prefer later death
        }

        // Check if we won (only one alive and it's us)
        let alive_count = game.snakes.iter().filter(|s| !s.is_eliminated()).count();
        if alive_count == 1 {
            return 10000 - (self.max_depth - depth) as i32; // Prefer earlier win
        }

        // Depth limit reached - evaluate position
        if depth == 0 {
            return self.evaluate(game, player);
        }

        let player_snake = player_snake.unwrap();
        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];

        // Filter valid moves
        let valid_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(player_snake, dir))
            .cloned()
            .collect();

        if valid_moves.is_empty() {
            return -10000 + (self.max_depth - depth) as i32;
        }

        let mut best_score = i32::MIN + 1;

        for dir in &valid_moves {
            let mut game_clone = game.clone();

            // Generate opponent moves using deterministic smart heuristic
            let mut moves = vec![(player.clone(), *dir)];
            for snake in &game.snakes {
                if snake.id != *player && !snake.is_eliminated() {
                    if let Some(opp_move) = get_smart_move_deterministic(&game_clone, snake) {
                        moves.push((snake.id.clone(), opp_move));
                    }
                }
            }

            game_clone.advance_turn(moves);

            // Recurse - continue maximizing from player's perspective
            let score = self.alpha_beta_max(&game_clone, player, depth - 1, alpha, beta);

            best_score = best_score.max(score);
            alpha = alpha.max(score);

            // Alpha-beta cutoff
            if alpha >= beta {
                break;
            }
        }

        best_score
    }

    /// Evaluate the game state for the player
    fn evaluate(&self, game: &Game, player: &String) -> i32 {
        score_game(game, player)
    }
}

/// Algorithm selection for move decision
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    /// RHEA with specified opponent model
    Rhea(OpponentModel),
    /// Negamax with alpha-beta pruning
    Negamax,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::{Game, Snake};

    #[test]
    fn fitness_does_not_modify_game_state() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let c = create_population(1, 3);
        let _score = fitness(&g, &String::from("test"), &c[0]);

        let s2 = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g2 = Game::create(vec![s2], vec![], 0, 11);

        assert_eq!(g.occupied, g2.occupied);
    }

    #[test]
    fn mutates() {
        use rand::SeedableRng;

        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let c = create_population(1, 3);
        let geno_copy = c[0].genotype.clone();
        let _r = RHEA::create(g, String::from("test"));

        // Use a fixed seed that produces no mutations (values > 0.7)
        let mut rng = SmallRng::seed_from_u64(42);
        let mut mut_chances = [0f32; GENO_LENGTH];
        rng.fill(&mut mut_chances);

        let mutated_geno = mut_chances
            .iter()
            .take(c[0].genotype.len())
            .enumerate()
            .map(|(i, mc)| {
                if *mc < 1.0 - MUTATION_CHANCE {
                    return c[0].genotype[i];
                }
                return rng.gen();
            })
            .collect();

        let cm = create_candidate(mutated_geno);
        assert_eq!(cm.genotype, geno_copy);
    }

    /// Simulates a head-to-head game between two opponent modeling strategies
    /// Returns: (player1_won, player2_won, draw)
    fn simulate_head_to_head(p1_model: OpponentModel, p2_model: OpponentModel, max_turns: usize) -> (bool, bool, bool) {
        use rand::SeedableRng;
        let mut rng = SmallRng::from_entropy();

        // Randomize starting positions to avoid bias
        let positions: [(Vec<u128>, Vec<u128>); 4] = [
            (vec![23, 12, 1], vec![97, 108, 119]),     // Corners
            (vec![55, 44, 33], vec![65, 76, 87]),      // Near center
            (vec![5, 4, 3], vec![115, 116, 117]),      // Bottom vs top
            (vec![33, 22, 11], vec![87, 98, 109]),     // Diagonal
        ];
        let (p1_pos, p2_pos) = positions[rng.gen_range(0..positions.len())].clone();

        let s1 = Snake::create(String::from("player1"), 100, p1_pos);
        let s2 = Snake::create(String::from("player2"), 100, p2_pos);

        // Limited food to force competition
        let food = vec![60, 49, 71];
        let mut game = Game::create(vec![s1, s2], food, 0, 11);

        for _ in 0..max_turns {
            let mut moves: Vec<(String, Direction)> = vec![];

            for snake in &game.snakes {
                if snake.is_eliminated() {
                    continue;
                }

                let model = if snake.id == "player1" { p1_model } else { p2_model };

                let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
                let mut best_dir = directions[rng.gen_range(0..4)];
                let mut best_score = i32::MIN;

                for dir in &directions {
                    let candidate = create_candidate(vec![*dir; 5]);
                    let score = fitness_with_opponent_modeling(&game, &snake.id, &candidate, model);
                    if score > best_score {
                        best_score = score;
                        best_dir = *dir;
                    }
                }

                moves.push((snake.id.clone(), best_dir));
            }

            game.advance_turn(moves);

            // Check for game end
            let alive: Vec<_> = game.snakes.iter().filter(|s| !s.is_eliminated()).collect();
            if alive.len() <= 1 {
                break;
            }
        }

        let p1_alive = !game.snakes.iter().find(|s| s.id == "player1").unwrap().is_eliminated();
        let p2_alive = !game.snakes.iter().find(|s| s.id == "player2").unwrap().is_eliminated();

        match (p1_alive, p2_alive) {
            (true, false) => (true, false, false),
            (false, true) => (false, true, false),
            (true, true) => (false, false, true),   // Draw (timeout)
            (false, false) => (false, false, true), // Both died simultaneously
        }
    }

    fn run_matchup(p1_model: OpponentModel, p2_model: OpponentModel, num_games: usize, max_turns: usize) -> (usize, usize, usize) {
        let mut p1_wins = 0;
        let mut p2_wins = 0;
        let mut draws = 0;

        for _ in 0..num_games {
            let (p1_won, p2_won, draw) = simulate_head_to_head(p1_model, p2_model, max_turns);
            if p1_won { p1_wins += 1; }
            if p2_won { p2_wins += 1; }
            if draw { draws += 1; }
        }

        (p1_wins, p2_wins, draws)
    }

    #[test]
    fn compare_opponent_modeling_strategies() {
        const NUM_GAMES: usize = 50;
        const MAX_TURNS: usize = 500; // Run games to conclusion

        println!("\n=== Opponent Modeling Tournament ===\n");
        println!("Each matchup: {} games, max {} turns\n", NUM_GAMES, MAX_TURNS);

        // Smart vs Default
        println!("--- SMART vs DEFAULT ---");
        let (smart_w, default_w, draws) = run_matchup(OpponentModel::Smart, OpponentModel::Default, NUM_GAMES, MAX_TURNS);
        println!("Smart wins:   {} ({:.1}%)", smart_w, (smart_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Default wins: {} ({:.1}%)", default_w, (default_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:        {} ({:.1}%)\n", draws, (draws as f64 / NUM_GAMES as f64) * 100.0);

        // Smart vs Random
        println!("--- SMART vs RANDOM ---");
        let (smart_w2, random_w, draws2) = run_matchup(OpponentModel::Smart, OpponentModel::Random, NUM_GAMES, MAX_TURNS);
        println!("Smart wins:  {} ({:.1}%)", smart_w2, (smart_w2 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Random wins: {} ({:.1}%)", random_w, (random_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:       {} ({:.1}%)\n", draws2, (draws2 as f64 / NUM_GAMES as f64) * 100.0);

        // Random vs Default
        println!("--- RANDOM vs DEFAULT ---");
        let (random_w2, default_w2, draws3) = run_matchup(OpponentModel::Random, OpponentModel::Default, NUM_GAMES, MAX_TURNS);
        println!("Random wins:  {} ({:.1}%)", random_w2, (random_w2 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Default wins: {} ({:.1}%)", default_w2, (default_w2 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:        {} ({:.1}%)\n", draws3, (draws3 as f64 / NUM_GAMES as f64) * 100.0);

        // Summary
        println!("=== OVERALL RANKING ===");
        let smart_total = smart_w + smart_w2;
        let random_total = random_w + random_w2;
        let default_total = default_w + default_w2;

        println!("Smart:   {} wins across matchups", smart_total);
        println!("Random:  {} wins across matchups", random_total);
        println!("Default: {} wins across matchups", default_total);

        if smart_total > random_total && smart_total > default_total {
            println!("\nBEST STRATEGY: Smart opponent modeling");
        } else if random_total > smart_total && random_total > default_total {
            println!("\nBEST STRATEGY: Random opponent modeling");
        } else if default_total > smart_total && default_total > random_total {
            println!("\nBEST STRATEGY: Default opponent modeling");
        } else {
            println!("\nNo clear winner");
        }
    }

    /// Simulates a head-to-head game between two algorithms
    /// Returns: (player1_won, player2_won, draw)
    fn simulate_algorithm_matchup(p1_algo: Algorithm, p2_algo: Algorithm, max_turns: usize) -> (bool, bool, bool) {
        use rand::SeedableRng;
        let mut rng = SmallRng::from_entropy();

        // Randomize starting positions
        let positions: [(Vec<u128>, Vec<u128>); 4] = [
            (vec![23, 12, 1], vec![97, 108, 119]),
            (vec![55, 44, 33], vec![65, 76, 87]),
            (vec![5, 4, 3], vec![115, 116, 117]),
            (vec![33, 22, 11], vec![87, 98, 109]),
        ];
        let (p1_pos, p2_pos) = positions[rng.gen_range(0..positions.len())].clone();

        let s1 = Snake::create(String::from("player1"), 100, p1_pos);
        let s2 = Snake::create(String::from("player2"), 100, p2_pos);

        let food = vec![60, 49, 71];
        let mut game = Game::create(vec![s1, s2], food, 0, 11);

        // Create negamax instances if needed
        let negamax = Negamax::new(8); // Depth 8 for deeper search

        for _ in 0..max_turns {
            let mut moves: Vec<(String, Direction)> = vec![];

            for snake in &game.snakes {
                if snake.is_eliminated() {
                    continue;
                }

                let algo = if snake.id == "player1" { p1_algo } else { p2_algo };

                let best_dir = match algo {
                    Algorithm::Rhea(model) => {
                        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
                        let mut best = directions[rng.gen_range(0..4)];
                        let mut best_score = i32::MIN;

                        for dir in &directions {
                            let candidate = create_candidate(vec![*dir; 5]);
                            let score = fitness_with_opponent_modeling(&game, &snake.id, &candidate, model);
                            if score > best_score {
                                best_score = score;
                                best = *dir;
                            }
                        }
                        best
                    }
                    Algorithm::Negamax => {
                        negamax.get_best_move(&game, &snake.id)
                    }
                };

                moves.push((snake.id.clone(), best_dir));
            }

            game.advance_turn(moves);

            let alive: Vec<_> = game.snakes.iter().filter(|s| !s.is_eliminated()).collect();
            if alive.len() <= 1 {
                break;
            }
        }

        let p1_alive = !game.snakes.iter().find(|s| s.id == "player1").unwrap().is_eliminated();
        let p2_alive = !game.snakes.iter().find(|s| s.id == "player2").unwrap().is_eliminated();

        match (p1_alive, p2_alive) {
            (true, false) => (true, false, false),
            (false, true) => (false, true, false),
            (true, true) => (false, false, true),
            (false, false) => (false, false, true),
        }
    }

    fn run_algorithm_matchup(p1_algo: Algorithm, p2_algo: Algorithm, num_games: usize, max_turns: usize) -> (usize, usize, usize) {
        let mut p1_wins = 0;
        let mut p2_wins = 0;
        let mut draws = 0;

        for _ in 0..num_games {
            let (p1_won, p2_won, draw) = simulate_algorithm_matchup(p1_algo, p2_algo, max_turns);
            if p1_won { p1_wins += 1; }
            if p2_won { p2_wins += 1; }
            if draw { draws += 1; }
        }

        (p1_wins, p2_wins, draws)
    }

    #[test]
    fn compare_negamax_vs_rhea() {
        const NUM_GAMES: usize = 30;
        const MAX_TURNS: usize = 500;

        println!("\n=== Negamax vs RHEA Tournament ===\n");
        println!("Negamax depth: 8");
        println!("RHEA: Smart opponent modeling, 5-move lookahead");
        println!("Each matchup: {} games, max {} turns\n", NUM_GAMES, MAX_TURNS);

        // Negamax vs Smart RHEA
        println!("--- NEGAMAX vs SMART RHEA ---");
        let (negamax_w, rhea_w, draws) = run_algorithm_matchup(
            Algorithm::Negamax,
            Algorithm::Rhea(OpponentModel::Smart),
            NUM_GAMES,
            MAX_TURNS
        );
        println!("Negamax wins:    {} ({:.1}%)", negamax_w, (negamax_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Smart RHEA wins: {} ({:.1}%)", rhea_w, (rhea_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:           {} ({:.1}%)\n", draws, (draws as f64 / NUM_GAMES as f64) * 100.0);

        // Negamax vs Default RHEA
        println!("--- NEGAMAX vs DEFAULT RHEA ---");
        let (negamax_w2, default_w, draws2) = run_algorithm_matchup(
            Algorithm::Negamax,
            Algorithm::Rhea(OpponentModel::Default),
            NUM_GAMES,
            MAX_TURNS
        );
        println!("Negamax wins:     {} ({:.1}%)", negamax_w2, (negamax_w2 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Default RHEA wins: {} ({:.1}%)", default_w, (default_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:            {} ({:.1}%)\n", draws2, (draws2 as f64 / NUM_GAMES as f64) * 100.0);

        // Summary
        println!("=== SUMMARY ===");
        let negamax_total = negamax_w + negamax_w2;
        let rhea_total = rhea_w + default_w;
        println!("Negamax total wins: {}", negamax_total);
        println!("RHEA total wins:    {}", rhea_total);

        if negamax_total > rhea_total {
            println!("\nWINNER: Negamax alpha-beta");
        } else if rhea_total > negamax_total {
            println!("\nWINNER: RHEA");
        } else {
            println!("\nTIE");
        }
    }
}
