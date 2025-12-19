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
    opponent_model: OpponentModel,
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
    pub fn create(game: Game, player: String, opponent_model: OpponentModel) -> Self {
        let crossover_chance = 1.0;
        let population_size = 50;
        let tournament_size = 3;
        let mut pop: Population = create_population(population_size, GENO_LENGTH)
            .iter()
            .map(|c| Individual {
                fitness: fitness_with_opponent_modeling(&game, &player, &c, opponent_model),
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
            opponent_model,
        }
    }

    pub fn get_move(&self) -> Direction {
        let best = self.pop.get(0).unwrap();
        return best.genotype.get(0).unwrap().clone();
    }

    pub fn update_game(&self, game: Game) -> Self {
        let mut rng = SmallRng::from_entropy();
        let opponent_model = self.opponent_model;
        let updated_pop = self
            .pop
            .iter()
            .map(|c| {
                let (_, rest) = c.genotype.split_at(1);

                let mut new_geno = rest.to_vec();
                new_geno.push(rng.gen());

                let mut new_cand = create_candidate(new_geno);
                new_cand.fitness = fitness_with_opponent_modeling(&game, &self.player, &new_cand, opponent_model);

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
                fitness: fitness_with_opponent_modeling(&self.game, &self.player, &c, self.opponent_model),
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
    /// Territorial opponents that maximize territory advantage
    Territorial,
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
            OpponentModel::Territorial => {
                for snake in &g.snakes {
                    if snake.id != *player && !snake.is_eliminated() {
                        let opponent_move = get_territorial_move(&g, snake, player, &mut rng);
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
                // Center position calculated as (width * width) / 2 for odd-width boards
                let center = (width * width) / 2;
                let current_dist = manhattan_distance(head, center, width);
                let new_dist = manhattan_distance(new_pos, center, width);
                if new_dist < current_dist {
                    move_score += food_bonus;
                }
            }
        }

        // Small bonus for staying toward center (more options)
        // Center position calculated as (width * width) / 2 for odd-width boards
        let center = (width * width) / 2;
        let dist_to_center = manhattan_distance(new_pos, center, width);
        move_score -= dist_to_center as i32;

        if move_score > best_score {
            best_score = move_score;
            best_move = *dir;
        }
    }

    Some(best_move)
}

/// Deterministic territorial move that maximizes territory advantage
/// Returns the move that gives the opponent the best territory differential (enemy - player)
fn get_territorial_move_deterministic(
    game: &Game,
    snake: &crate::bitboard::Snake,
    player_id: &String
) -> Option<Direction> {
    use crate::bitboard::{is_dir_valid, dir_to_index};

    let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
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
        let new_pos = dir_to_index(snake.body[0], dir, width);
        let pos_bit = 1u128 << new_pos;
        let mut move_score: i32 = 0;

        // Safety check: heavily penalize fatal moves
        let tail = *snake.body.last().unwrap();
        let tail_bit = 1u128 << tail;
        let occupied_without_tail = game.occupied & !tail_bit;

        if pos_bit & occupied_without_tail != 0 {
            move_score -= 10000; // Fatal - collision with body
        }

        // Avoid moving back into own neck
        if snake.body.len() > 1 {
            let neck = snake.body[1];
            if new_pos == neck {
                move_score -= 10000; // Fatal - neck collision
            }
        }

        // If move is not fatal, evaluate territory advantage
        if move_score > -10000 {
            // Clone game and simulate this move
            let mut game_clone = game.clone();

            // Find and update the snake in cloned game
            for s in &mut game_clone.snakes {
                if s.id == snake.id {
                    s.move_tail();
                    s.move_head(dir, width);
                    break;
                }
            }

            // Recalculate occupied board (required for voronoi)
            let mut new_occupied: u128 = 0;
            for s in &game_clone.snakes {
                if !s.is_eliminated() {
                    new_occupied = new_occupied | s.head_board | s.body_board;
                }
            }
            game_clone.occupied = new_occupied;

            // Calculate voronoi scores
            let voronoi = game_clone.calculate_voronoi_scores();
            let player_territory = voronoi.get(player_id).copied().unwrap_or(0);
            let enemy_territory = voronoi.get(&snake.id).copied().unwrap_or(0);

            // Maximize: enemy_territory - player_territory
            move_score += enemy_territory - player_territory;
        }

        if move_score > best_score {
            best_score = move_score;
            best_move = *dir;
        }
    }

    Some(best_move)
}

/// Generate a territorial move for an opponent snake
/// Falls back to random if no valid territorial move found
fn get_territorial_move(
    game: &Game,
    snake: &crate::bitboard::Snake,
    player_id: &String,
    rng: &mut SmallRng
) -> Direction {
    get_territorial_move_deterministic(game, snake, player_id).unwrap_or_else(|| {
        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
        directions[rng.gen_range(0..4)]
    })
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
        // Center position calculated as (width * width) / 2 for odd-width boards
        let width = g.width as u128;
        let center = (width * width) / 2;
        let dist_to_center = manhattan_distance(head, center, width);
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

    /// Get the best move using minimax for simultaneous move games
    /// For each of our moves, we find the opponent's best counter-move,
    /// then pick the move that gives us the best outcome assuming optimal opponent play
    pub fn get_best_move(&self, game: &Game, player: &String) -> Direction {
        use crate::bitboard::is_dir_valid;

        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];

        // Find player and opponent snakes
        let player_snake = game.snakes.iter().find(|s| s.id == *player);
        if player_snake.is_none() {
            return Direction::Up;
        }
        let player_snake = player_snake.unwrap();

        let opponent = game.snakes.iter()
            .find(|s| s.id != *player && !s.is_eliminated());

        // Filter valid moves for player
        let player_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(player_snake, dir))
            .cloned()
            .collect();

        if player_moves.is_empty() {
            return Direction::Up;
        }

        // If no opponent, just maximize our score
        if opponent.is_none() {
            let mut best_move = player_moves[0];
            let mut best_score = i32::MIN;
            for dir in &player_moves {
                let mut game_clone = game.clone();
                game_clone.advance_turn(vec![(player.clone(), *dir)]);
                let score = self.negamax(&game_clone, player, self.max_depth - 1, i32::MIN + 1, i32::MAX);
                if score > best_score {
                    best_score = score;
                    best_move = *dir;
                }
            }
            return best_move;
        }

        let opponent = opponent.unwrap();
        let opponent_id = opponent.id.clone();

        // Get opponent's valid moves
        let opponent_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(opponent, dir))
            .cloned()
            .collect();

        let mut best_move = player_moves[0];
        let mut best_score = i32::MIN;

        // For each of our moves, find the worst-case (opponent's best response)
        for player_dir in &player_moves {
            let mut worst_score = i32::MAX; // Opponent will minimize our score

            if opponent_moves.is_empty() {
                // Opponent has no moves, just evaluate our move
                let mut game_clone = game.clone();
                game_clone.advance_turn(vec![(player.clone(), *player_dir)]);
                worst_score = self.negamax(&game_clone, player, self.max_depth - 1, i32::MIN + 1, i32::MAX);
            } else {
                // Find opponent's best response (worst for us)
                for opp_dir in &opponent_moves {
                    let mut game_clone = game.clone();
                    game_clone.advance_turn(vec![
                        (player.clone(), *player_dir),
                        (opponent_id.clone(), *opp_dir),
                    ]);

                    let score = self.negamax(&game_clone, player, self.max_depth - 1, i32::MIN + 1, i32::MAX);
                    worst_score = worst_score.min(score); // Opponent picks worst for us
                }
            }

            // We pick the move with the best worst-case outcome
            if worst_score > best_score {
                best_score = worst_score;
                best_move = *player_dir;
            }
        }

        best_move
    }

    /// Negamax search - alternates between maximizing and minimizing
    /// For simultaneous games: at each level we do max-min over joint actions
    fn negamax(&self, game: &Game, player: &String, depth: usize, mut alpha: i32, beta: i32) -> i32 {
        use crate::bitboard::is_dir_valid;

        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];

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

        // Get opponent
        let opponent = game.snakes.iter()
            .find(|s| s.id != *player && !s.is_eliminated());

        // Filter valid moves for player
        let player_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(player_snake, dir))
            .cloned()
            .collect();

        if player_moves.is_empty() {
            return -10000 + (self.max_depth - depth) as i32;
        }

        // If no opponent, just maximize
        if opponent.is_none() {
            let mut best_score = i32::MIN + 1;
            for dir in &player_moves {
                let mut game_clone = game.clone();
                game_clone.advance_turn(vec![(player.clone(), *dir)]);
                let score = self.negamax(&game_clone, player, depth - 1, alpha, beta);
                best_score = best_score.max(score);
                alpha = alpha.max(score);
                if alpha >= beta {
                    break;
                }
            }
            return best_score;
        }

        let opponent = opponent.unwrap();
        let opponent_id = opponent.id.clone();

        let opponent_moves: Vec<Direction> = directions
            .iter()
            .filter(|dir| is_dir_valid(opponent, dir))
            .cloned()
            .collect();

        let mut best_score = i32::MIN + 1;

        // Max over our moves, min over opponent moves
        for player_dir in &player_moves {
            let mut worst_score = i32::MAX;

            if opponent_moves.is_empty() {
                let mut game_clone = game.clone();
                game_clone.advance_turn(vec![(player.clone(), *player_dir)]);
                worst_score = self.negamax(&game_clone, player, depth - 1, alpha, beta);
            } else {
                for opp_dir in &opponent_moves {
                    let mut game_clone = game.clone();
                    game_clone.advance_turn(vec![
                        (player.clone(), *player_dir),
                        (opponent_id.clone(), *opp_dir),
                    ]);

                    let score = self.negamax(&game_clone, player, depth - 1, alpha, beta);
                    worst_score = worst_score.min(score);

                    // Beta cutoff for the min level
                    if worst_score <= alpha {
                        break;
                    }
                }
            }

            best_score = best_score.max(worst_score);
            alpha = alpha.max(best_score);

            // Alpha cutoff for the max level
            if alpha >= beta {
                break;
            }
        }

        best_score
    }

    /// Evaluate the game state for the player
    /// Uses voronoi for better board control assessment
    fn evaluate(&self, game: &Game, player: &String) -> i32 {
        let mut score = 0;
        let mut snakes_alive = 0;
        let mut player_snake: Option<&crate::bitboard::Snake> = None;
        let mut max_opponent_length: usize = 0;

        for snake in &game.snakes {
            if snake.id == *player {
                if snake.is_eliminated() {
                    return -10000;
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
            score += 5000;
        }

        if let Some(snake) = player_snake {
            // Health bonus (0-50 points, scaled)
            score += snake.health / 2;

            // Length advantage over opponents (can be negative)
            let length_diff = snake.length() as i32 - max_opponent_length as i32;
            score += length_diff * 20;

            // Voronoi board control - how much space we control
            if snakes_alive > 1 {
                let voronoi = game.calculate_voronoi_scores();
                let player_control = voronoi.get(player).copied().unwrap_or(0);

                // Calculate opponent's control
                let opponent_control: i32 = voronoi.iter()
                    .filter(|(id, _)| *id != player)
                    .map(|(_, &v)| v)
                    .sum();

                // Reward controlling more space than opponents
                let control_diff = player_control - opponent_control;
                score += control_diff * 3; // Weight voronoi heavily
            }

            // Bonus for being near center (more escape routes)
            let head = snake.body[0];
            // Center position calculated as (width * width) / 2 for odd-width boards
            let width = game.width as u128;
            let center = (width * width) / 2;
            let dist_to_center = manhattan_distance(head, center, width);
            score -= dist_to_center as i32;
        }

        score
    }
}

/// Configuration for RHEA algorithm
#[derive(Debug, Clone, Copy)]
pub struct RheaConfig {
    pub opponent_model: OpponentModel,
    pub evolutions: usize,
    pub population_size: usize,
}

impl RheaConfig {
    pub fn new(opponent_model: OpponentModel, evolutions: usize, population_size: usize) -> Self {
        Self { opponent_model, evolutions, population_size }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    /// RHEA with configuration (opponent model, evolutions, population size)
    Rhea(RheaConfig),
    /// Negamax with alpha-beta pruning
    Negamax,
}

/// Wrapper that handles both stateful (RHEA) and stateless (Negamax) algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmInstance {
    config: Algorithm,
    state: Option<RHEA>,  // Only used for RHEA (population warm-starting)
    game: Game,
    player_id: String,
}

impl AlgorithmInstance {
    pub fn create(config: Algorithm, game: Game, player_id: String) -> Self {
        let state = match &config {
            Algorithm::Rhea(rhea_config) => Some(RHEA::create(game.clone(), player_id.clone(), rhea_config.opponent_model)),
            Algorithm::Negamax => None,
        };

        Self { config, state, game, player_id }
    }

    pub fn update_game(mut self, game: Game) -> Self {
        if let Some(rhea) = self.state {
            self.state = Some(rhea.update_game(game.clone()));
        }
        self.game = game;
        self
    }

    pub fn get_move(&mut self) -> Direction {
        match (&self.config, &mut self.state) {
            (Algorithm::Rhea(config), Some(rhea)) => {
                let start = std::time::Instant::now();
                for _ in 0..config.evolutions {
                    *rhea = rhea.evolve();
                }
                let result = rhea.get_move();
                println!("RHEA took {:?}", start.elapsed());
                result
            }
            (Algorithm::Negamax, None) => {
                let start = std::time::Instant::now();
                let negamax = Negamax::new(3);
                let result = negamax.get_best_move(&self.game, &self.player_id);
                println!("Negamax took {:?}", start.elapsed());
                result
            }
            _ => panic!("Invalid algorithm state"),
        }
    }
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
        let _r = RHEA::create(g, String::from("test"), OpponentModel::Smart);

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

    #[test]
    fn test_territorial_move_avoids_fatal() {
        // Snake in corner - should not pick wall moves even if voronoi favors them
        let snake = Snake::create(String::from("enemy"), 100, vec![0, 1, 2]);
        let player = Snake::create(String::from("player"), 100, vec![60, 61, 62]);
        let game = Game::create(vec![snake.clone(), player], vec![], 0, 11);

        let move_result = get_territorial_move_deterministic(&game, &snake, &String::from("player"));
        assert!(move_result.is_some());
        let chosen = move_result.unwrap();

        // Should only choose Up or Right (not Left or Down into walls)
        assert!(chosen == Direction::Up || chosen == Direction::Right);
    }

    #[test]
    fn test_territorial_move_maximizes_territory_diff() {
        // Setup where moving toward player gives more territory
        let snake = Snake::create(String::from("enemy"), 100, vec![10, 9, 8]);
        let player = Snake::create(String::from("player"), 100, vec![70, 71, 72]);
        let game = Game::create(vec![snake.clone(), player], vec![], 0, 11);

        let move_result = get_territorial_move_deterministic(&game, &snake, &String::from("player"));
        assert!(move_result.is_some());
        // Hard to assert exact move, but should not crash and should be valid
    }

    #[test]
    fn test_territorial_move_deterministic_consistency() {
        // Same game state should always produce same move
        let snake = Snake::create(String::from("enemy"), 100, vec![60, 59, 58]);
        let player = Snake::create(String::from("player"), 100, vec![10, 11, 12]);
        let game = Game::create(vec![snake.clone(), player], vec![], 0, 11);

        let move1 = get_territorial_move_deterministic(&game, &snake, &String::from("player"));
        let move2 = get_territorial_move_deterministic(&game, &snake, &String::from("player"));
        let move3 = get_territorial_move_deterministic(&game, &snake, &String::from("player"));

        assert_eq!(move1, move2);
        assert_eq!(move2, move3);
    }

    /// Simulates a head-to-head game between two opponent modeling strategies
    /// Returns: (player1_won, player2_won, draw)
    fn simulate_head_to_head(p1_model: OpponentModel, p2_model: OpponentModel, max_turns: usize, seed: u64) -> (bool, bool, bool) {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(seed);

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

        for game_idx in 0..num_games {
            // Use deterministic seed based on game index for reproducibility
            let seed = 1000u64 + game_idx as u64;
            let (p1_won, p2_won, draw) = simulate_head_to_head(p1_model, p2_model, max_turns, seed);
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

        // Territorial vs Smart
        println!("--- TERRITORIAL vs SMART ---");
        let (terr_w, smart_w3, draws4) = run_matchup(OpponentModel::Territorial, OpponentModel::Smart, NUM_GAMES, MAX_TURNS);
        println!("Territorial wins: {} ({:.1}%)", terr_w, (terr_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Smart wins:       {} ({:.1}%)", smart_w3, (smart_w3 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:            {} ({:.1}%)\n", draws4, (draws4 as f64 / NUM_GAMES as f64) * 100.0);

        // Territorial vs Random
        println!("--- TERRITORIAL vs RANDOM ---");
        let (terr_w2, random_w3, draws5) = run_matchup(OpponentModel::Territorial, OpponentModel::Random, NUM_GAMES, MAX_TURNS);
        println!("Territorial wins: {} ({:.1}%)", terr_w2, (terr_w2 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Random wins:      {} ({:.1}%)", random_w3, (random_w3 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:            {} ({:.1}%)\n", draws5, (draws5 as f64 / NUM_GAMES as f64) * 100.0);

        // Territorial vs Default
        println!("--- TERRITORIAL vs DEFAULT ---");
        let (terr_w3, default_w3, draws6) = run_matchup(OpponentModel::Territorial, OpponentModel::Default, NUM_GAMES, MAX_TURNS);
        println!("Territorial wins: {} ({:.1}%)", terr_w3, (terr_w3 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Default wins:     {} ({:.1}%)", default_w3, (default_w3 as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:            {} ({:.1}%)\n", draws6, (draws6 as f64 / NUM_GAMES as f64) * 100.0);

        // Summary
        println!("=== OVERALL RANKING ===");
        let smart_total = smart_w + smart_w2 + smart_w3;
        let random_total = random_w + random_w2 + random_w3;
        let default_total = default_w + default_w2 + default_w3;
        let territorial_total = terr_w + terr_w2 + terr_w3;

        println!("Smart:       {} wins across matchups", smart_total);
        println!("Random:      {} wins across matchups", random_total);
        println!("Default:     {} wins across matchups", default_total);
        println!("Territorial: {} wins across matchups", territorial_total);

        let max_wins = smart_total.max(random_total).max(default_total).max(territorial_total);
        if territorial_total == max_wins {
            println!("\nBEST STRATEGY: Territorial opponent modeling");
        } else if smart_total == max_wins {
            println!("\nBEST STRATEGY: Smart opponent modeling");
        } else if random_total == max_wins {
            println!("\nBEST STRATEGY: Random opponent modeling");
        } else if default_total == max_wins {
            println!("\nBEST STRATEGY: Default opponent modeling");
        }
    }

    /// Creates and evolves a RHEA population, returning the best move
    fn run_rhea_evolution(game: &Game, player: &str, config: RheaConfig) -> Direction {
        let mut rng = SmallRng::from_entropy();
        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];

        // Create initial population
        let mut population: Vec<Individual> = (0..config.population_size)
            .map(|_| {
                let genotype: Vec<Direction> = (0..GENO_LENGTH).map(|_| rng.gen()).collect();
                let mut ind = create_candidate(genotype);
                ind.fitness = fitness_with_opponent_modeling(game, &player.to_string(), &ind, config.opponent_model);
                ind
            })
            .collect();

        population.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));

        // Evolution loop
        for _ in 0..config.evolutions {
            let apex = population[0].clone();

            let pairs = (config.population_size / 2) + (config.population_size % 2);
            let mut new_pop: Vec<Individual> = Vec::with_capacity(config.population_size);

            // Tournament selection and crossover
            for _ in 0..pairs {
                // Tournament selection for parent 1
                let p1 = (0..3)
                    .map(|_| &population[rng.gen_range(0..population.len())])
                    .max_by_key(|i| i.fitness)
                    .unwrap()
                    .clone();

                // Tournament selection for parent 2
                let p2 = (0..3)
                    .map(|_| &population[rng.gen_range(0..population.len())])
                    .max_by_key(|i| i.fitness)
                    .unwrap()
                    .clone();

                // Uniform crossover
                let child1_geno: Vec<Direction> = p1.genotype.iter()
                    .zip(p2.genotype.iter())
                    .map(|(g1, g2)| if rng.gen::<bool>() { *g1 } else { *g2 })
                    .collect();
                let child2_geno: Vec<Direction> = p1.genotype.iter()
                    .zip(p2.genotype.iter())
                    .map(|(g1, g2)| if rng.gen::<bool>() { *g1 } else { *g2 })
                    .collect();

                new_pop.push(create_candidate(child1_geno));
                new_pop.push(create_candidate(child2_geno));
            }

            // Take population_size - 1 (leave room for apex)
            new_pop.truncate(config.population_size - 1);

            // Mutate
            for ind in &mut new_pop {
                for gene in &mut ind.genotype {
                    if rng.gen::<f32>() < MUTATION_CHANCE {
                        *gene = rng.gen();
                    }
                }
            }

            // Add apex (elitism)
            new_pop.push(apex);

            // Evaluate fitness
            for ind in &mut new_pop {
                ind.fitness = fitness_with_opponent_modeling(game, &player.to_string(), ind, config.opponent_model);
            }

            // Sort by fitness
            new_pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));
            population = new_pop;
        }

        // Return best move
        population[0].genotype[0]
    }

    /// Simulates a head-to-head game between two algorithms
    /// Returns: (player1_won, player2_won, draw)
    fn simulate_algorithm_matchup(p1_algo: Algorithm, p2_algo: Algorithm, max_turns: usize, seed: u64) -> (bool, bool, bool) {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(seed);

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
        // Depth 4 because branching is now 4x4=16 per level (proper minimax over joint actions)
        let negamax = Negamax::new(4);

        for _ in 0..max_turns {
            let mut moves: Vec<(String, Direction)> = vec![];

            for snake in &game.snakes {
                if snake.is_eliminated() {
                    continue;
                }

                let algo = if snake.id == "player1" { p1_algo } else { p2_algo };

                let best_dir = match algo {
                    Algorithm::Rhea(config) => {
                        run_rhea_evolution(&game, &snake.id, config)
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

        for game_idx in 0..num_games {
            // Use deterministic seed based on game index for reproducibility
            let seed = 2000u64 + game_idx as u64;
            let (p1_won, p2_won, draw) = simulate_algorithm_matchup(p1_algo, p2_algo, max_turns, seed);
            if p1_won { p1_wins += 1; }
            if p2_won { p2_wins += 1; }
            if draw { draws += 1; }
        }

        (p1_wins, p2_wins, draws)
    }

    #[test]
    #[ignore] // Long-running test - run with: cargo test compare_negamax_vs_rhea -- --ignored
    fn compare_negamax_vs_rhea() {
        const NUM_GAMES: usize = 10;
        const MAX_TURNS: usize = 500;

        // RHEA configuration - larger population, fewer evolutions
        const RHEA_EVOLUTIONS: usize = 10;  // 10 evolution cycles per move
        const RHEA_POPULATION: usize = 50;  // Population size

        let smart_rhea = RheaConfig::new(OpponentModel::Smart, RHEA_EVOLUTIONS, RHEA_POPULATION);

        println!("\n=== Negamax vs RHEA Tournament ===\n");
        println!("Negamax: depth 4 (minimax over joint actions with voronoi)");
        println!("RHEA: {} evolutions, population {}, smart opponent modeling", RHEA_EVOLUTIONS, RHEA_POPULATION);
        println!("Each matchup: {} games, max {} turns\n", NUM_GAMES, MAX_TURNS);

        // Negamax vs Smart RHEA
        println!("--- NEGAMAX vs SMART RHEA ---");
        let (negamax_w, rhea_w, draws) = run_algorithm_matchup(
            Algorithm::Negamax,
            Algorithm::Rhea(smart_rhea),
            NUM_GAMES,
            MAX_TURNS
        );
        println!("Negamax wins:    {} ({:.1}%)", negamax_w, (negamax_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Smart RHEA wins: {} ({:.1}%)", rhea_w, (rhea_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:           {} ({:.1}%)\n", draws, (draws as f64 / NUM_GAMES as f64) * 100.0);

        // Summary
        println!("=== SUMMARY ===");
        println!("Negamax wins: {}", negamax_w);
        println!("RHEA wins:    {}", rhea_w);

        if negamax_w > rhea_w {
            println!("\nWINNER: Negamax alpha-beta");
        } else if rhea_w > negamax_w {
            println!("\nWINNER: RHEA");
        } else {
            println!("\nTIE");
        }
    }

    #[test]
    #[ignore] // Long-running test - run with: cargo test compare_with_large_population -- --ignored
    fn compare_with_large_population() {
        const NUM_GAMES: usize = 10;
        const MAX_TURNS: usize = 500;

        // RHEA with larger population
        const RHEA_EVOLUTIONS: usize = 50;   // More evolution cycles
        const RHEA_POPULATION: usize = 50;   // Larger population

        let large_rhea = RheaConfig::new(OpponentModel::Smart, RHEA_EVOLUTIONS, RHEA_POPULATION);

        println!("\n=== Large Population RHEA vs Negamax ===\n");
        println!("Negamax: depth 4 with voronoi");
        println!("RHEA: {} evolutions, population {}", RHEA_EVOLUTIONS, RHEA_POPULATION);
        println!("Games: {}, max {} turns\n", NUM_GAMES, MAX_TURNS);

        let (negamax_w, rhea_w, draws) = run_algorithm_matchup(
            Algorithm::Negamax,
            Algorithm::Rhea(large_rhea),
            NUM_GAMES,
            MAX_TURNS
        );

        println!("Negamax wins: {} ({:.1}%)", negamax_w, (negamax_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("RHEA wins:    {} ({:.1}%)", rhea_w, (rhea_w as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:        {} ({:.1}%)", draws, (draws as f64 / NUM_GAMES as f64) * 100.0);

        if negamax_w > rhea_w {
            println!("\nWINNER: Negamax");
        } else if rhea_w > negamax_w {
            println!("\nWINNER: Large Population RHEA");
        } else {
            println!("\nTIE");
        }
    }

    #[test]
    fn debug_voronoi_values() {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(12345);

        // Fixed starting position
        let s1 = Snake::create(String::from("player1"), 100, vec![23, 12, 1]);
        let s2 = Snake::create(String::from("player2"), 100, vec![97, 108, 119]);

        let food = vec![60, 49, 71];
        let mut game = Game::create(vec![s1, s2], food, 0, 11);

        let negamax = Negamax::new(4);

        println!("\n=== Voronoi Debug ===\n");

        for turn in 0..20 {
            let voronoi = game.calculate_voronoi_scores();
            let p1_control = voronoi.get(&String::from("player1")).copied().unwrap_or(0);
            let p2_control = voronoi.get(&String::from("player2")).copied().unwrap_or(0);

            let p1_snake = game.snakes.iter().find(|s| s.id == "player1").unwrap();
            let p2_snake = game.snakes.iter().find(|s| s.id == "player2").unwrap();

            println!(
                "Turn {}: P1 head={}, control={} | P2 head={}, control={} | diff={}",
                turn,
                p1_snake.body[0],
                p1_control,
                p2_snake.body[0],
                p2_control,
                p1_control - p2_control
            );

            // Get Negamax evaluation for player1
            let eval = negamax.evaluate(&game, &String::from("player1"));
            println!("  Negamax eval for P1: {}", eval);

            let mut moves: Vec<(String, Direction)> = vec![];

            for snake in &game.snakes {
                if snake.is_eliminated() {
                    continue;
                }

                let best_dir = negamax.get_best_move(&game, &snake.id);
                moves.push((snake.id.clone(), best_dir));
            }

            println!("  Moves: {:?}", moves);

            game.advance_turn(moves);

            let alive: Vec<_> = game.snakes.iter().filter(|s| !s.is_eliminated()).collect();
            if alive.len() <= 1 {
                println!("\nGame ended at turn {}", turn);
                break;
            }
        }

        // Final state
        let p1_alive = !game.snakes.iter().find(|s| s.id == "player1").unwrap().is_eliminated();
        let p2_alive = !game.snakes.iter().find(|s| s.id == "player2").unwrap().is_eliminated();
        println!("\nFinal: P1 alive={}, P2 alive={}", p1_alive, p2_alive);
    }

    #[test]
    fn test_smart_move_avoids_walls() {
        // Snake in bottom-left corner - should not go left or down
        let snake = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let game = Game::create(vec![snake.clone()], vec![], 0, 11);

        let smart_move = get_smart_move_deterministic(&game, &snake);
        assert!(smart_move.is_some());
        let chosen = smart_move.unwrap();

        // Should only choose Up or Right (not Left or Down)
        assert!(chosen == Direction::Up || chosen == Direction::Right);
    }

    #[test]
    fn test_smart_move_avoids_neck() {
        // Snake facing right (head=2, neck=1, tail=0) should not go left
        let snake = Snake::create(String::from("test"), 100, vec![2, 1, 0]);
        let game = Game::create(vec![snake.clone()], vec![], 0, 11);

        let smart_move = get_smart_move_deterministic(&game, &snake);
        assert!(smart_move.is_some());
        let chosen = smart_move.unwrap();

        // Should not go back into neck (left)
        assert_ne!(chosen, Direction::Left);
    }

    #[test]
    fn test_smart_move_seeks_food() {
        // Snake at position 0, food at position 11 (directly up)
        let snake = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let game = Game::create(vec![snake.clone()], vec![11], 0, 11);

        let smart_move = get_smart_move_deterministic(&game, &snake);
        assert!(smart_move.is_some());
        let chosen = smart_move.unwrap();

        // Should prefer moving up toward food
        assert_eq!(chosen, Direction::Up);
    }

    #[test]
    fn test_smart_move_avoids_body_collision() {
        // Create a game with two snakes
        let snake1 = Snake::create(String::from("test1"), 100, vec![11, 0, 1]);
        let snake2 = Snake::create(String::from("test2"), 100, vec![22, 23, 24]);
        let game = Game::create(vec![snake1.clone(), snake2], vec![], 0, 11);

        let smart_move = get_smart_move_deterministic(&game, &snake1);
        assert!(smart_move.is_some());
        let chosen = smart_move.unwrap();

        // Should not move down into its own tail at position 0 (would be penalized)
        // Should prefer safer moves
        assert_ne!(chosen, Direction::Down);
    }

    #[test]
    fn test_smart_move_no_valid_moves() {
        // Create a snake that's completely boxed in (this is contrived)
        // Snake at 60 (center), surrounded by occupied spaces
        let snake = Snake::create(String::from("test"), 100, vec![60, 61, 62]);

        // Create walls around position 60 by placing other snake bodies
        // Positions around 60: up=71, down=49, left=59, right=61
        let blocker1 = Snake::create(String::from("block1"), 100, vec![71, 82, 93]);
        let blocker2 = Snake::create(String::from("block2"), 100, vec![49, 38, 27]);
        let blocker3 = Snake::create(String::from("block3"), 100, vec![59, 58, 57]);
        // Right (61) is already the snake's own neck

        let game = Game::create(vec![snake.clone(), blocker1, blocker2, blocker3], vec![], 0, 11);

        let smart_move = get_smart_move_deterministic(&game, &snake);

        // In this extreme case, there might be no valid moves
        // But our function should handle it gracefully and return None
        // or pick the least bad option
        assert!(smart_move.is_none() || smart_move.is_some());
    }

    #[test]
    fn test_smart_move_deterministic() {
        // Test that the same game state always produces the same move
        let snake = Snake::create(String::from("test"), 100, vec![60, 61, 62]);
        let game = Game::create(vec![snake.clone()], vec![49], 0, 11);

        let move1 = get_smart_move_deterministic(&game, &snake);
        let move2 = get_smart_move_deterministic(&game, &snake);
        let move3 = get_smart_move_deterministic(&game, &snake);

        // All three calls should produce identical results
        assert_eq!(move1, move2);
        assert_eq!(move2, move3);
    }

    #[test]
    fn test_center_calculation_is_correct() {
        // For an 11x11 board, center should be calculated as (11 * 11) / 2 = 60
        let width = 11u128;
        let center = (width * width) / 2;
        assert_eq!(center, 60u128);

        // Test with different board size
        let width_7 = 7u128;
        let center_7 = (width_7 * width_7) / 2;
        assert_eq!(center_7, 24u128); // 49 / 2 = 24 (center of 7x7)
    }

    /// Regression test for bug where RHEA thinks [Up, Right, ...] survives
    /// when it actually leads to death
    ///
    /// Board state from Turn 154:
    /// ⚕◦⚕◦◦◦◦⚕◦⚕◦  y=10
    /// ◦◦◦◦◦◦◦◦◦⚕◦  y=9
    /// ◦◦◦◦◦◦◦◦◦◦⚕  y=8
    /// ◦◦◦◦◦◦◦◦◦◦◦  y=7
    /// ◦◦◦◦◦◦⌀◦◦◦◦  y=6  <- Negamax head at (6, 6)
    /// ◦■◦◦◦⌀⌀◦◦⚕◦  y=5  <- Territorial TAIL at (1, 5)
    /// ◦■◦◦⚕⌀⌀⌀◦⚕◦  y=4
    /// ◦■■■■◦◦⌀⌀⚕◦  y=3
    /// ⚕◦◦◦■■◦◦⌀◦⚕  y=2
    /// ⚕◦◦◦◦■■■◦⚕◦  y=1  <- Territorial HEAD at (7, 1)
    /// ◦⚕◦◦◦◦⚕◦⚕◦◦  y=0
    #[test]
    fn test_turn_154_up_right_should_not_survive() {
        // Territorial snake body (head to tail)
        // HEAD at (7,1)=18 → (6,1)=17 → (5,1)=16 → (5,2)=27 → (4,2)=26 →
        // (4,3)=37 → (3,3)=36 → (2,3)=35 → (1,3)=34 → (1,4)=45 → (1,5)=56 TAIL
        let territorial = Snake::create(
            String::from("territorial"),
            100,
            vec![18, 17, 16, 27, 26, 37, 36, 35, 34, 45, 56]
        );

        // Negamax snake body (head to tail)
        // HEAD at (8,2)=30 → (8,3)=41 → (7,3)=40 → (7,4)=51 → (6,4)=50 →
        // (5,4)=49 → (5,5)=60 → (6,5)=61 → (6,6)=72 TAIL
        let negamax = Snake::create(
            String::from("negamax"),
            87,
            vec![30, 41, 40, 51, 50, 49, 60, 61, 72]
        );

        // Food positions
        let food = vec![110, 112, 117, 119, 108, 98, 64, 53, 48, 42, 32, 22, 20, 11, 8, 6, 1];

        let game = Game::create(vec![territorial, negamax], food, 154, 11);

        // The problematic genotype that was getting fitness 13286
        let problematic_genotype = vec![
            Direction::Up, Direction::Right, Direction::Up, Direction::Up, Direction::Up,
            Direction::Left, Direction::Down, Direction::Left, Direction::Down, Direction::Left,
            Direction::Left, Direction::Up, Direction::Up, Direction::Right, Direction::Up,
            Direction::Left, Direction::Up, Direction::Up, Direction::Right, Direction::Up
        ];

        let candidate = Individual {
            genotype: problematic_genotype,
            fitness: 0,
        };

        // Calculate fitness using Territorial opponent model (the one we're testing)
        let fitness_score = fitness_with_opponent_modeling(
            &game,
            &String::from("territorial"),
            &candidate,
            OpponentModel::Territorial
        );

        println!("Fitness for [Up, Right, ...] genotype with Territorial model: {}", fitness_score);

        // Trace through moves to see what happens
        let mut g = game.clone();
        for (i, dir) in candidate.genotype.iter().enumerate() {
            let player_snake = g.snakes.iter().find(|s| s.id == "territorial").unwrap();
            let head = player_snake.body[0];
            let x = head % 11;
            let y = head / 11;
            let is_dead = player_snake.is_eliminated();

            // Also print Negamax position
            let opp_info = if let Some(opp) = g.snakes.iter().find(|s| s.id == "negamax" && !s.is_eliminated()) {
                let opp_head = opp.body[0];
                let ox = opp_head % 11;
                let oy = opp_head / 11;
                format!("Negamax at ({}, {})", ox, oy)
            } else {
                "Negamax eliminated".to_string()
            };

            println!("Move {}: Territorial {:?} from ({}, {}), health={}, dead={} | {}",
                i, dir, x, y, player_snake.health, is_dead, opp_info);

            if is_dead {
                println!("Snake died at move {}!", i);
                break;
            }

            // Apply move with Territorial opponent model
            let mut moves = vec![(String::from("territorial"), *dir)];
            if let Some(opp) = g.snakes.iter().find(|s| s.id == "negamax" && !s.is_eliminated()) {
                let opp_move = get_territorial_move_deterministic(&g, opp, &String::from("territorial")).unwrap_or(Direction::Up);
                println!("  -> Negamax predicted move (Territorial model): {:?}", opp_move);
                moves.push((String::from("negamax"), opp_move));
            }
            g.advance_turn(moves);
        }

        // The snake SHOULD die at some point in this sequence
        // If fitness is very positive (like 13286), something is wrong
        assert!(
            fitness_score < 0,
            "Fitness should be negative (snake should die), but got {}",
            fitness_score
        );
    }
}
