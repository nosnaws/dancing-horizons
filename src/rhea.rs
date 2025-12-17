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

pub fn fitness(game: &Game, player: &String, c: &Individual) -> i32 {
    fitness_with_opponent_modeling(game, player, c, true)
}

pub fn fitness_with_opponent_modeling(game: &Game, player: &String, c: &Individual, random_opponents: bool) -> i32 {
    let mut g = game.clone();
    let mut score = 0;
    let mut rng = SmallRng::from_entropy();

    for dir in &c.genotype {
        let mut moves: Vec<(String, Direction)> = vec![(player.clone(), dir.clone())];

        if random_opponents {
            // Generate random moves for all opponents
            for snake in &g.snakes {
                if snake.id != *player && !snake.is_eliminated() {
                    let opponent_move: Direction = rng.gen();
                    moves.push((snake.id.clone(), opponent_move));
                }
            }
        }

        g.advance_turn(moves);
        score += score_game(&g, &player);
    }

    return score;
}

pub fn score_game(g: &Game, player: &String) -> i32 {
    let mut score = 0;
    let mut snakes_alive = 0;

    for snake in &g.snakes {
        if snake.id == *player {
            if snake.is_eliminated() {
                return -1000;
            }
            // score += snake.health;
        }

        if !snake.is_eliminated() {
            snakes_alive += 1;
        }
    }

    if snakes_alive == 1 {
        // we're the only ones alive now
        score += 100;
    }

    return score;
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

    /// Simulates a head-to-head game where:
    /// - Player 1 uses random opponent modeling
    /// - Player 2 uses default (continue straight) opponent modeling
    /// Returns: (player1_won, player2_won, draw)
    fn simulate_head_to_head(max_turns: usize) -> (bool, bool, bool) {
        use rand::SeedableRng;
        let mut rng = SmallRng::from_entropy();

        // Create two snakes - closer together to encourage conflict
        let s1 = Snake::create(String::from("player1"), 100, vec![23, 12, 1]);
        let s2 = Snake::create(String::from("player2"), 100, vec![97, 108, 119]);

        // Add food in the middle to encourage conflict
        let food = vec![60, 59, 61, 49, 71, 48, 72, 37, 83];
        let mut game = Game::create(vec![s1, s2], food, 0, 11);

        for _ in 0..max_turns {
            let mut moves: Vec<(String, Direction)> = vec![];

            for snake in &game.snakes {
                if snake.is_eliminated() {
                    continue;
                }

                // Player 1 uses random opponent modeling, Player 2 uses default
                let use_random = snake.id == "player1";

                let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
                let mut best_dir = directions[rng.gen_range(0..4)];
                let mut best_score = i32::MIN;

                for dir in &directions {
                    let candidate = create_candidate(vec![*dir; 5]);
                    let score = fitness_with_opponent_modeling(&game, &snake.id, &candidate, use_random);
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

    #[test]
    fn compare_opponent_modeling_strategies() {
        const NUM_GAMES: usize = 50;
        const MAX_TURNS: usize = 500;

        println!("\n=== Head-to-Head: Random vs Default Opponent Modeling ===\n");
        println!("Player 1 (P1): Uses RANDOM opponent modeling");
        println!("Player 2 (P2): Uses DEFAULT (continue straight) opponent modeling\n");

        let mut p1_wins = 0;
        let mut p2_wins = 0;
        let mut draws = 0;

        for i in 0..NUM_GAMES {
            let (p1_won, p2_won, draw) = simulate_head_to_head(MAX_TURNS);
            if p1_won { p1_wins += 1; }
            if p2_won { p2_wins += 1; }
            if draw { draws += 1; }

            let result = if p1_won { "P1 (random)" } else if p2_won { "P2 (default)" } else { "Draw" };
            println!("Game {:2}: {}", i + 1, result);
        }

        println!("\n=== Results Summary ===");
        println!("P1 (random modeling) wins:  {} ({:.1}%)", p1_wins, (p1_wins as f64 / NUM_GAMES as f64) * 100.0);
        println!("P2 (default modeling) wins: {} ({:.1}%)", p2_wins, (p2_wins as f64 / NUM_GAMES as f64) * 100.0);
        println!("Draws:                      {} ({:.1}%)", draws, (draws as f64 / NUM_GAMES as f64) * 100.0);

        if p1_wins > p2_wins {
            println!("\nConclusion: Random opponent modeling appears BETTER");
        } else if p2_wins > p1_wins {
            println!("\nConclusion: Default opponent modeling appears BETTER");
        } else {
            println!("\nConclusion: No clear winner - strategies appear equal");
        }
    }
}
